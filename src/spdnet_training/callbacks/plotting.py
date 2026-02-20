"""Plotting callback for training visualization."""
from pytorch_lightning.callbacks import Callback
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
import numpy as np


class PlottingCallback(Callback):
    """
    Callback for generating training plots.

    Args:
        save_dir: Directory to save plots
        plot_every_n_epochs: Generate intermediate plots every N epochs
    """

    def __init__(self, save_dir: Path = None, plot_every_n_epochs: int = 10):
        super().__init__()
        self.save_dir = Path(save_dir) if save_dir else None
        self.plot_every_n_epochs = plot_every_n_epochs

        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.learning_rates = []

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics

        # Helper to convert metric to float
        def to_float(val):
            if torch.is_tensor(val):
                return val.item()
            return float(val) if val is not None else 0.0

        self.train_losses.append(to_float(metrics.get('train/loss', 0)))
        self.train_accs.append(to_float(metrics.get('train/accuracy', 0)))
        self.val_losses.append(to_float(metrics.get('val/loss', 0)))
        self.val_accs.append(to_float(metrics.get('val/accuracy', 0)))

        if trainer.optimizers:
            lr = trainer.optimizers[0].param_groups[0]['lr']
            self.learning_rates.append(lr)

    def on_train_end(self, trainer, pl_module):
        if self.save_dir is None:
            self.save_dir = Path(trainer.log_dir) / 'plots'

        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Create triple subplot figure: accuracy, loss, learning rate
        self._plot_training_curves()

    def on_test_end(self, trainer, pl_module):
        if hasattr(pl_module, 'test_confusion_matrix'):
            # Get per-class F1 scores from test metrics
            from sklearn.metrics import f1_score
            from spdnet_training.callbacks.results_saver import ResultsSaver

            # Find ResultsSaver callback to get predictions and targets
            results_saver = None
            for callback in trainer.callbacks:
                if isinstance(callback, ResultsSaver):
                    results_saver = callback
                    break

            if results_saver and results_saver.test_preds and results_saver.test_targets:
                preds = torch.cat(results_saver.test_preds).numpy()
                targets = torch.cat(results_saver.test_targets).numpy()

                # Calculate per-class F1 scores
                num_classes = pl_module.hparams.output_dim
                f1_per_class = f1_score(targets, preds, labels=range(num_classes), average=None, zero_division=0)

                self._plot_confusion_and_f1(
                    pl_module.test_confusion_matrix,
                    f1_per_class,
                    num_classes=num_classes
                )
            else:
                # Fallback: plot only confusion matrix if F1 data not available
                self._plot_confusion_matrix(
                    pl_module.test_confusion_matrix,
                    num_classes=pl_module.hparams.output_dim
                )

    def _plot_training_curves(self):
        """Create triple subplot with accuracy, loss, and learning rate."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        epochs = range(1, len(self.train_losses) + 1)

        # Subplot 1: Accuracy
        axes[0].plot(epochs, self.train_accs, 'b-', label='Train', linewidth=2, marker='o', markersize=4)
        axes[0].plot(epochs, self.val_accs, 'r-', label='Val', linewidth=2, marker='s', markersize=4)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Accuracy', fontsize=12)
        axes[0].set_title('Accuracy Curves', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([0, 1])

        # Subplot 2: Loss
        axes[1].plot(epochs, self.train_losses, 'b-', label='Train', linewidth=2, marker='o', markersize=4)
        axes[1].plot(epochs, self.val_losses, 'r-', label='Val', linewidth=2, marker='s', markersize=4)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Loss', fontsize=12)
        axes[1].set_title('Loss Curves', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)

        # Subplot 3: Learning Rate
        if self.learning_rates:
            axes[2].plot(epochs, self.learning_rates, 'g-', linewidth=2, marker='d', markersize=4)
            axes[2].set_xlabel('Epoch', fontsize=12)
            axes[2].set_ylabel('Learning Rate', fontsize=12)
            axes[2].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            axes[2].set_yscale('log')
            axes[2].grid(True, alpha=0.3)
        else:
            axes[2].text(0.5, 0.5, 'No LR data', ha='center', va='center', fontsize=14)
            axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_confusion_and_f1(self, cm: torch.Tensor, f1_per_class: np.ndarray, num_classes: int):
        """Create double subplot with confusion matrix and F1 score per class bar chart."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Subplot 1: Confusion Matrix
        cm_np = cm.cpu().numpy()
        # Avoid division by zero by replacing zero sums with 1
        row_sums = cm_np.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_norm = cm_np / row_sums

        sns.heatmap(
            cm_norm,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            ax=axes[0],
            cbar_kws={'label': 'Proportion'},
            square=True
        )

        axes[0].set_xlabel('Predicted Class', fontsize=12)
        axes[0].set_ylabel('True Class', fontsize=12)
        axes[0].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')

        # Subplot 2: F1 Score per Class
        class_labels = [str(i) for i in range(num_classes)]
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, num_classes))

        bars = axes[1].bar(class_labels, f1_per_class, color=colors, edgecolor='black', linewidth=1.2)
        axes[1].set_xlabel('Class', fontsize=12)
        axes[1].set_ylabel('F1 Score', fontsize=12)
        axes[1].set_title('F1 Score per Class', fontsize=14, fontweight='bold')
        axes[1].set_ylim([0, 1])
        axes[1].grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, score in zip(bars, f1_per_class):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{score:.3f}',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.save_dir / 'test_results.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_confusion_matrix(self, cm: torch.Tensor, num_classes: int):
        """Fallback: plot only confusion matrix."""
        fig, ax = plt.subplots(figsize=(10, 8))

        cm_np = cm.cpu().numpy()
        cm_norm = cm_np / cm_np.sum(axis=1, keepdims=True)

        sns.heatmap(
            cm_norm,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            ax=ax,
            cbar_kws={'label': 'Proportion'},
            square=True
        )

        ax.set_xlabel('Predicted Class', fontsize=12)
        ax.set_ylabel('True Class', fontsize=12)
        ax.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.save_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_covariance_stats(self, stats_history: list):
        """Plot covariance statistics over training."""
        if not stats_history:
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        epochs = [s['epoch'] for s in stats_history]

        # Min/Max eigenvalues
        if 'min_eigenvalue' in stats_history[0]:
            min_eigs = [s['min_eigenvalue'] for s in stats_history]
            max_eigs = [s['max_eigenvalue'] for s in stats_history]
            axes[0, 0].semilogy(epochs, min_eigs, 'b-', label='Min')
            axes[0, 0].semilogy(epochs, max_eigs, 'r-', label='Max')
            axes[0, 0].set_title('Eigenvalue Range', fontweight='bold')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Eigenvalue')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

        # Condition number
        if 'mean_condition_number' in stats_history[0]:
            cond_mean = [s['mean_condition_number'] for s in stats_history]
            cond_max = [s['max_condition_number'] for s in stats_history]
            axes[0, 1].semilogy(epochs, cond_mean, 'g-', label='Mean')
            axes[0, 1].semilogy(epochs, cond_max, 'orange', label='Max')
            axes[0, 1].set_title('Condition Number', fontweight='bold')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Condition Number')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        # Determinant
        if 'mean_determinant' in stats_history[0]:
            det_mean = [s['mean_determinant'] for s in stats_history]
            axes[0, 2].semilogy(epochs, np.abs(det_mean), 'm-')
            axes[0, 2].set_title('Mean Determinant (abs)', fontweight='bold')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('Determinant')
            axes[0, 2].grid(True, alpha=0.3)

        # Trace
        if 'mean_trace' in stats_history[0]:
            trace_mean = [s['mean_trace'] for s in stats_history]
            axes[1, 0].plot(epochs, trace_mean, 'c-')
            axes[1, 0].set_title('Mean Trace', fontweight='bold')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Trace')
            axes[1, 0].grid(True, alpha=0.3)

        # Frobenius norm
        if 'mean_frobenius_norm' in stats_history[0]:
            frob_mean = [s['mean_frobenius_norm'] for s in stats_history]
            axes[1, 1].plot(epochs, frob_mean, 'y-')
            axes[1, 1].set_title('Mean Frobenius Norm', fontweight='bold')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Norm')
            axes[1, 1].grid(True, alpha=0.3)

        # Eigenvalue statistics
        if 'mean_eigenvalue' in stats_history[0]:
            eig_mean = [s['mean_eigenvalue'] for s in stats_history]
            eig_std = [s.get('std_eigenvalue', 0) for s in stats_history]
            axes[1, 2].plot(epochs, eig_mean, 'purple', label='Mean')
            axes[1, 2].fill_between(
                epochs,
                np.array(eig_mean) - np.array(eig_std),
                np.array(eig_mean) + np.array(eig_std),
                alpha=0.3, color='purple', label='±1 std'
            )
            axes[1, 2].set_title('Eigenvalue Statistics', fontweight='bold')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Eigenvalue')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)

        fig.suptitle('Covariance Matrix Statistics', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.save_dir / 'covariance_stats.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_sample_covariance_matrices(self, cov_matrices: torch.Tensor,
                                       title: str = "Sample Covariance Matrices",
                                       n_samples: int = 9):
        """
        Plot sample covariance matrices as heatmaps.

        Args:
            cov_matrices: Batch of covariance matrices [B, C, C]
            title: Plot title
            n_samples: Number of samples to plot
        """
        n_samples = min(n_samples, cov_matrices.shape[0])

        ncols = 3
        nrows = (n_samples + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
        if nrows == 1 and ncols == 1:
            axes = np.array([[axes]])
        elif nrows == 1:
            axes = axes.reshape(1, -1)
        elif ncols == 1:
            axes = axes.reshape(-1, 1)

        for idx in range(n_samples):
            row = idx // ncols
            col = idx % ncols
            ax = axes[row, col]

            cov = cov_matrices[idx].cpu().numpy()

            # Normalize for visualization
            vmax = np.abs(cov).max()

            im = ax.imshow(cov, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
            ax.set_title(f'Sample {idx+1}', fontsize=10)
            ax.set_xlabel('Feature')
            ax.set_ylabel('Feature')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Hide empty subplots
        for idx in range(n_samples, nrows * ncols):
            row = idx // ncols
            col = idx % ncols
            axes[row, col].axis('off')

        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()

        filename = title.lower().replace(' ', '_') + '.png'
        plt.savefig(self.save_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
