"""Covariance analysis callback for per-class test set analysis."""

from pytorch_lightning.callbacks import Callback
import torch
from pathlib import Path
from typing import Optional, Dict, Any
import json
import numpy as np
import matplotlib.pyplot as plt


NSAMPLES = 1


def setup_matplotlib(serif=False):
    """Setup matplotlib with better defaults for LaTeX-like rendering"""
    if serif:
        plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["EB Garamond", "Times New Roman", "serif"],
            "text.usetex": False,  # Disable LaTeX rendering to use system fonts
            "mathtext.fontset": "dejavusans",  # For math symbols
        }
    )
    else:
        plt.rcParams.update(
        {
            "font.family": "sans-serif",  # Use sans-serif family
            "font.sans-serif": ["Fira Sans", "DejaVu Sans", "Arial"],  # Fira Sans as priority
            "text.usetex": False,  # Disable LaTeX rendering to use system fonts
            "mathtext.fontset": "dejavusans",  # For math symbols
        }
    )
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams['figure.figsize'] = (15, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['legend.fontsize'] = 10


class CovarianceAnalysisCallback(Callback):
    """
    Callback for analyzing covariance matrices per class on test set.

    Analyzes SPD matrices at two key points:
    1. Network input (test set covariances)
    2. Before LogEig operation (output of SPDNet layers)

    For each class, computes statistics over all test samples:
    - Mean min/max values in covariance matrices
    - Mean min/max eigenvalues
    - Mean condition number

    Args:
        save_dir: Directory to save analysis results
        num_samples_per_class: Number of sample covariances to visualize per class
    """

    def __init__(
        self,
        save_dir: Optional[Path] = None,
        num_samples_per_class: int = 3,
    ):
        super().__init__()
        self.save_dir = Path(save_dir) if save_dir else None
        self.num_samples_per_class = num_samples_per_class

        self.hook_handles = []
        self.captured_before_logeig = None

        # Store per-class data during test
        self.input_covs_per_class = {}  # class_id -> list of input covariance matrices
        self.output_covs_per_class = {}  # class_id -> list of output matrices before logeig

    def on_test_start(self, trainer, pl_module):
        """Setup hooks before test."""
        if self.save_dir is None:
            self.save_dir = Path(trainer.log_dir) / 'covariance_analysis'
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Register hooks to capture matrices before LogEig
        self._register_hooks(pl_module)

        # Initialize storage
        num_classes = pl_module.hparams.output_dim
        for class_id in range(num_classes):
            self.input_covs_per_class[class_id] = []
            self.output_covs_per_class[class_id] = []

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Collect covariances per class during test."""
        x, y = batch

        pl_module.eval()
        with torch.no_grad():
            # Store input covariances per class
            for class_id in torch.unique(y):
                class_id = class_id.item()
                class_mask = y == class_id
                class_inputs = x[class_mask]

                if class_inputs.shape[0] > 0:
                    self.input_covs_per_class[class_id].append(class_inputs.detach().cpu())

            # Forward pass to capture output before LogEig
            self.captured_before_logeig = None
            _ = pl_module(x)

            # Store output covariances per class
            if self.captured_before_logeig is not None:
                for class_id in torch.unique(y):
                    class_id = class_id.item()
                    class_mask = y == class_id
                    class_outputs = self.captured_before_logeig[class_mask]

                    if class_outputs.shape[0] > 0:
                        self.output_covs_per_class[class_id].append(class_outputs.detach().cpu())

    def on_test_end(self, trainer, pl_module):
        """Analyze collected covariances and save results."""
        # Concatenate all batches per class
        for class_id in self.input_covs_per_class.keys():
            if self.input_covs_per_class[class_id]:
                self.input_covs_per_class[class_id] = torch.cat(self.input_covs_per_class[class_id], dim=0)
            else:
                self.input_covs_per_class[class_id] = None

            if self.output_covs_per_class[class_id]:
                self.output_covs_per_class[class_id] = torch.cat(self.output_covs_per_class[class_id], dim=0)
            else:
                self.output_covs_per_class[class_id] = None

        # Compute statistics per class
        stats = self._compute_per_class_stats()

        # Save statistics to JSON
        stats_file = self.save_dir / 'covariance_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        # Generate covariance matrix figures per class
        self._generate_covariance_figures()

        # Remove hooks
        for handle in self.hook_handles:
            handle.remove()

    def _register_hooks(self, pl_module):
        """Register forward hook to capture matrices before LogEig."""

        def hook_before_logeig(module, input, output):
            # Capture input to LogEig (output of previous layer)
            if isinstance(input, tuple) and len(input) > 0:
                self.captured_before_logeig = input[0].detach()
            elif isinstance(input, torch.Tensor):
                self.captured_before_logeig = input.detach()

        # Hook into SPDNet layers - specifically LogEig
        model = pl_module.model if hasattr(pl_module, 'model') else pl_module

        for name, module in model.named_modules():
            module_type = module.__class__.__name__

            # Hook the LogEig layer to capture input
            if 'LogEig' in module_type:
                handle = module.register_forward_hook(hook_before_logeig)
                self.hook_handles.append(handle)
                break  # Only hook the first LogEig we find

    def _compute_per_class_stats(self) -> Dict[str, Any]:
        """
        Compute statistics per class for input and output covariances.

        Returns dict with structure:
        {
            "class_0": {
                "input": {
                    "mean_min_matrix_value": float,
                    "mean_max_matrix_value": float,
                    "mean_min_eigenvalue": float,
                    "mean_max_eigenvalue": float,
                    "mean_condition_number": float
                },
                "output_before_logeig": { ... same structure ... }
            },
            ...
        }
        """
        stats = {}

        for class_id in sorted(self.input_covs_per_class.keys()):
            class_stats = {}

            # Input covariance statistics
            input_covs = self.input_covs_per_class[class_id]
            if input_covs is not None and input_covs.shape[0] > 0:
                class_stats["input"] = self._compute_covariance_stats(input_covs)
            else:
                class_stats["input"] = None

            # Output covariance statistics (before LogEig)
            output_covs = self.output_covs_per_class[class_id]
            if output_covs is not None and output_covs.shape[0] > 0:
                class_stats["output_before_logeig"] = self._compute_covariance_stats(output_covs)
            else:
                class_stats["output_before_logeig"] = None

            stats[f"class_{class_id}"] = class_stats

        return stats

    def _compute_covariance_stats(self, cov_matrices: torch.Tensor) -> Dict[str, float]:
        """
        Compute statistics for a batch of covariance matrices.

        Args:
            cov_matrices: Tensor of shape [N, C, C] where N is number of samples

        Returns:
            Dictionary with mean statistics across all samples
        """
        if cov_matrices.dim() != 3:
            return {}

        # Compute min/max matrix values per sample
        min_vals = cov_matrices.amin(dim=(1, 2))  # [N]
        max_vals = cov_matrices.amax(dim=(1, 2))  # [N]

        # Compute eigenvalues per sample
        eigenvalues = torch.linalg.eigvalsh(cov_matrices.double())  # [N, C]
        min_eigs = eigenvalues[:, 0]  # [N]
        max_eigs = eigenvalues[:, -1]  # [N]

        # Compute condition numbers
        condition_numbers = max_eigs / (min_eigs + 1e-10)  # [N]

        # Return mean statistics across all samples
        return {
            "mean_min_matrix_value": float(min_vals.mean()),
            "mean_max_matrix_value": float(max_vals.mean()),
            "mean_min_eigenvalue": float(min_eigs.mean()),
            "mean_max_eigenvalue": float(max_eigs.mean()),
            "mean_condition_number": float(condition_numbers.mean())
        }

    def _generate_covariance_figures(self):
        """Generate combined covariance matrix figures: 4 figures with 4 subplots each."""

        # Setup matplotlib with LaTeX-like fonts for PDF output
        setup_matplotlib(serif=True)

        # Create subdirectories for figures
        input_fig_dir = self.save_dir / 'input_covariance'
        output_fig_dir = self.save_dir / 'output_covariance'
        input_fig_dir.mkdir(exist_ok=True)
        output_fig_dir.mkdir(exist_ok=True)

        # Compute global color scale (5% and 95% percentiles)
        all_input_vals = []
        all_output_vals = []

        for class_id in sorted(self.input_covs_per_class.keys()):
            if self.input_covs_per_class[class_id] is not None:
                all_input_vals.extend(self.input_covs_per_class[class_id].flatten().tolist())
            if self.output_covs_per_class[class_id] is not None:
                all_output_vals.extend(self.output_covs_per_class[class_id].flatten().tolist())

        vmin_input = np.percentile(all_input_vals, 5) if all_input_vals else 0
        vmax_input = np.percentile(all_input_vals, 95) if all_input_vals else 1
        vmin_output = np.percentile(all_output_vals, 5) if all_output_vals else 0
        vmax_output = np.percentile(all_output_vals, 95) if all_output_vals else 1

        # Select first 4 classes for combined figures
        sorted_class_indices = sorted(self.input_covs_per_class.keys())[:4]

        # ========================================================================
        # INPUT COVARIANCE FIGURES (2 combined figures for first 4 classes)
        # ========================================================================
        if len(sorted_class_indices) > 0:
            # Check if we have data for at least one class
            has_input_data = any(
                self.input_covs_per_class[class_id] is not None and
                self.input_covs_per_class[class_id].shape[0] > 0
                for class_id in sorted_class_indices
            )

            if has_input_data:
                # Figure 1: Input covariance matrices heatmaps (4 subplots)
                fig1, axes1_raw = plt.subplots(1, 4, figsize=(12, 3), sharex=True, sharey=True)
                axes1 = np.atleast_1d(axes1_raw).tolist()

                im = None
                for idx, class_id in enumerate(sorted_class_indices):
                    if self.input_covs_per_class[class_id] is not None and \
                       self.input_covs_per_class[class_id].shape[0] > 0:
                        mat = self.input_covs_per_class[class_id][0].numpy()
                        im = axes1[idx].imshow(mat, cmap='gist_rainbow', aspect='equal',
                                              vmin=vmin_input, vmax=vmax_input)
                        axes1[idx].set_title(f'Classe {class_id}')
                        axes1[idx].tick_params(labelsize=8)
                    else:
                        axes1[idx].axis('off')

                # Hide unused subplots
                for idx in range(len(sorted_class_indices), 4):
                    axes1[idx].axis('off')

                # Add colorbar
                if im is not None:
                    plt.tight_layout()
                    fig1.subplots_adjust(right=0.92)
                    cbar_ax = fig1.add_axes([0.94, 0.15, 0.02, 0.7])
                    fig1.colorbar(im, cax=cbar_ax)

                fig1.savefig(input_fig_dir / 'combined_matrices.pdf',
                            dpi=300, bbox_inches='tight')
                plt.close(fig1)

                # Figure 2: Input eigenvalues (4 subplots with shared y-axis)
                fig2, axes2_raw = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
                axes2 = np.atleast_1d(axes2_raw).tolist()

                for idx, class_id in enumerate(sorted_class_indices):
                    if self.input_covs_per_class[class_id] is not None and \
                       self.input_covs_per_class[class_id].shape[0] > 0:
                        mat = self.input_covs_per_class[class_id][0].numpy()
                        eigvals = np.linalg.eigvalsh(mat)
                        eigvals_sorted = np.sort(eigvals)[::-1]
                        axes2[idx].semilogy(eigvals_sorted, 'bo-', markersize=3, linewidth=1)
                        axes2[idx].set_title(f'Classe {class_id}')
                        axes2[idx].set_xlabel('Index')
                        if idx == 0:
                            axes2[idx].set_ylabel('Eigenvalue (log)')
                        axes2[idx].grid(True, alpha=0.3)
                    else:
                        axes2[idx].axis('off')

                # Hide unused subplots
                for idx in range(len(sorted_class_indices), 4):
                    axes2[idx].axis('off')

                plt.tight_layout()
                fig2.savefig(input_fig_dir / 'combined_eigenvalues.pdf',
                            dpi=300, bbox_inches='tight')
                plt.close(fig2)

        # ========================================================================
        # OUTPUT COVARIANCE FIGURES (2 combined figures for first 4 classes)
        # ========================================================================
        if len(sorted_class_indices) > 0:
            # Check if we have data for at least one class
            has_output_data = any(
                self.output_covs_per_class[class_id] is not None and
                self.output_covs_per_class[class_id].shape[0] > 0
                for class_id in sorted_class_indices
            )

            if has_output_data:
                # Figure 3: Output covariance matrices heatmaps (4 subplots)
                fig3, axes3_raw = plt.subplots(1, 4, figsize=(12, 3), sharex=True, sharey=True)
                axes3 = np.atleast_1d(axes3_raw).tolist()

                im = None
                for idx, class_id in enumerate(sorted_class_indices):
                    if self.output_covs_per_class[class_id] is not None and \
                       self.output_covs_per_class[class_id].shape[0] > 0:
                        mat = self.output_covs_per_class[class_id][0].numpy()
                        im = axes3[idx].imshow(mat, cmap='gist_rainbow', aspect='equal',
                                              vmin=vmin_output, vmax=vmax_output)
                        axes3[idx].set_title(f'Classe {class_id}')
                        axes3[idx].tick_params(labelsize=8)
                    else:
                        axes3[idx].axis('off')

                # Hide unused subplots
                for idx in range(len(sorted_class_indices), 4):
                    axes3[idx].axis('off')

                # Add colorbar
                if im is not None:
                    plt.tight_layout()
                    fig3.subplots_adjust(right=0.92)
                    cbar_ax = fig3.add_axes([0.94, 0.15, 0.02, 0.7])
                    fig3.colorbar(im, cax=cbar_ax)

                fig3.savefig(output_fig_dir / 'combined_matrices.pdf',
                            dpi=300, bbox_inches='tight')
                plt.close(fig3)

                # Figure 4: Output eigenvalues (4 subplots with shared y-axis)
                fig4, axes4_raw = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
                axes4 = np.atleast_1d(axes4_raw).tolist()

                for idx, class_id in enumerate(sorted_class_indices):
                    if self.output_covs_per_class[class_id] is not None and \
                       self.output_covs_per_class[class_id].shape[0] > 0:
                        mat = self.output_covs_per_class[class_id][0].numpy()
                        eigvals = np.linalg.eigvalsh(mat)
                        eigvals_sorted = np.sort(eigvals)[::-1]
                        axes4[idx].semilogy(eigvals_sorted, 'ro-', markersize=3, linewidth=1)
                        axes4[idx].set_title(f'Classe {class_id}')
                        axes4[idx].set_xlabel('Index')
                        if idx == 0:
                            axes4[idx].set_ylabel('Eigenvalue (log)')
                        axes4[idx].grid(True, alpha=0.3)
                    else:
                        axes4[idx].axis('off')

                # Hide unused subplots
                for idx in range(len(sorted_class_indices), 4):
                    axes4[idx].axis('off')

                plt.tight_layout()
                fig4.savefig(output_fig_dir / 'combined_eigenvalues.pdf',
                            dpi=300, bbox_inches='tight')
                plt.close(fig4)
