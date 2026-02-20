"""Rich logging callback for training progress."""
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from rich.console import Console
from rich.table import Table
from pathlib import Path
import json
import time
import sys
import os


class RichMetricsLogger(Callback):
    """
    Callback for rich console output of training metrics.

    Uses rich library to display formatted tables with training progress.
    Also saves metrics to JSON file.
    """

    def __init__(self, save_dir=None):
        super().__init__()
        # Only use Rich console if running in TTY (interactive terminal)
        # In batch/submitit environments, disable Rich to avoid raw ANSI codes in output
        is_tty = sys.stdout.isatty() if hasattr(sys.stdout, 'isatty') else False
        self.use_rich = is_tty
        if self.use_rich:
            self.console = Console(width=120)
        else:
            # In batch mode, use null console that doesn't output anything
            self.console = Console(file=open(os.devnull, 'w'))
        self.best_val_acc = 0.0
        self.metrics_history = []
        self.save_dir = Path(save_dir) if save_dir else None
        self.epoch_start_time = None

    def _get_early_stopping_info(self, trainer):
        """Get early stopping callback info if available."""
        for callback in trainer.callbacks:
            if isinstance(callback, EarlyStopping):
                wait_count = callback.wait_count
                patience = callback.patience
                return wait_count, patience
        return None, None

    def _get_current_lr(self, trainer):
        """Get current learning rate from optimizer."""
        if trainer.optimizers:
            optimizer = trainer.optimizers[0]
            if hasattr(optimizer, 'param_groups') and optimizer.param_groups:
                return optimizer.param_groups[0]['lr']
        return None

    def on_train_start(self, trainer, pl_module):
        self.console.print("\n[bold green]Starting Training[/bold green]\n")

        table = Table(title="Configuration", show_header=True)
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="magenta")

        hparams = pl_module.hparams
        table.add_row("Model", "SPDNet")
        table.add_row("Hidden Layers", str(hparams['hidden_layers_size']))

        # Get batch size from train dataloader if available
        batch_size = "N/A"
        if trainer.train_dataloader is not None:
            batch_size = str(trainer.train_dataloader.batch_size)

        table.add_row("Batch Size", batch_size)
        table.add_row("Learning Rate", f"{hparams['optimizer']['lr']:.2e}")
        table.add_row("Max Epochs", str(trainer.max_epochs))
        table.add_row("Device", str(pl_module.device))

        self.console.print(table)
        self.console.print()
        self.epoch_start_time = time.time()  # Start timing first epoch

    def on_train_epoch_start(self, trainer, pl_module):
        """Record epoch start time."""
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        # Calculate epoch duration
        epoch_duration = time.time() - self.epoch_start_time if self.epoch_start_time else 0

        metrics = trainer.callback_metrics
        current_epoch = trainer.current_epoch

        val_acc = metrics.get('val/accuracy', 0)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            improvement = "NEW BEST"
        else:
            improvement = ""

        # Get early stopping info
        wait_count, patience = self._get_early_stopping_info(trainer)

        # Get current learning rate
        current_lr = self._get_current_lr(trainer)

        # Save metrics to history
        epoch_metrics = {
            'epoch': current_epoch + 1,
            'train_loss': float(metrics.get('train/loss', 0)),
            'train_accuracy': float(metrics.get('train/accuracy', 0)),
            'val_loss': float(metrics.get('val/loss', 0)),
            'val_accuracy': float(metrics.get('val/accuracy', 0)),
            'is_best': bool(improvement),
            'learning_rate': current_lr if current_lr else 0.0,
            'early_stopping_counter': wait_count if wait_count else 0,
            'epoch_duration_s': epoch_duration
        }

        self.metrics_history.append(epoch_metrics)

        # Print to stderr for submitit capture (flush immediately)
        # Format: [EPOCH N] time=Xs | train_acc=X val_acc=X | train_loss=X val_loss=X | LR=X | ES=X/Y | [NEW BEST]
        lr_str = f"{current_lr:.2e}" if current_lr is not None else "N/A"
        es_str = f"{wait_count}/{patience}" if wait_count is not None and patience is not None else "N/A"
        train_acc = float(metrics.get('train/accuracy', 0))
        val_acc = float(metrics.get('val/accuracy', 0))
        train_loss = float(metrics.get('train/loss', 0))
        val_loss = float(metrics.get('val/loss', 0))
        best_str = f" [{improvement}]" if improvement else ""

        log_line = f"[EPOCH {current_epoch+1:3d}] time={epoch_duration:6.1f}s | acc: train={train_acc:.4f} val={val_acc:.4f} | loss: train={train_loss:.4f} val={val_loss:.4f} | LR={lr_str} | ES={es_str}{best_str}"
        print(log_line, file=sys.stderr, flush=True)

        # Save to JSON file
        if self.save_dir:
            json_file = self.save_dir / 'metrics_per_epoch.json'
            with open(json_file, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)

        # Compact table without Info column
        table = Table(show_header=True, padding=(0, 1))
        table.add_column("Metric", style="cyan", justify="left", width=10)
        table.add_column("Train", style="green", justify="right", width=10)
        table.add_column("Val", style="yellow", justify="right", width=17)

        # Loss row
        table.add_row(
            "Loss",
            f"{metrics.get('train/loss', 0):.4f}",
            f"{metrics.get('val/loss', 0):.4f}"
        )

        # Accuracy row
        table.add_row(
            "Accuracy",
            f"{metrics.get('train/accuracy', 0):.4f}",
            f"{metrics.get('val/accuracy', 0):.4f}"
        )
        table.add_row(f"{'-'*10}", f"{'-'*10}", f"{'-'*17}")
        table.add_row(
            f"[bold green]{improvement}[/bold green]" if improvement else "",
            f"ES: {wait_count}/{patience}" if wait_count is not None and patience is not None else "",
            f"LR: {current_lr:.2e}" if current_lr is not None else ""
        )

        self.console.print(table)


    def on_train_end(self, trainer, pl_module):
        self.console.print("\n[bold green]Training Complete[/bold green]\n")

        table = Table(title="Final Results", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Best Value", style="magenta")

        table.add_row("Best Val Accuracy", f"{self.best_val_acc:.4f}")
        table.add_row("Total Epochs", str(trainer.current_epoch + 1))

        self.console.print(table)
        self.console.print()

        # Delete metrics JSON file since data is now in CSV
        if self.save_dir:
            json_file = self.save_dir / 'metrics_per_epoch.json'
            if json_file.exists():
                json_file.unlink()

        self.console.print(table)
