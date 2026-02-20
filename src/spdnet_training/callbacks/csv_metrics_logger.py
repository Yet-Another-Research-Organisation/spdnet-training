"""Custom CSV metrics logger with clean format."""
from pytorch_lightning.callbacks import Callback
from pathlib import Path
import csv


class CleanCSVMetricsLogger(Callback):
    """
    Custom CSV logger that writes clean metrics format:
    - One row per epoch
    - Train metrics (loss, accuracy, precision, recall, f1) in first columns
    - Val metrics in middle columns
    - Test metrics (if available) in last columns
    """

    def __init__(self, save_dir: Path):
        super().__init__()
        self.save_dir = Path(save_dir)
        self.csv_file = self.save_dir / 'metrics.csv'
        self.epoch_metrics = []
        self.test_metrics = None

        # Define column order
        self.train_cols = ['epoch', 'train_loss', 'train_accuracy', 'train_precision', 'train_recall', 'train_f1']
        self.val_cols = ['val_loss', 'val_accuracy', 'val_precision', 'val_recall', 'val_f1']
        self.test_cols = ['test_loss', 'test_accuracy', 'test_precision', 'test_recall', 'test_f1']

    def on_train_epoch_end(self, trainer, pl_module):
        """Collect metrics at end of each epoch."""
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch

        # Extract train and val metrics
        row = {
            'epoch': epoch,
            'train_loss': float(metrics.get('train/loss', 0)),
            'train_accuracy': float(metrics.get('train/accuracy', 0)),
            'train_precision': float(metrics.get('train/precision', 0)),
            'train_recall': float(metrics.get('train/recall', 0)),
            'train_f1': float(metrics.get('train/f1', 0)),
            'val_loss': float(metrics.get('val/loss', 0)),
            'val_accuracy': float(metrics.get('val/accuracy', 0)),
            'val_precision': float(metrics.get('val/precision', 0)),
            'val_recall': float(metrics.get('val/recall', 0)),
            'val_f1': float(metrics.get('val/f1', 0)),
        }

        self.epoch_metrics.append(row)

    def on_train_end(self, trainer, pl_module):
        """Write CSV at end of training (before test phase)."""
        self._write_csv()

    def on_test_end(self, trainer, pl_module):
        """Collect test metrics."""
        metrics = trainer.callback_metrics

        self.test_metrics = {
            'test_loss': float(metrics.get('test/loss', 0)),
            'test_accuracy': float(metrics.get('test/accuracy', 0)),
            'test_precision': float(metrics.get('test/precision', 0)),
            'test_recall': float(metrics.get('test/recall', 0)),
            'test_f1': float(metrics.get('test/f1', 0)),
        }

        # Write all metrics to CSV (with test metrics this time)
        self._write_csv()

    def _write_csv(self):
        """Write clean CSV file."""
        if not self.epoch_metrics:
            return

        # Determine columns
        all_cols = self.train_cols + self.val_cols
        if self.test_metrics:
            all_cols += self.test_cols

        # Write CSV
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_cols)
            writer.writeheader()

            # Write epoch metrics
            for row in self.epoch_metrics:
                # Add test metrics to last row if available
                if self.test_metrics and row == self.epoch_metrics[-1]:
                    row.update(self.test_metrics)
                writer.writerow(row)
