"""Optuna pruning callback for PyTorch Lightning.

Reports intermediate validation metrics to an Optuna trial so that
unpromising trials can be pruned early (MedianPruner, etc.).
"""

import optuna
import pytorch_lightning as pl


class OptunaPruningCallback(pl.Callback):
    """Report intermediate metrics to Optuna and handle pruning.

    Parameters
    ----------
    trial : optuna.Trial
        The Optuna trial being evaluated.
    monitor : str
        The metric key to report (must appear in ``trainer.callback_metrics``).
    """

    def __init__(self, trial: optuna.Trial, monitor: str = "val/loss"):
        super().__init__()
        self.trial = trial
        self.monitor = monitor

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        epoch = trainer.current_epoch
        current_value = trainer.callback_metrics.get(self.monitor)

        if current_value is None:
            return

        self.trial.report(float(current_value), step=epoch)

        if self.trial.should_prune():
            raise optuna.TrialPruned(
                f"Trial pruned at epoch {epoch} with {self.monitor}={float(current_value):.4f}"
            )
