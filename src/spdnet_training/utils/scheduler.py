"""
Custom learning rate schedulers with warmup functionality.

This module provides learning rate schedulers that combine warmup phases
with different decay strategies commonly used in deep learning.
"""

import math
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, LRScheduler


class WarmupReduceLROnPlateau(LRScheduler):
    """
    Learning rate scheduler that combines warmup with ReduceLROnPlateau.

    During warmup phase, learning rate increases from initial LR to target LR
    using the specified warmup strategy. After warmup, ReduceLROnPlateau is used.

    This scheduler is compatible with PyTorch Lightning's LRScheduler API.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer to schedule
    warmup_epochs : int
        Number of warmup epochs
    target_lr : float, optional
        Target learning rate after warmup. If None, uses initial LR from optimizer
    warmup_type : str, default='linear'
        Type of warmup: 'linear', 'cosine', 'exponential', 'polynomial'
    warmup_power : float, default=2.0
        Power for polynomial warmup (only used when warmup_type='polynomial')
    mode : str, default='min'
        Mode for ReduceLROnPlateau
    factor : float, default=0.5
        Factor for ReduceLROnPlateau
    patience : int, default=10
        Patience for ReduceLROnPlateau
    min_lr : float, default=0.0
        Minimum learning rate
    **plateau_kwargs
        Additional kwargs for ReduceLROnPlateau
    """

    def __init__(
        self,
        optimizer,
        warmup_epochs,
        target_lr=None,
        warmup_type="linear",
        warmup_power=2.0,
        mode="min",
        factor=0.5,
        patience=10,
        min_lr=0.0,
        last_epoch=-1,
        **plateau_kwargs,
    ):
        self.warmup_epochs = warmup_epochs
        self.warmup_type = warmup_type
        self.warmup_power = warmup_power

        # Store initial learning rates before super().__init__
        self.initial_lrs = [group["lr"] for group in optimizer.param_groups]

        # Set target learning rates
        if target_lr is not None:
            self.target_lrs = [target_lr] * len(optimizer.param_groups)
        else:
            self.target_lrs = self.initial_lrs.copy()

        # Create ReduceLROnPlateau scheduler (will be used after warmup)
        self.plateau_scheduler = ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            min_lr=min_lr,
            **plateau_kwargs
        )

        # Track if we're in warmup or plateau phase
        self._in_warmup = warmup_epochs > 0
        self._metrics_received = False

        # Initialize with starting LR if warmup is used
        # Set the optimizer to the initial LR (before warmup starts)
        if warmup_epochs > 0:
            for param_group in optimizer.param_groups:
                param_group["lr"] = self.initial_lrs[0]

        # Don't call parent __init__ as it does an automatic step
        # Instead, manually set last_epoch
        self.optimizer = optimizer
        self.last_epoch = last_epoch  # Will be -1 by default
        self._last_lr = self.initial_lrs.copy()

    def _get_warmup_lr(self, epoch):
        """Calculate learning rate for warmup phase."""
        if self.warmup_epochs == 0:
            return self.target_lrs

        progress = min(epoch / self.warmup_epochs, 1.0)

        if self.warmup_type == "linear":
            warmup_factor = progress
        elif self.warmup_type == "cosine":
            warmup_factor = 0.5 * (1 + math.cos(math.pi * (1 - progress)))
        elif self.warmup_type == "exponential":
            warmup_factor = progress**2
        elif self.warmup_type == "polynomial":
            warmup_factor = progress**self.warmup_power
        else:
            raise ValueError(f"Unknown warmup_type: {self.warmup_type}")

        return [
            initial_lr + (target_lr - initial_lr) * warmup_factor
            for initial_lr, target_lr in zip(self.initial_lrs, self.target_lrs)
        ]

    def get_lr(self):
        """Return current learning rates."""
        # During warmup phase
        if self.last_epoch < self.warmup_epochs:
            return self._get_warmup_lr(self.last_epoch + 1)
        # After warmup, return current LRs (plateau handles its own updates)
        return [group['lr'] for group in self.optimizer.param_groups]

    def step(self, metrics=None):
        """Step the scheduler."""
        # Increment epoch counter
        self.last_epoch += 1

        if self.last_epoch < self.warmup_epochs:
            # Warmup phase
            lrs = self._get_warmup_lr(self.last_epoch + 1)
            for param_group, lr in zip(self.optimizer.param_groups, lrs):
                param_group["lr"] = lr
        elif self.last_epoch == self.warmup_epochs:
            # Transition from warmup to plateau: set target LR
            for param_group, target_lr in zip(self.optimizer.param_groups, self.target_lrs):
                param_group["lr"] = target_lr
            # If metrics provided, step plateau
            if metrics is not None:
                self.plateau_scheduler.step(metrics)
        else:
            # Plateau phase
            if metrics is not None:
                # old_lr = self.optimizer.param_groups[0]["lr"]
                self.plateau_scheduler.step(metrics)
                # new_lr = self.optimizer.param_groups[0]["lr"]
            #     print(f"[DEBUG] Plateau phase: metric={metrics:.4f}, LR {old_lr:.4e} -> {new_lr:.4e}, best={self.plateau_scheduler.best:.4f}, num_bad={self.plateau_scheduler.num_bad_epochs}")
            # else:
            #     print(f"[DEBUG] Plateau phase: NO METRICS RECEIVED!")

    def state_dict(self):
        """Return state of the scheduler."""
        state = super().state_dict()
        state.update({
            "initial_lrs": self.initial_lrs,
            "target_lrs": self.target_lrs,
            "plateau_scheduler": self.plateau_scheduler.state_dict(),
        })
        return state

    def load_state_dict(self, state_dict):
        """Load scheduler state."""
        self.initial_lrs = state_dict.pop("initial_lrs")
        self.target_lrs = state_dict.pop("target_lrs")
        plateau_state = state_dict.pop("plateau_scheduler")
        super().load_state_dict(state_dict)
        self.plateau_scheduler.load_state_dict(plateau_state)
