"""SPDNet Lightning Module."""
import torch
import pytorch_lightning as pl
from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix
from typing import Dict, Any, Optional


class SPDNetModule(pl.LightningModule):
    """
    Lightning Module for SPDNet model.

    This module accepts all SPDNet parameters and passes them dynamically
    to the underlying model. This allows for flexible configuration without
    needing to update the module when SPDNet parameters change.

    Args:
        output_dim: Number of output classes (required)
        optimizer: Optimizer configuration dict
        scheduler: Scheduler configuration dict (optional)
        **model_kwargs: All other arguments are passed to SPDnet
    """

    def __init__(
        self,
        output_dim: int,
        optimizer: Dict[str, Any] = None,
        scheduler: Optional[Dict[str, Any]] = None,
        **model_kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        from yetanotherspdnet.model import SPDnet

        # Extract known non-SPDnet parameters
        non_model_params = {'optimizer', 'scheduler', 'input_channels', 'name', 'batchnorm_adaptive_mean_type', 'dropout_rate'}  # input_channels is for CNN, not SPDnet

        # Prepare model kwargs - filter out optimizer/scheduler/input_channels
        spdnet_kwargs = {k: v for k, v in model_kwargs.items() if k not in non_model_params}

        # Add output_dim
        spdnet_kwargs['output_dim'] = output_dim

        # Handle backward compatibility for renamed parameters
        # Map common parameter names to SPDnet expected names (v.0.2.0)
        param_mapping = {
            'eps': 'reeig_eps',  # eps -> reeig_eps
            'batchnorm_method': 'batchnorm_mean_type',
        }

        for old_name, new_name in param_mapping.items():
            if old_name in spdnet_kwargs and new_name not in spdnet_kwargs:
                spdnet_kwargs[new_name] = spdnet_kwargs.pop(old_name)
            elif old_name in spdnet_kwargs:
                spdnet_kwargs.pop(old_name)  # Remove duplicate if new_name already present

        # Handle use_vech -> vec_type conversion
        if 'use_vech' in spdnet_kwargs:
            spdnet_kwargs['vec_type'] = 'vech' if spdnet_kwargs.pop('use_vech') else 'vec'

        # Remove legacy bimap parametrization params (v.0.2.0 uses bimap_parametrization_mode: 'static'/'dynamic')
        # Old keys: bimap_parametrization_name, bimap_parametrization → no longer accepted by SPDnet
        for legacy_key in ('bimap_parametrization_name', 'bimap_parametrization'):
            spdnet_kwargs.pop(legacy_key, None)

        # Remove batchnorm_minibatch_momentum if present (renamed to batchnorm_momentum in v.0.2.0)
        if 'batchnorm_minibatch_momentum' in spdnet_kwargs and 'batchnorm_momentum' not in spdnet_kwargs:
            spdnet_kwargs['batchnorm_momentum'] = spdnet_kwargs.pop('batchnorm_minibatch_momentum')
        elif 'batchnorm_minibatch_momentum' in spdnet_kwargs:
            spdnet_kwargs.pop('batchnorm_minibatch_momentum')

        # Set device and dtype if not already specified
        if 'device' not in spdnet_kwargs:
            spdnet_kwargs['device'] = self.device
        if 'dtype' not in spdnet_kwargs:
            spdnet_kwargs['dtype'] = torch.float64

        # Create SPDnet model with all parameters
        try:
            self.model = SPDnet(**spdnet_kwargs)
        except TypeError as e:
            print(f"Error creating SPDnet with parameters: {spdnet_kwargs.keys()}")
            print(f"Error: {e}")
            raise

        # Compile model for faster execution (PyTorch 2.0+)
        # Note: Disabled by default as SPDNet uses custom operations that may not benefit
        # Uncomment to test: self.model = torch.compile(self.model, mode='reduce-overhead')

        self.criterion = torch.nn.CrossEntropyLoss()

        metric_kwargs = {'task': 'multiclass', 'num_classes': output_dim}

        self.train_metrics = torch.nn.ModuleDict({
            'accuracy': Accuracy(**metric_kwargs),
            'precision': Precision(**metric_kwargs, average='macro'),
            'recall': Recall(**metric_kwargs, average='macro'),
            'f1': F1Score(**metric_kwargs, average='macro'),
        })

        self.val_metrics = torch.nn.ModuleDict({
            'accuracy': Accuracy(**metric_kwargs),
            'precision': Precision(**metric_kwargs, average='macro'),
            'recall': Recall(**metric_kwargs, average='macro'),
            'f1': F1Score(**metric_kwargs, average='macro'),
            'confusion_matrix': ConfusionMatrix(**metric_kwargs),
        })

        self.test_metrics = torch.nn.ModuleDict({
            'accuracy': Accuracy(**metric_kwargs),
            'precision': Precision(**metric_kwargs, average='macro'),
            'recall': Recall(**metric_kwargs, average='macro'),
            'f1': F1Score(**metric_kwargs, average='macro'),
            'confusion_matrix': ConfusionMatrix(**metric_kwargs),
        })

    def _get_parametrization(self, name: str):
        """Get parametrization from name."""
        if name == 'orthogonal':
            from torch.nn.utils import parametrizations
            return parametrizations.orthogonal
        elif name == 'StiefelProjectionQRParametrization':
            from yetanotherspdnet.nn.base import StiefelProjectionQRParametrization
            return StiefelProjectionQRParametrization
        else:
            raise ValueError(f"Unknown parametrization: {name}")

    def forward(self, x):
        # Data should already be in double precision from dataloader
        # Only convert if needed (avoid redundant conversions)
        if x.dtype != torch.float64:
            x = x.double()
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        for name, metric in self.train_metrics.items():
            metric(preds, y)

        # Log loss with sync_dist for proper averaging across batches
        self.log('train/loss', loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('train/accuracy', self.train_metrics['accuracy'], prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def on_train_epoch_end(self):
        # Log all other metrics after epoch (not in progress bar)
        for name, metric in self.train_metrics.items():
            if name not in ['accuracy']:  # accuracy already logged in training_step
                value = metric.compute()
                self.log(f'train/{name}', value, prog_bar=False)
            metric.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        for name, metric in self.val_metrics.items():
            metric(preds, y)

        # Validation metrics shown after epoch completes
        self.log('val/loss', loss, prog_bar=False)
        self.log('val/accuracy', self.val_metrics['accuracy'], prog_bar=False)

        return loss

    def on_validation_epoch_end(self):
        # Log validation metrics after epoch completes (displayed in summary)
        for name, metric in self.val_metrics.items():
            if name not in ['confusion_matrix']:  # confusion matrix handled separately
                value = metric.compute()
                self.log(f'val/{name}', value, prog_bar=False)
            metric.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        for name, metric in self.test_metrics.items():
            metric(preds, y)

        self.log('test/loss', loss)
        self.log('test/accuracy', self.test_metrics['accuracy'])

        return loss

    def on_test_epoch_end(self):
        for name, metric in self.test_metrics.items():
            if name != 'confusion_matrix':
                value = metric.compute()
                if value is not None:
                    self.log(f'test/{name}', value)
            metric.reset()

        # Only compute confusion matrix if test metrics have been updated
        cm_metric = self.test_metrics['confusion_matrix']
        if hasattr(cm_metric, 'total') and cm_metric.total > 0:
            self.test_confusion_matrix = cm_metric.compute()
        else:
            # Initialize with zeros if no test data processed
            num_classes = self.hparams.output_dim
            self.test_confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    def lr_scheduler_step(self, scheduler, metric):
        """
        Custom lr_scheduler_step to handle WarmupReduceLROnPlateau.

        PyTorch Lightning doesn't recognize our custom scheduler as needing metrics,
        so we must extract the monitored metric manually from callback_metrics.
        """
        from spdnet_training.utils.scheduler import WarmupReduceLROnPlateau

        if isinstance(scheduler, WarmupReduceLROnPlateau):
            # Get the monitor key from scheduler config
            monitor_key = self.hparams.scheduler.get('monitor', 'val/accuracy')

            # Extract metric from trainer's callback_metrics
            if self.trainer and hasattr(self.trainer, 'callback_metrics'):
                metric_value = self.trainer.callback_metrics.get(monitor_key)
                if metric_value is not None:
                    metric = float(metric_value)

            scheduler.step(metric)
        elif metric is None:
            scheduler.step()
        else:
            scheduler.step(metric)

    def _get_parameter_groups(self, opt_config):
        """
        Create parameter groups with potentially different learning rates.

        Separates adaptive parameters (like t_gah) from regular parameters
        to allow different learning rates for faster convergence of adaptive params.

        Args:
            opt_config: Optimizer configuration dict

        Returns:
            List of parameter group dicts
        """
        base_lr = opt_config['lr']
        adaptive_lr_multiplier = opt_config.get('adaptive_lr_multiplier', 1.0)

        # Identify adaptive parameters (t_gah in batchnorm layers)
        adaptive_params = []
        regular_params = []

        for name, param in self.named_parameters():
            if param.requires_grad:
                # t_gah parameters are in batchnorm layers with 'parametrizations.t_gah'
                if 't_gah' in name:
                    adaptive_params.append(param)
                else:
                    regular_params.append(param)

        param_groups = []

        # Regular parameters with base learning rate
        if regular_params:
            param_groups.append({
                'params': regular_params,
                'lr': base_lr,
                'name': 'regular'
            })

        # Adaptive parameters with potentially higher learning rate
        if adaptive_params:
            adaptive_lr = base_lr * adaptive_lr_multiplier
            param_groups.append({
                'params': adaptive_params,
                'lr': adaptive_lr,
                'name': 'adaptive'
            })
            if adaptive_lr_multiplier != 1.0:
                print(f"[Optimizer] Using separate LR for adaptive params: "
                      f"base_lr={base_lr:.6f}, adaptive_lr={adaptive_lr:.6f} "
                      f"(multiplier={adaptive_lr_multiplier}x)")

        return param_groups if param_groups else [{'params': self.parameters()}]

    def configure_optimizers(self):
        opt_config = self.hparams.optimizer
        opt_name = opt_config['name'].lower()

        # Get parameter groups (separates adaptive params like t_gah)
        param_groups = self._get_parameter_groups(opt_config)

        # Configure optimizer based on type
        if opt_name == 'adam':
            optimizer = torch.optim.Adam(
                param_groups,
                lr=opt_config['lr'],
                betas=opt_config.get('betas', (0.9, 0.999)),
                eps=opt_config.get('eps', 1e-8),
                weight_decay=opt_config.get('weight_decay', 0.0),
                amsgrad=opt_config.get('amsgrad', False)
            )
        elif opt_name == 'adamw':
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=opt_config['lr'],
                betas=opt_config.get('betas', (0.9, 0.999)),
                eps=opt_config.get('eps', 1e-8),
                weight_decay=opt_config.get('weight_decay', 0.01),
                amsgrad=opt_config.get('amsgrad', False)
            )
        elif opt_name == 'sgd':
            optimizer = torch.optim.SGD(
                param_groups,
                lr=opt_config['lr'],
                momentum=opt_config.get('momentum', 0.0),
                weight_decay=opt_config.get('weight_decay', 0.0),
                dampening=opt_config.get('dampening', 0.0),
                nesterov=opt_config.get('nesterov', False)
            )
        elif opt_name == 'rmsprop':
            optimizer = torch.optim.RMSprop(
                param_groups,
                lr=opt_config['lr'],
                alpha=opt_config.get('alpha', 0.99),
                eps=opt_config.get('eps', 1e-8),
                weight_decay=opt_config.get('weight_decay', 0.0),
                momentum=opt_config.get('momentum', 0.0)
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}. Supported: adam, adamw, sgd, rmsprop")

        if self.hparams.scheduler is None:
            return optimizer

        sch_config = self.hparams.scheduler
        sch_name = sch_config['name'].lower()

        # Configure scheduler based on type
        if sch_name == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=sch_config.get('mode', 'min'),
                factor=sch_config.get('factor', 0.1),
                patience=sch_config.get('patience', 10),
                min_lr=sch_config.get('min_lr', 0.0),
            )

            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': sch_config.get('monitor', 'val/loss'),
                    'interval': 'epoch',
                    'frequency': 1,
                }
            }
        elif sch_name == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=sch_config.get('step_size', 30),
                gamma=sch_config.get('gamma', 0.1)
            )
        elif sch_name == 'multistep':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=sch_config.get('milestones', [30, 60, 90]),
                gamma=sch_config.get('gamma', 0.1)
            )
        elif sch_name == 'exponential':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=sch_config.get('gamma', 0.95)
            )
        elif sch_name == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=sch_config.get('T_max', 100),
                eta_min=sch_config.get('eta_min', 0.0)
            )
        elif sch_name == 'cosine_warm':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=sch_config.get('T_0', 10),
                T_mult=sch_config.get('T_mult', 2),
                eta_min=sch_config.get('eta_min', 0.0)
            )
        elif sch_name == 'warmup_plateau':
            from spdnet_training.utils.scheduler import WarmupReduceLROnPlateau
            scheduler = WarmupReduceLROnPlateau(
                optimizer,
                warmup_epochs=sch_config.get('warmup_epochs', 4),
                target_lr=sch_config.get('target_lr', opt_config['lr']),
                warmup_type=sch_config.get('warmup_type', 'linear'),
                mode=sch_config.get('mode', 'min'),
                factor=sch_config.get('factor', 0.5),
                patience=sch_config.get('patience', 10),
                min_lr=sch_config.get('min_lr', 0.0),
            )

            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': sch_config.get('monitor', 'val/loss'),
                    'interval': 'epoch',
                    'frequency': 1,
                    'strict': True,  # Ensure the monitored metric exists
                }
            }
        elif sch_name == 'none' or sch_name is None:
            return optimizer
        else:
            raise ValueError(f"Unknown scheduler: {sch_name}. Supported: plateau, step, multistep, exponential, cosine, cosine_warm, warmup_plateau, none")

        # For non-plateau schedulers
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': sch_config.get('interval', 'epoch'),
                'frequency': sch_config.get('frequency', 1),
            }
        }
