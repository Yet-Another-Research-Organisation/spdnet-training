"""Main training CLI with Hydra configuration."""
import sys
import os
import warnings
from pathlib import Path
import json
import logging
import time

import torch

# Third-party imports
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# Local imports
from spdnet_training.lightning_module import SPDNetModule
from spdnet_training.callbacks import (
    PlottingCallback,
    RichMetricsLogger,
    ResultsSaver,
    CovarianceAnalysisCallback,
)
from spdnet_training.callbacks.csv_metrics_logger import CleanCSVMetricsLogger
from spdnet_datasets import DatasetManager
from spdnet_training.utils.metrics import EnhancedMetricsWriter

# Compute config path relative to this file
config_path = str(Path(__file__).parent / "configs")

# Filter torchmetrics warnings about computing metrics before update
warnings.filterwarnings('ignore', message='.*compute.*method.*metric.*before.*update.*', category=UserWarning)

# Setup logger
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path=config_path, config_name="config")
def main(cfg: DictConfig):
    """
    Main training function.

    Args:
        cfg: Hydra configuration object
    """

    # Auto-assign GPU based on Hydra job number for multi-GPU parallelization
    # This works with joblib launcher when running multiple jobs in parallel
    gpu_id = None
    job_num = None
    try:
        from hydra.core.hydra_config import HydraConfig
        if HydraConfig.initialized():
            job_num = HydraConfig.get().job.num
            if job_num is not None:
                # Get available GPUs from config or auto-detect
                available_gpus = OmegaConf.select(cfg, "gpus")
                if available_gpus is None:
                    if torch.cuda.is_available():
                        available_gpus = list(range(torch.cuda.device_count()))
                    else:
                        available_gpus = [0]

                if len(available_gpus) > 0:
                    gpu_id = available_gpus[job_num % len(available_gpus)]
                    # Set CUDA_VISIBLE_DEVICES before any CUDA operation
                    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
                    log.info(f"Job {job_num} auto-assigned to GPU {gpu_id} (available GPUs: {available_gpus})")
    except Exception as e:
        log.warning(f"Could not get Hydra job number: {e}")

    # Log Tensor Core precision setting
    matmul_precision = torch.get_float32_matmul_precision()
    log.info(f"Tensor Core matmul precision: {matmul_precision}")

    # Setup output directory
    # In multirun mode, Hydra creates separate subdirectories for each job
    # We need to use the actual runtime directory, not the configured one
    try:
        from hydra.core.hydra_config import HydraConfig
        if HydraConfig.initialized():
            hydra_cfg = HydraConfig.get()
            # Use Hydra's runtime output dir which includes trial_X in multirun
            runtime_output_dir = Path(hydra_cfg.runtime.output_dir)
            # Update cfg.paths.output to use the actual runtime directory
            OmegaConf.update(cfg, "paths.output", str(runtime_output_dir), merge=False)
    except Exception as e:
        log.debug(f"Could not get Hydra runtime directory: {e}")

    # Setup file logging to results directory
    output_dir = Path(cfg.paths.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "training.log"

    # Add file handler to root logger
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] - %(message)s')
    file_handler.setFormatter(file_formatter)
    logging.getLogger().addHandler(file_handler)

    log.info(f"Logging to file: {log_file}")

    # Start timing
    start_time = time.time()

    # Print configuration
    log.info("\n" + "="*80)
    log.info("CONFIGURATION")
    log.info("="*80)
    log.info("\n" + OmegaConf.to_yaml(cfg))
    log.info("="*80 + "\n")

    # Initialize enhanced metrics writer
    metrics_writer = EnhancedMetricsWriter(
        log_dir=cfg.paths.output,
        experiment_name=cfg.get('experiment_name', 'spdnet_experiment'),
        monitoring_interval=1.0
    )

    # Set seed
    pl.seed_everything(cfg.seed, workers=True)
    log.info(f"Random seed: {cfg.seed}")

    # Create dataset and dataloaders using DatasetManager
    log.info("Loading dataset...")
    dataset_config = OmegaConf.to_container(cfg.dataset, resolve=True)
    train_loader, val_loader, test_loader, num_classes = DatasetManager.create_dataloaders(dataset_config)

    log.info(f"✓ Dataset loaded: {cfg.dataset.name}")
    log.info(f"  - Number of classes: {num_classes}")
    log.info(f"  - Train batches: {len(train_loader)}")
    log.info(f"  - Val batches: {len(val_loader)}")
    log.info(f"  - Test batches: {len(test_loader)}")

    # Get a sample to determine input dimensions
    sample_batch = next(iter(train_loader))
    sample_x = sample_batch[0]
    input_shape = sample_x.shape
    log.info(f"  - Input shape: {input_shape}")

    # Initialize model with ALL parameters from config
    log.info("\nBuilding model...")

    # Convert model config to dict and add dataset-specific parameters
    model_config = OmegaConf.to_container(cfg.model, resolve=True)
    model_config['output_dim'] = num_classes

    # Override optimizer and scheduler with trainer settings if available
    # This ensures trainer config takes precedence over model defaults
    if hasattr(cfg.trainer, 'optimizer') and cfg.trainer.optimizer is not None:
        model_config['optimizer'] = OmegaConf.to_container(cfg.trainer.optimizer, resolve=True)
    if hasattr(cfg.trainer, 'scheduler') and cfg.trainer.scheduler is not None:
        model_config['scheduler'] = OmegaConf.to_container(cfg.trainer.scheduler, resolve=True)

    # Log model configuration
    log.info("Model configuration:")
    for key, value in sorted(model_config.items()):
        if key not in ['optimizer', 'scheduler']:
            log.info(f"  - {key}: {value}")

    # Log optimizer and scheduler from trainer
    log.info(f"Optimizer: {model_config['optimizer']['name']} (lr={model_config['optimizer']['lr']})")
    log.info(f"Scheduler: {model_config['scheduler']['name']}")

    # Create model - pass ALL parameters
    model = SPDNetModule(**model_config)

    # Log model info to metrics writer
    metrics_writer.log_model_info(
        model_name='SPDNet',
        total_params=sum(p.numel() for p in model.parameters()),
        trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
        **{k: v for k, v in model_config.items() if k not in ['optimizer', 'scheduler']}
    )

    log.info("✓ Model created")
    log.info(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    log.info(f"  - Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Log device info
    if torch.cuda.is_available():
        log.info(f"  - GPU: {torch.cuda.get_device_name(0)}")
        log.info(f"  - GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        log.info("  - Device: CPU")

    # Setup callbacks
    log.info("\nSetting up callbacks...")
    callbacks = []

    # Checkpoint callback - save only best model
    checkpoint_dir = Path(cfg.paths.output) / 'checkpoints'
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='best_model',
        monitor=cfg.trainer.checkpoint.monitor,
        mode=cfg.trainer.checkpoint.mode,
        save_top_k=1,  # Save only the best model
        save_last=False,  # Don't save last checkpoint
        verbose=True,
    )
    callbacks.append(checkpoint_callback)
    log.info(f"  ✓ Checkpoint callback (save best only, monitor: {cfg.trainer.checkpoint.monitor})")

    # Early stopping callback
    early_stop = EarlyStopping(
        monitor=cfg.trainer.early_stopping.monitor,
        patience=cfg.trainer.early_stopping.patience,
        mode=cfg.trainer.early_stopping.mode,
        min_delta=cfg.trainer.early_stopping.min_delta,
        verbose=True,
    )
    callbacks.append(early_stop)
    log.info(f"  ✓ Early stopping (patience: {cfg.trainer.early_stopping.patience})")

    # Plotting callback
    if cfg.logging.save_plots:
        plot_callback = PlottingCallback(
            save_dir=Path(cfg.paths.output) / 'plots'
        )
        callbacks.append(plot_callback)
        log.info("  ✓ Plotting callback")

    # Covariance analysis callback - analyzes per-class covariances on test set
    if cfg.logging.get('analyze_covariance', False):
        cov_callback = CovarianceAnalysisCallback(
            save_dir=Path(cfg.paths.output) / 'covariance_analysis',
            num_samples_per_class=cfg.logging.get('num_covariance_samples_per_class', 3),
        )
        callbacks.append(cov_callback)
        log.info("  ✓ Covariance analysis callback (per-class on test set)")

    # Rich progress logger
    if cfg.logging.rich_progress:
        rich_logger = RichMetricsLogger(save_dir=Path(cfg.paths.output))
        callbacks.append(rich_logger)
        log.info("  ✓ Rich metrics logger")

    # Results saver
    results_saver = ResultsSaver(save_dir=Path(cfg.paths.output))
    callbacks.append(results_saver)
    log.info("  ✓ Results saver")

    # Clean CSV metrics logger
    csv_metrics_callback = CleanCSVMetricsLogger(save_dir=Path(cfg.paths.output))
    callbacks.append(csv_metrics_callback)
    log.info("  ✓ Clean CSV metrics logger")

    # Initialize trainer
    log.info("\nConfiguring trainer...")
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        callbacks=callbacks,
        logger=False,  # Disable default logger, using custom CSV callback instead
        deterministic=cfg.trainer.deterministic,
        benchmark=cfg.trainer.benchmark,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        enable_progress_bar=True,
        enable_model_summary=False,  # Use custom ModelSummary callback instead
    )

    log.info(f"  - Max epochs: {cfg.trainer.max_epochs}")
    log.info(f"  - Accelerator: {cfg.trainer.accelerator}")
    log.info(f"  - Devices: {cfg.trainer.devices}")
    log.info(f"  - Precision: {cfg.trainer.precision}")

    # Start energy monitoring
    log.info("\nStarting energy monitoring...")
    metrics_writer.start_energy_monitoring()

    # Train
    log.info("\n" + "="*80)
    log.info("STARTING TRAINING")
    log.info("="*80 + "\n")

    training_start = time.time()
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    training_duration = time.time() - training_start

    # Stop energy monitoring and get stats
    energy_stats = metrics_writer.stop_energy_monitoring()

    log.info("\n" + "="*80)
    log.info("TRAINING COMPLETE")
    log.info("="*80)
    log.info(f"Training duration: {training_duration/60:.2f} minutes")
    log.info(f"Best model saved at: {checkpoint_callback.best_model_path}")
    log.info(f"Best validation {cfg.trainer.checkpoint.monitor}: {checkpoint_callback.best_model_score:.4f}")

    # Test
    log.info("\n" + "="*80)
    log.info("RUNNING TEST")
    log.info("="*80 + "\n")

    test_start = time.time()
    test_results = trainer.test(model, dataloaders=test_loader, ckpt_path='best')
    test_duration = time.time() - test_start

    log.info("\n" + "="*40)
    log.info("TEST RESULTS")
    log.info("="*40)
    for key, value in test_results[0].items():
        log.info(f"  - {key}: {value:.4f}")

    # Note: Hydra automatically saves the complete config to .hydra/config.yaml
    # This includes all composed configs (dataset, model, launcher, trainer)
    # We can also save specific configs for convenience
    output_path = Path(cfg.paths.output)

    # Save experiment status with durations and energy stats
    experiment_status = {
        "training_duration_minutes": training_duration / 60,
        "test_duration_minutes": test_duration / 60,
        "total_duration_minutes": (time.time() - start_time) / 60,
        "best_model_path": checkpoint_callback.best_model_path,
        "best_model_score": float(checkpoint_callback.best_model_score),
        "monitor_metric": cfg.trainer.checkpoint.monitor,
    }

    # Add energy stats if available
    if energy_stats:
        experiment_status["energy_consumption"] = energy_stats

    # Save experiment status to JSON
    status_file = output_path / 'spdnet_experiment_status.json'
    with open(status_file, 'w') as f:
        json.dump(experiment_status, f, indent=2)

    # Save individual config components for easy reference
    config_dir = output_path / 'configs'
    config_dir.mkdir(parents=True, exist_ok=True)

    # Save dataset config
    dataset_config_path = config_dir / 'dataset.yaml'
    with open(dataset_config_path, 'w') as f:
        OmegaConf.save(cfg.dataset, f)

    # Save model config
    model_config_path = config_dir / 'model.yaml'
    with open(model_config_path, 'w') as f:
        OmegaConf.save(cfg.model, f)

    # Save trainer config
    trainer_config_path = config_dir / 'trainer.yaml'
    with open(trainer_config_path, 'w') as f:
        OmegaConf.save(cfg.trainer, f)

    # Save launcher config if present
    if 'launcher' in cfg and cfg.launcher is not None:
        launcher_config_path = config_dir / 'launcher.yaml'
        with open(launcher_config_path, 'w') as f:
            OmegaConf.save(cfg.launcher, f)

    # Copy complete Hydra config to output directory for analysis
    # This is needed because Hydra saves it in a different location during multirun
    hydra_dir = output_path / '.hydra'
    hydra_dir.mkdir(parents=True, exist_ok=True)
    hydra_config_path = hydra_dir / 'config.yaml'
    with open(hydra_config_path, 'w') as f:
        OmegaConf.save(cfg, f)

    log.info(f"\n✓ All configurations saved to: {config_dir}")
    log.info(f"  - Complete config also in: {output_path}/.hydra/config.yaml")

    total_duration = time.time() - start_time
    log.info(f"\n{'='*80}")
    log.info("EXPERIMENT COMPLETE")
    log.info(f"{'='*80}")
    log.info(f"Total duration: {total_duration/60:.2f} minutes")
    log.info(f"Results directory: {cfg.paths.output}")
    log.info(f"{'='*80}\n")

    # Clean up empty train.log file created by Hydra
    train_log_file = output_dir / 'train.log'
    if train_log_file.exists() and train_log_file.stat().st_size == 0:
        train_log_file.unlink()

    # Close file handler
    log.info(f"Training complete. Full log saved to: {output_dir / 'training.log'}")

    return test_results[0]


if __name__ == "__main__":
    main()
