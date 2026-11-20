"""
Optuna-based hyperparameter optimization for SPDNet training.

Performs joint optimization of all hyperparameters using Tree-structured
Parzen Estimator (TPE).  Each trial trains the model across multiple seeds
and reports the mean objective over the seeds. The objective is read from the
metrics named in the Optuna config: use validation metrics (``val/...``, the
default), so that the test set plays no part in the choice of hyperparameters.
The best trial is also written as an experiment preset for ``spdnet-train``
(``best_experiment.yaml``).

Usage
-----
From an installed environment::

    python -m spdnet_training.optimize \
        --config kaggle_wheat_sgd \
        --experiment kaggle_wheat_sgd_batchnorm \
        --n-trials 60 \
        --storage "sqlite:///optuna_results/study.db" \
        --resume

Another dataset, with configs outside the package::

    python -m spdnet_training.optimize --config-dir my_project/configs \
        --config eeg --dataset bnci2014_001 --experiment bnci2014_001

The script discovers its config directory relative to the installed package
so it works both from an editable install and from ``python -m``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import torch
import yaml
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

# Package directory = spdnet_training/
PACKAGE_DIR = Path(__file__).resolve().parent
CONFIGS_DIR = PACKAGE_DIR / "configs"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def config_file(group: str, name: str, config_dir: Path | None = None) -> Path:
    """``<group>/<name>.yaml`` of ``config_dir`` if it is there, else of the package."""
    if config_dir is not None and (Path(config_dir) / group / f"{name}.yaml").exists():
        return Path(config_dir) / group / f"{name}.yaml"
    return CONFIGS_DIR / group / f"{name}.yaml"


def load_optuna_config(config_name: str, config_dir: Path | None = None) -> DictConfig:
    """Load an Optuna YAML search-space configuration.

    Looks inside ``configs/optuna/<config_name>.yaml`` (of ``config_dir`` first).
    """
    cfg_path = config_file("optuna", config_name, config_dir)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Optuna config not found: {cfg_path}")
    with open(cfg_path) as fh:
        return OmegaConf.create(yaml.safe_load(fh))


# Batch-norm entries of the search space, left out when the batch norm is off
_BATCHNORM_PARAMS = ("batchnorm_method", "batchnorm_type", "batchnorm_norm_strategy",
                     "batchnorm_momentum", "batchnorm_t_gah_init")


def fix_params(optuna_cfg: DictConfig, fixed: dict[str, Any] | None) -> DictConfig:
    """Search space where each entry of ``fixed`` takes a single value.

    ``batchnorm`` (not in the search spaces) can be fixed too. With
    ``batchnorm=False``, the batch-norm entries are removed from the search.
    """
    space = optuna_cfg.search_space
    for name, value in (fixed or {}).items():
        if name != "batchnorm" and name not in space:
            raise ValueError(f"--fix {name}: not in the search space "
                             f"(choices: batchnorm, {', '.join(space)})")
        space[name] = {"type": "categorical", "choices": [value]}
    if fixed and fixed.get("batchnorm") is False:
        for name in _BATCHNORM_PARAMS:
            space.pop(name, None)
    return optuna_cfg


def parse_fixed(items: list[str]) -> dict[str, Any]:
    """``["batchnorm_type=mean_only", "batchnorm=false"]`` -> values parsed as YAML."""
    if bad := [item for item in items if "=" not in item]:
        raise ValueError(f"--fix expects NAME=VALUE, got {bad}")
    pairs = (item.split("=", 1) for item in items)
    return {name: yaml.safe_load(value) for name, value in pairs}


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

# Params sampled conditionally (not in the main loop)
_CONDITIONAL_PARAMS = {
    "sgd_momentum", "warmup_epochs",
    "batchnorm_t_gah_init", "adaptive_lr_multiplier",
}


def _suggest_param(trial: optuna.Trial, name: str, spec) -> Any:
    """Suggest a single hyperparameter from a search-space spec."""
    ptype = spec["type"]
    if ptype == "float":
        return trial.suggest_float(
            name, spec["low"], spec["high"], log=spec.get("log", False),
        )
    elif ptype == "int":
        return trial.suggest_int(
            name, spec["low"], spec["high"], step=spec.get("step", 1),
        )
    elif ptype == "categorical":
        return trial.suggest_categorical(name, list(spec["choices"]))
    else:
        raise ValueError(f"Unknown search-space type: {ptype}")


def sample_hyperparameters(
    trial: optuna.Trial,
    search_space: DictConfig,
) -> dict[str, Any]:
    """Sample hyperparameters from the Optuna trial.

    Handles:
    - Conditional params (optimizer-specific, BN method-specific)
    - Hidden layer depth (h3 = -1 -> skip, 2-layer model)
    - Strictly decreasing layers (values sorted descending)
    - No hidden layer in the search space: the sizes of the model config are kept
    """
    params: dict[str, Any] = {}

    # 1. Sample non-conditional params
    for name, spec in search_space.items():
        if name in _CONDITIONAL_PARAMS:
            continue
        params[name] = _suggest_param(trial, name, spec)

    # 2. Optimizer-conditional params
    optimizer_type = str(params.get("optimizer", "sgd")).lower()
    if optimizer_type == "sgd":
        for name in ("sgd_momentum", "warmup_epochs"):
            if name in search_space:
                params[name] = _suggest_param(trial, name, search_space[name])
    else:
        # Adam: SGD-only params get defaults (not logged by Optuna)
        params["sgd_momentum"] = 0.9
        params["warmup_epochs"] = 10

    # 3. BN method-conditional params (AGAH-specific)
    bn_method = str(params.get("batchnorm_method", "")).lower()
    if "adaptive" in bn_method:
        for name in ("batchnorm_t_gah_init", "adaptive_lr_multiplier"):
            if name in search_space:
                params[name] = _suggest_param(trial, name, search_space[name])
    else:
        params["batchnorm_t_gah_init"] = 0.5
        params["adaptive_lr_multiplier"] = 75.0

    # 4. Build hidden layers: collect valid sizes, sort descending
    #    hidden_layer_2 = -1 -> 1-layer model (h3 also forced to -1)
    #    hidden_layer_3 = -1 -> 2-layer model
    if int(params.get("hidden_layer_2", -1)) == -1:
        params["hidden_layer_3"] = -1  # can't have h3 without h2
    layers = []
    for key in ("hidden_layer_1", "hidden_layer_2", "hidden_layer_3"):
        val = params.get(key, -1)
        if isinstance(val, (int, float)) and val > 0:
            layers.append(int(val))
    layers.sort(reverse=True)
    if len(layers) < 1 and "hidden_layer_1" in search_space:
        layers = [60, 30]  # fallback
    params["_hidden_layers_size"] = layers or None  # None: sizes of the model config

    return params


# ---------------------------------------------------------------------------
# Single-seed training
# ---------------------------------------------------------------------------

def _gpu_available() -> bool:
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


def base_configs(
    experiment_name: str,
    optimizer_type: str,
    dataset_name: str = "kaggle_wheat",
    config_dir: Path | None = None,
) -> tuple[dict, dict, dict]:
    """Dataset, model and trainer configs of a trial, before the sampled params.

    Composed from the YAML files and the experiment overrides, without Hydra, to
    stay independent of its singleton state.
    """
    # 1. Load base configs
    with open(config_file("dataset", dataset_name, config_dir)) as f:
        dataset_cfg = yaml.safe_load(f)
    with open(CONFIGS_DIR / "model" / "spdnet.yaml") as f:
        model_cfg = yaml.safe_load(f)
    # Load trainer config based on optimizer type
    trainer_yaml = "adam_plateau.yaml" if optimizer_type == "adam" else "sgd_warmup_plateau.yaml"
    with open(CONFIGS_DIR / "trainer" / trainer_yaml) as f:
        trainer_cfg = yaml.safe_load(f)
    # Load experiment overrides
    exp_path = config_file("experiment", experiment_name, config_dir)
    exp_overrides: dict = {}
    if exp_path.exists():
        with open(exp_path) as f:
            exp_overrides = yaml.safe_load(f) or {}

    # 2. Apply experiment overrides
    if "model" in exp_overrides:
        model_cfg.update(exp_overrides["model"])
    if "dataset" in exp_overrides:
        dataset_cfg.update(exp_overrides["dataset"])
    if "trainer" in exp_overrides:
        for k, v in exp_overrides["trainer"].items():
            if isinstance(v, dict) and k in trainer_cfg and isinstance(trainer_cfg[k], dict):
                trainer_cfg[k].update(v)
            else:
                trainer_cfg[k] = v

    # 3. Resolve Hydra interpolations that we skipped --------------------------
    # input_dim is ${dataset.input_dim} in the raw YAML -> resolve manually
    model_cfg["input_dim"] = int(dataset_cfg["input_dim"])
    return dataset_cfg, model_cfg, trainer_cfg


def apply_params(
    params: dict[str, Any],
    seed: int,
    optuna_cfg: DictConfig,
    dataset_cfg: dict,
    model_cfg: dict,
    trainer_cfg: dict,
) -> int:
    """Set the sampled params, the seed and the training constraints, in place.

    Returns the early-stopping patience (also written to the trainer config).
    """
    optimizer_type = str(params.get("optimizer", "sgd")).lower()
    # 4. Apply Optuna sampled params ------------------------------------------
    if params["_hidden_layers_size"] is not None:
        model_cfg["hidden_layers_size"] = params["_hidden_layers_size"]
    model_cfg["eps"] = float(params["epsilon"])
    model_cfg["batchnorm"] = bool(params.get("batchnorm", True))  # --fix batchnorm=false: no BN
    if model_cfg["batchnorm"]:
        model_cfg["batchnorm_method"] = params["batchnorm_method"]
        model_cfg["batchnorm_type"] = params["batchnorm_type"]
        model_cfg["batchnorm_norm_strategy"] = params["batchnorm_norm_strategy"]
        model_cfg["batchnorm_momentum"] = float(params["batchnorm_momentum"])
        model_cfg["batchnorm_t_gah_init"] = float(params["batchnorm_t_gah_init"])
    model_cfg["use_logeig"] = bool(params["use_logeig"])
    model_cfg["dropout_rate"] = float(params.get("dropout_rate", 0.0))

    dataset_cfg["batch_size"] = int(params["batch_size"])
    dataset_cfg["seed"] = int(seed)

    # Apply dataset split overrides from optuna config
    ds_overrides = optuna_cfg.get("dataset", {})
    if ds_overrides:
        for k in ("val_ratio", "test_ratio"):
            if k in ds_overrides:
                dataset_cfg[k] = float(ds_overrides[k])

    # Optimizer-specific overrides
    lr_key = "learning_rate" if "learning_rate" in params else "target_lr"  # v2 / v1 compat
    if optimizer_type == "sgd":
        trainer_cfg["optimizer"]["lr"] = 1.0e-5  # warmup ramps to target
        trainer_cfg["optimizer"]["momentum"] = float(params.get("sgd_momentum", 0.9))
        trainer_cfg["optimizer"]["adaptive_lr_multiplier"] = float(params.get("adaptive_lr_multiplier", 75.0))
        trainer_cfg["scheduler"]["target_lr"] = float(params[lr_key])
        trainer_cfg["scheduler"]["warmup_epochs"] = int(params.get("warmup_epochs", 10))
    else:  # adam
        trainer_cfg["optimizer"]["lr"] = float(params[lr_key])
        trainer_cfg["optimizer"]["adaptive_lr_multiplier"] = float(params.get("adaptive_lr_multiplier", 50.0))
    trainer_cfg["scheduler"]["patience"] = int(params["scheduler_patience"])
    trainer_cfg["scheduler"]["factor"] = float(params["scheduler_factor"])

    # Training constraints from optuna config
    training_cfg = optuna_cfg.get("training", {})
    max_epochs = int(training_cfg.get("max_epochs", trainer_cfg.get("max_epochs", 200)))
    es_patience = int(training_cfg.get("early_stopping_patience",
                                       trainer_cfg.get("early_stopping", {}).get("patience", 50)))
    trainer_cfg["max_epochs"] = max_epochs
    trainer_cfg.setdefault("early_stopping", {})["patience"] = es_patience
    return es_patience


def train_single_seed(
    params: dict[str, Any],
    seed: int,
    experiment_name: str,
    optuna_cfg: DictConfig,
    trial: optuna.Trial | None = None,
    enable_pruning: bool = True,
    output_dir: Path | None = None,
    dataset_name: str = "kaggle_wheat",
    config_dir: Path | None = None,
) -> dict[str, float]:
    """Train **one** model with the given hyper-parameters and seed.

    Returns the validation and test metrics of the trained model
    (``val/loss``, ``val/accuracy``, ``test/loss``, ``test/accuracy``, ...).
    """
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

    from spdnet_training.callbacks.optuna_pruning import OptunaPruningCallback
    from spdnet_datasets import DatasetManager
    from spdnet_training import SPDNetModule

    # ---- reproducibility ----
    pl.seed_everything(seed, workers=True)

    optimizer_type = str(params.get("optimizer", "sgd")).lower()
    dataset_cfg, model_cfg, trainer_cfg = base_configs(
        experiment_name, optimizer_type, dataset_name, config_dir)
    es_patience = apply_params(params, seed, optuna_cfg, dataset_cfg, model_cfg, trainer_cfg)
    training_cfg = optuna_cfg.get("training", {})
    max_epochs = trainer_cfg["max_epochs"]

    # ---- create dataloaders ----
    dataset_config_plain = {k: v for k, v in dataset_cfg.items()}
    train_loader, val_loader, test_loader, num_classes = DatasetManager.create_dataloaders(
        dataset_config_plain,
    )

    # ---- create model ----
    model_config_plain = dict(model_cfg)
    model_config_plain["output_dim"] = num_classes
    model_config_plain["optimizer"] = dict(trainer_cfg["optimizer"])
    model_config_plain["scheduler"] = dict(trainer_cfg["scheduler"])
    model = SPDNetModule(**model_config_plain)

    # ---- callbacks ----
    callbacks = [
        EarlyStopping(
            monitor="val/accuracy",
            patience=es_patience,
            mode="max",
            min_delta=0.0,
        ),
    ]

    if output_dir is not None:
        ckpt_dir = output_dir / f"seed_{seed}" / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(ckpt_dir),
                filename="best_model",
                monitor="val/accuracy",
                mode="max",
                save_top_k=1,
            )
        )

    if enable_pruning and trial is not None:
        callbacks.append(OptunaPruningCallback(trial, monitor="val/loss"))

    # ---- trainer ----
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if _gpu_available() else "cpu",
        devices=1,
        precision=int(training_cfg.get("precision", 64)),
        callbacks=callbacks,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        enable_checkpointing=(output_dir is not None),
        deterministic=True,
        benchmark=False,
    )

    # ---- train ----
    try:
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    except optuna.TrialPruned:
        raise

    # ---- evaluate ----
    test_results = trainer.test(model, dataloaders=test_loader, verbose=False)
    # Validation metrics, read by the objective by default so that the test set plays
    # no part in the choice of hyperparameters. They are computed on the best
    # checkpoint when one is saved (as spdnet-train reports), with a trainer without
    # callbacks: the pruning callback must not see this extra validation.
    ckpt = next((c.best_model_path for c in callbacks if isinstance(c, ModelCheckpoint)), "")
    evaluator = pl.Trainer(accelerator="gpu" if _gpu_available() else "cpu", devices=1,
                           precision=int(training_cfg.get("precision", 64)),
                           logger=False, enable_progress_bar=False,
                           enable_model_summary=False, enable_checkpointing=False)
    val_results = evaluator.validate(model, dataloaders=val_loader, ckpt_path=ckpt or None,
                                     verbose=False)
    metrics = {}
    for results in (val_results, test_results):
        if results:
            metrics.update(results[0])
    return metrics


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------

def create_objective(
    optuna_cfg: DictConfig,
    experiment_name: str,
    output_dir: Path | None = None,
    dataset_name: str = "kaggle_wheat",
    config_dir: Path | None = None,
):
    """Return an Optuna objective function (closure)."""

    seeds = list(optuna_cfg.seeds)
    metric = str(optuna_cfg.get("metric", "val/loss"))  # objective of type "single"
    search_space = optuna_cfg.search_space
    pruning_enabled = optuna_cfg.get("pruning", {}).get("enabled", True)

    # --- Objective config ---
    # Validation metrics by default (val/loss, val/accuracy): selecting the
    # hyperparameters on test metrics would bias the reported test results.
    obj_cfg = optuna_cfg.get("objective", {})
    obj_type = str(obj_cfg.get("type", "single")).lower()
    loss_metric = str(obj_cfg.get("loss_metric", "val/loss"))
    acc_metric = str(obj_cfg.get("acc_metric", "val/accuracy"))
    loss_weight = float(obj_cfg.get("loss_weight", 0.5))
    acc_weight = float(obj_cfg.get("acc_weight", 0.5))
    use_combined = (obj_type == "combined")

    def objective(trial: optuna.Trial) -> float:
        params = sample_hyperparameters(trial, search_space)
        trial.set_user_attr("params", params)  # for write_best_experiment, also on --resume
        log.info(f"Trial {trial.number}: {params}")

        seed_results: list[float] = []
        seed_losses: list[float] = []
        seed_accs: list[float] = []

        for i, seed in enumerate(seeds):
            trial_dir = None
            if output_dir is not None:
                trial_dir = output_dir / f"trial_{trial.number:04d}"
                trial_dir.mkdir(parents=True, exist_ok=True)

            try:
                metrics = train_single_seed(
                    params=params,
                    seed=seed,
                    experiment_name=experiment_name,
                    optuna_cfg=optuna_cfg,
                    trial=trial,
                    # Only prune on the FIRST seed to save time
                    enable_pruning=pruning_enabled and (i == 0),
                    output_dir=trial_dir,
                    dataset_name=dataset_name,
                    config_dir=config_dir,
                )
                if use_combined:
                    loss_val = float(metrics.get(loss_metric, 999.0))
                    acc_val = float(metrics.get(acc_metric, 0.0))
                    # Scalarize: minimize (w_loss * loss - w_acc * acc)
                    val = loss_weight * loss_val - acc_weight * acc_val
                    seed_losses.append(loss_val)
                    seed_accs.append(acc_val)
                    log.info(f"  seed={seed}  loss={loss_val:.4f}  acc={acc_val:.4f}  combined={val:.4f}")
                else:
                    val = metrics.get(metric, None)
                    # Fallback: try common aliases
                    if val is None:
                        for k in metrics:
                            if "loss" in k.lower():
                                val = metrics[k]
                                break
                    if val is None:
                        val = 999.0  # worst-case for minimize
                    log.info(f"  seed={seed}  {metric}={val:.4f}")
                seed_results.append(float(val))
            except optuna.TrialPruned:
                log.info(f"  Trial {trial.number} pruned (seed={seed})")
                raise
            except Exception as exc:
                log.warning(f"  seed={seed} FAILED: {exc}")
                seed_results.append(999.0)  # worst-case for minimize
                if use_combined:
                    seed_losses.append(999.0)
                    seed_accs.append(0.0)

        if not seed_results:
            return 999.0  # worst-case for minimize

        mean_val = float(np.mean(seed_results))
        std_val = float(np.std(seed_results, ddof=1)) if len(seed_results) > 1 else 0.0

        trial.set_user_attr("mean_metric", mean_val)
        trial.set_user_attr("std_metric", std_val)
        trial.set_user_attr("seed_results", seed_results)
        trial.set_user_attr("n_seeds_completed", len(seed_results))
        if use_combined:
            mean_loss = float(np.mean(seed_losses))
            mean_acc = float(np.mean(seed_accs))
            trial.set_user_attr("mean_loss", mean_loss)
            trial.set_user_attr("mean_acc", mean_acc)
            log.info(f"Trial {trial.number}: combined={mean_val:.4f} (loss={mean_loss:.4f}, acc={mean_acc:.4f})")
        else:
            log.info(f"Trial {trial.number}: mean={mean_val:.4f} +/- {std_val:.4f}")

        # Save trial params+results to JSON for easy inspection
        if output_dir is not None:
            trial_dir = output_dir / f"trial_{trial.number:04d}"
            trial_dir.mkdir(parents=True, exist_ok=True)
            summary = {
                "trial": trial.number,
                "params": params,
                "seed_results": seed_results,
                "mean": mean_val,
                "std": std_val,
            }
            if use_combined:
                summary["seed_losses"] = seed_losses
                summary["seed_accs"] = seed_accs
                summary["mean_loss"] = float(np.mean(seed_losses))
                summary["mean_acc"] = float(np.mean(seed_accs))
            with open(trial_dir / "trial_summary.json", "w") as fh:
                json.dump(summary, fh, indent=2, default=str)
        return mean_val

    return objective


# ---------------------------------------------------------------------------
# Study runner
# ---------------------------------------------------------------------------

def run_optimization(
    optuna_config_name: str,
    experiment_name: str,
    n_trials: int | None = None,
    storage: str | None = None,
    study_name: str | None = None,
    resume: bool = False,
    output_dir: Path | None = None,
    gpu_id: int | None = None,
    dataset_name: str = "kaggle_wheat",
    config_dir: Path | None = None,
    fixed: dict[str, Any] | None = None,
) -> optuna.Study:
    """Create / load an Optuna study and optimise.

    ``fixed`` replaces search-space entries by a single value (``--fix``).
    """

    optuna_cfg = fix_params(load_optuna_config(optuna_config_name, config_dir), fixed)
    base_configs(experiment_name, "sgd", dataset_name, config_dir)  # fail fast on a bad config

    n_trials = n_trials or int(optuna_cfg.get("n_trials", 60))
    study_name = study_name or str(optuna_cfg.get("study_name", "spdnet_optuna"))
    direction = str(optuna_cfg.get("direction", "maximize"))

    # GPU pinning
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        log.info(f"Pinned to GPU {gpu_id}")

    # Pruner
    pr_cfg = optuna_cfg.get("pruning", {})
    if pr_cfg.get("enabled", False):
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=int(pr_cfg.get("n_startup_trials", 10)),
            n_warmup_steps=int(pr_cfg.get("n_warmup_steps", 20)),
            interval_steps=int(pr_cfg.get("interval_steps", 5)),
        )
    else:
        pruner = optuna.pruners.NopPruner()

    # Create / load study
    if storage:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction=direction,
            pruner=pruner,
            load_if_exists=resume,
        )
    else:
        study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            pruner=pruner,
        )

    objective = create_objective(optuna_cfg, experiment_name, output_dir,
                                 dataset_name, config_dir)

    log.info("=" * 70)
    log.info(f"Optuna study: {study_name}")
    log.info(f"  direction  : {direction}")
    log.info(f"  n_trials   : {n_trials}")
    log.info(f"  seeds      : {list(optuna_cfg.seeds)}")
    log.info(f"  experiment : {experiment_name}")
    log.info(f"  output_dir : {output_dir}")
    log.info("=" * 70)

    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
        gc_after_trial=True,
    )

    return study


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_study_summary(study: optuna.Study, output_dir: Path | None = None):
    """Print results and save CSV / YAML / HTML visualisations."""

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]

    print("\n" + "=" * 80)
    print("OPTUNA OPTIMIZATION SUMMARY")
    print("=" * 80)
    print(f"Study         : {study.study_name}")
    print(f"Total trials  : {len(study.trials)}")
    print(f"  Completed   : {len(completed)}")
    print(f"  Pruned      : {len(pruned)}")
    print(f"  Failed      : {len(failed)}")

    if not completed:
        print("\n  No completed trials -- nothing to report.")
        return

    best = study.best_trial
    print(f"\n--- Best Trial (#{best.number}) ---")
    print(f"  Combined value     : {best.value:.4f}")
    mean_m = best.user_attrs.get("mean_metric", best.value)
    std_m = best.user_attrs.get("std_metric", 0.0)
    print(f"  Mean +/- Std       : {mean_m:.4f} +/- {std_m:.4f}")
    if "mean_loss" in best.user_attrs:
        print(f"  Mean loss          : {best.user_attrs['mean_loss']:.4f}")
        print(f"  Mean accuracy      : {best.user_attrs['mean_acc']:.4f}")
    print(f"  Seed results       : {best.user_attrs.get('seed_results', 'N/A')}")
    print("  Parameters:")
    for k, v in best.params.items():
        print(f"    {k:30s}: {v}")

    # Top 5
    is_minimize = study.direction == optuna.study.StudyDirection.MINIMIZE
    sorted_trials = sorted(
        completed,
        key=lambda t: t.value if t.value is not None else (float('inf') if is_minimize else float('-inf')),
        reverse=not is_minimize,
    )
    print("\n--- Top 5 Trials ---")
    for i, t in enumerate(sorted_trials[:5]):
        extra = ""
        if "mean_loss" in t.user_attrs:
            extra = (f"  loss={t.user_attrs['mean_loss']:.4f}"
                     f"  acc={t.user_attrs['mean_acc']:.4f}")
        print(f"  #{i+1}  Trial {t.number:>3d}  combined={t.value:.4f}  "
              f"mean={t.user_attrs.get('mean_metric', t.value):.4f} +/- "
              f"{t.user_attrs.get('std_metric', 0):.4f}{extra}")

    # Importance
    print("\n--- Parameter Importance ---")
    try:
        importances = optuna.importance.get_param_importances(study)
        for p, imp in importances.items():
            bar = "#" * int(imp * 50)
            print(f"  {p:30s}: {imp:.4f}  {bar}")
    except Exception as exc:
        print(f"  (could not compute: {exc})")

    # ---- persist artefacts ---------------------------------------------------
    if output_dir is None:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV of all trials
    df = study.trials_dataframe()
    df.to_csv(output_dir / "trials.csv", index=False)

    # Best params YAML
    best_info = {
        "best_trial": best.number,
        "best_value": float(best.value),
        "mean_metric": float(mean_m),
        "std_metric": float(std_m),
        "seed_results": best.user_attrs.get("seed_results", []),
        "params": {k: (float(v) if isinstance(v, (np.floating, float)) else v)
                   for k, v in best.params.items()},
    }
    if "mean_loss" in best.user_attrs:
        best_info["mean_loss"] = float(best.user_attrs["mean_loss"])
        best_info["mean_acc"] = float(best.user_attrs["mean_acc"])
    with open(output_dir / "best_params.yaml", "w") as fh:
        yaml.dump(best_info, fh, default_flow_style=False)

    # HTML plots (require plotly)
    try:
        import optuna.visualization as vis

        for name, func in [
            ("optimization_history", vis.plot_optimization_history),
            ("param_importances", vis.plot_param_importances),
            ("parallel_coordinate", vis.plot_parallel_coordinate),
            ("contour", vis.plot_contour),
            ("slice", vis.plot_slice),
        ]:
            try:
                fig = func(study)
                fig.write_html(str(output_dir / f"{name}.html"))
            except Exception:
                pass
        print(f"\n  Plots saved to {output_dir}")
    except ImportError:
        print("\n  (install plotly for HTML visualisations)")

    print(f"  Results saved to {output_dir}")


# ---------------------------------------------------------------------------
# Best trial as an experiment preset
# ---------------------------------------------------------------------------

def _changes(new: dict, old: dict) -> dict:
    """Entries of ``new`` that differ from ``old``, recursively."""
    out = {}
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(old.get(k), dict):
            if sub := _changes(v, old[k]):
                out[k] = sub
        elif k not in old or old[k] != v:
            out[k] = v
    return out


def write_best_experiment(
    path: Path,
    study: optuna.Study,
    optuna_cfg: DictConfig,
    experiment_name: str,
    dataset_name: str = "kaggle_wheat",
    config_dir: Path | None = None,
) -> bool:
    """Write the best trial as an experiment preset of ``spdnet-train``.

    The preset includes the experiment ``experiment_name`` and the dataset
    ``dataset_name``, and sets what the search changed in the configs of a trial
    (``apply_params``): model and trainer settings, the trainer config of the
    optimizer, the batch size. The dataset seed is left out: spdnet-train keeps
    the one of the dataset config. Its runs are named after the study. To use it,
    copy it to ``<config-dir>/experiment/``.

    Returns False, and writes nothing, when a seed of the best trial failed.
    """
    import copy

    best = study.best_trial
    if any(v >= 999.0 for v in best.user_attrs.get("seed_results", [best.value])):
        log.warning(f"Best trial #{best.number} has failed seeds: no preset written.")
        return False
    params = best.user_attrs.get("params") or sample_hyperparameters(
        optuna.trial.FixedTrial(best.params), optuna_cfg.search_space)
    optimizer_type = str(params.get("optimizer", "sgd")).lower()
    base = base_configs(experiment_name, optimizer_type, dataset_name, config_dir)
    tuned = copy.deepcopy(base)
    apply_params(params, 0, optuna_cfg, *tuned)
    dataset, model, trainer = (_changes(n, o) for n, o in zip(tuned, base, strict=True))
    dataset.pop("seed", None)
    trainer_name = "adam_plateau" if optimizer_type == "adam" else "sgd_warmup_plateau"
    preset = {
        "defaults": [experiment_name, {"override /dataset": dataset_name},
                     {"override /trainer": trainer_name}],
        "experiment": {"name": study.study_name},
        "dataset": dataset, "model": model, "trainer": trainer,
    }
    with open(path, "w") as fh:
        fh.write(f"# @package _global_\n# Best trial #{study.best_trial.number} of "
                 f"study {study.study_name} (spdnet-optimize)\n")
        yaml.safe_dump({k: v for k, v in preset.items() if v}, fh, sort_keys=False)
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Optuna hyperparameter optimisation for SPDNet training",
    )
    p.add_argument("--config", required=True,
                   help="Name of the Optuna config in configs/optuna/ (without .yaml)")
    p.add_argument("--experiment", required=True,
                   help="Hydra experiment name (e.g. kaggle_wheat_sgd_batchnorm)")
    p.add_argument("--n-trials", type=int, default=None,
                   help="Number of trials (overrides config)")
    p.add_argument("--storage", type=str, default=None,
                   help="Optuna storage URL, e.g. sqlite:///study.db")
    p.add_argument("--study-name", type=str, default=None)
    p.add_argument("--resume", action="store_true",
                   help="Resume an existing study (requires --storage)")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Directory for results (default: auto-generated)")
    p.add_argument("--gpu", type=int, default=None,
                   help="Pin to a specific GPU id (CUDA_VISIBLE_DEVICES)")
    p.add_argument("--dataset", default="kaggle_wheat",
                   help="Dataset config in configs/dataset/ (without .yaml)")
    p.add_argument("--fix", nargs="+", action="extend", default=[], metavar="NAME=VALUE",
                   help="Fix search-space entries to one value, e.g. "
                        "batchnorm_type=mean_only, or batchnorm=false (no batch norm)")
    p.add_argument("--config-dir", type=Path, default=None,
                   help="Directory with optuna/, dataset/ and experiment/ configs, "
                        "read before those of the package")
    return p.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
    )
    # Reduce Optuna / PL noise
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings("ignore", message=".*compute.*method.*metric.*before.*update.*")

    args = parse_args()

    output_dir: Path | None = None
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        output_dir = PACKAGE_DIR / "results" / f"optuna_{args.config}_{ts}"

    study = run_optimization(
        optuna_config_name=args.config,
        experiment_name=args.experiment,
        n_trials=args.n_trials,
        storage=args.storage,
        study_name=args.study_name,
        resume=args.resume,
        output_dir=output_dir,
        gpu_id=args.gpu,
        dataset_name=args.dataset,
        config_dir=args.config_dir,
        fixed=parse_fixed(args.fix),
    )

    print_study_summary(study, output_dir=output_dir)
    if any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
        optuna_cfg = fix_params(load_optuna_config(args.config, args.config_dir),
                                parse_fixed(args.fix))
        write_best_experiment(output_dir / "best_experiment.yaml", study, optuna_cfg,
                              args.experiment, args.dataset, args.config_dir)


if __name__ == "__main__":
    main()
