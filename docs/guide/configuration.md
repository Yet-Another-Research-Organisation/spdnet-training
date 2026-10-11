# Configuration

`spdnet-train` is a Hydra application. Its configuration is assembled from
config groups, then any key can be overridden on the command line
(`model.eps=0.01`, `trainer.max_epochs=100`) and swept with `--multirun`.

## The tree

```text
configs/                         (in the package)
├── config.yaml                  root: defaults, seed, logging, paths
├── model/
│   ├── spdnet.yaml              SPDnet on covariance matrices
│   └── backbone_spdnet.yaml     CNN backbone + covariance pooling + SPDnet (images)
├── trainer/
│   ├── default.yaml             Lightning settings only (no optimizer)
│   ├── adam_plateau.yaml        Adam + ReduceLROnPlateau on val/loss
│   └── sgd_warmup_plateau.yaml  SGD (Nesterov) + linear warm-up + plateau on val/accuracy
├── launcher/gpu_sweep.yaml      submitit launcher spreading a multirun over GPUs
└── optuna/*.yaml                search spaces of spdnet-optimize

my_project/configs/              (yours, through --config-dir)
├── dataset/<name>.yaml          keys of DatasetManager.create_dataloaders
└── experiment/<name>.yaml       optional presets (@package _global_)
```

```{warning}
`trainer=default` holds the Lightning settings but no `optimizer` /
`scheduler` block, which the model needs: `spdnet-train` stops at once with a
`ValueError` if neither the trainer nor the model config provides them. The
root config therefore defaults to `trainer=adam_plateau`; with
`trainer=default`, add an `optimizer` and a `scheduler` in your experiment
config.
```

## Root keys

| Key | Default | Meaning |
|---|---|---|
| `seed` | 42 | Seed of the dataset split and of the model initialization |
| `paths.data` | `${oc.env:DATA_ROOT}` | Data root, from the `DATA_ROOT` environment variable |
| `paths.results` | `results` | Root of the output directories |
| `paths.output` | `results/<group>/<name>/<date>` | Directory of one run (see {doc}`training`) |
| `experiment.name`, `experiment.group` | `BENCHMARKS`, `<model>_<dataset>` | Name the output directory |
| `logging.rich_progress`, `logging.save_plots`, `logging.analyze_covariance` | `true`, `true`, `false` | Optional callbacks |

## Model keys

The `model` group is passed to {py:class}`~spdnet_training.SPDNetModule`, which
forwards every key to `yetanotherspdnet.model.SPDnet` (or, with a `backbone`
key, to `build_backbone_spdnet`). Any `SPDnet` argument can therefore be set
from the command line. A few older names are translated:

| Config key | `SPDnet` argument |
|---|---|
| `eps` | `reeig_eps` |
| `batchnorm_method` | `batchnorm_mean_type` |
| `use_vech: true` | `vec_type="vech"` |
| `dtype: float32` | `dtype=torch.float32` (and `trainer.precision=32`) |
| `bimap_parametrization_name`, `bimap_parametrization` | removed (use `bimap_parametrization_mode`) |
| `input_channels`, `name`, `dropout_rate`, `batchnorm_adaptive_mean_type` | ignored |

`output_dim` is set from the number of classes of the dataset;
`input_dim: ${dataset.input_dim}` comes from the dataset config.

`batchnorm_t_gah_init` is converted to `batchnorm_mean_options={"t_init": ...}`
when `batchnorm_method` is `adaptive_geometric_arithmetic_harmonic`: it is the
initial value of the learned $t$ (0 harmonic, 1 arithmetic). It is ignored by
the other batch normalizations.

`trainer.precision` is derived from `model.dtype` when it is set, so that
Lightning does not cast a float32 model to float64 (default: float64,
precision 64).

## Optimizer and scheduler

They are read from `trainer.optimizer` and `trainer.scheduler`.

| `optimizer.name` | Keys |
|---|---|
| `adam`, `adamw` | `lr`, `betas`, `eps`, `weight_decay`, `amsgrad` |
| `sgd` | `lr`, `momentum`, `nesterov`, `dampening`, `weight_decay` |
| `rmsprop` | `lr`, `alpha`, `eps`, `momentum`, `weight_decay` |

`optimizer.adaptive_lr_multiplier` multiplies the learning rate of the
adaptive GAH parameters `t_gah` (separate parameter group).

| `scheduler.name` | Keys |
|---|---|
| `plateau` | `mode`, `factor`, `patience`, `min_lr`, `monitor` |
| `warmup_plateau` | `warmup_epochs`, `target_lr`, `warmup_type`, then the `plateau` keys; the optimizer `lr` is the starting rate of the warm-up |
| `step`, `multistep` | `step_size` / `milestones`, `gamma` |
| `exponential` | `gamma` |
| `cosine`, `cosine_warm` | `T_max`, `eta_min` / `T_0`, `T_mult`, `eta_min` |
| `none` | no scheduler |

## Reproducing the SPDNet batch normalization paper

```bash
spdnet-train --config-dir sigpro_2026/configs \
    dataset=hyperleaf_final trainer=sgd_warmup_plateau \
    model.hidden_layers_size=[184,158] model.eps=0.01 \
    model.batchnorm=true model.batchnorm_method=arithmetic model.batchnorm_momentum=0.01 \
    trainer.optimizer.lr=1e-5 trainer.scheduler.target_lr=0.05 \
    trainer.scheduler.warmup_epochs=5 trainer.scheduler.patience=3 \
    trainer.early_stopping.patience=20 dataset.batch_size=48
```

(per-method batch sizes and learning rates: see the paper's supplementary
table, or `spdnet-benchmark-demo/configs.py`).
