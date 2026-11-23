# Hyperparameter search

`spdnet-optimize` searches the hyperparameters jointly with Optuna (TPE). Each
trial trains the model on several seeds and reports the mean of the
objective. The objective reads the metrics named in the config: by default the
validation metrics (`val/loss`, `val/accuracy`), so that the test set plays no
part in the choice. The validation metrics are taken on the best checkpoint,
as `spdnet-train` reports them. The best trial is written as an experiment
preset, `best_experiment.yaml` in the output directory: copy it to
`<config-dir>/experiment/` and run it with `spdnet-train experiment=<name>`.
A best trial with a failed seed writes no preset.

```bash
spdnet-optimize --config kaggle_wheat_sgd --experiment kaggle_wheat_sgd_batchnorm \
    --n-trials 60 --storage sqlite:///optuna_results/study.db --resume
```

| Option | Meaning |
|---|---|
| `--config` | Search space `configs/optuna/<name>.yaml` |
| `--experiment` | Hydra experiment providing the fixed settings |
| `--n-trials` | Overrides `n_trials` of the config |
| `--storage`, `--study-name`, `--resume` | Optuna storage (SQLite URL) to resume a study |
| `--output-dir`, `--gpu` | Output directory; pin to one GPU |
| `--dataset` | Dataset config `configs/dataset/<name>.yaml` (default `kaggle_wheat`); the same dataset as the experiment |
| `--config-dir` | Directory whose `optuna/`, `dataset/` and `experiment/` configs are read before those of the package |
| `--fix NAME=VALUE ...` | Fix search-space entries to one value: `batchnorm_type=mean_only`, or `batchnorm=false` (no batch norm, batch-norm entries left out of the search). Other names are refused |
| `--split NAME` | Split of the dataset, replacing its `split_strategy` in every trial and in the preset, e.g. `subject_fold_3`: one study per fold for a nested cross-validation |

A key `sampler_seed` in the Optuna config seeds the TPE sampler, so that a rerun samples
the same trials. Without it the sampler is unseeded, as before.

## Search space

```yaml
study_name: kaggle_wheat_sgd_optuna
n_trials: 60
seeds: [42, 123, 456]         # each trial trains once per seed
direction: minimize
objective:                    # "single": metric only; "combined": weighted
  type: combined
  loss_metric: test/loss       # default: val/loss
  acc_metric: test/accuracy    # default: val/accuracy
  loss_weight: 0.5
  acc_weight: 0.5
search_space:
  target_lr: {type: float, low: 0.005, high: 0.15, log: true}
  warmup_epochs: {type: int, low: 3, high: 20, step: 1}
  scheduler_factor: {type: categorical, choices: [0.5, 0.6, 0.75]}
  hidden_layer_1: {type: int, low: 32, high: 120}
  batchnorm_method: {type: categorical, choices: [log_euclidean, adaptive_geometric_arithmetic_harmonic]}
```

Each entry is a `float` (optionally `log`), `int` (with `step`) or
`categorical` parameter. Some are conditional: `sgd_momentum` and
`warmup_epochs` only with SGD; `batchnorm_t_gah_init` and
`adaptive_lr_multiplier` only with the adaptive GAH batch normalization.
`hidden_layer_1..3` are sorted in decreasing order, and `-1` removes a layer.
Without them, the hidden layer sizes of the model config (after the experiment
overrides) are kept.
Unpromising trials are stopped early by a median pruner
(`OptunaPruningCallback`).

