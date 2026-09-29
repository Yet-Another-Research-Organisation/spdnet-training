# Hyperparameter search

`spdnet-optimize` searches the hyperparameters jointly with Optuna (TPE). Each
trial trains the model on several seeds and reports the mean of the
objective on the test set.

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

## Search space

```yaml
study_name: kaggle_wheat_sgd_optuna
n_trials: 60
seeds: [42, 123, 456]         # each trial trains once per seed
direction: minimize
objective:                    # "single": metric only; "combined": weighted
  type: combined
  loss_metric: test/loss
  acc_metric: test/accuracy
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
Unpromising trials are stopped early by a median pruner
(`OptunaPruningCallback`).

