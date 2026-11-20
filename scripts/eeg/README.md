# EEG datasets (MOABB motor imagery)

Hydra configs and scripts to train SPDNet on MOABB motor-imagery datasets with
`spdnet-train` and `spdnet-optimize`, organised as the other datasets of the
pipeline:

- `configs/dataset/<dataset>.yaml`: the data (keys of `DatasetManager`), one
  flat file per dataset, as `spdnet-optimize` reads them without Hydra;
- `configs/experiment/<dataset>.yaml`: the preset of a run on that dataset
  (dataset, results group `spdnet_<dataset>`, BiMap size);
- `configs/optuna/eeg.yaml`: the search space of `kaggle_wheat_v2.yaml` without
  the hidden layers (fixed by the presets), the split ratios, `dropout_rate`
  and the adaptive GAH mean, with the validation metrics as objective and
  the ranges narrowed on Kaggle Wheat reopened (`sgd_momentum`,
  `scheduler_factor`, `batch_size`).

The data follow the conventions of spdnet-datasets (`MOABBDataset`): geometric
scale fitted on the training split, trials with a non-SPD covariance excluded
(`n_excluded`), the same channels in the same order for every subject (subjects
with other channels are excluded: `channels`, `excluded_subjects`). Every
dataset: band-pass 8-32 Hz, 128 Hz, sample covariance per trial, 75 / 10 / 15 %
train / val / test split of all the trials, stratified by class, drawn with the
run seed. The model and trainer settings, ReEig threshold included, are those
of spdnet-training.

| Dataset | Subjects | Classes | Channels | BiMap |
|---|---|---|---|---|
| `bnci2014_001` | 9 | 4 | 22 | 16 |
| `weibo2014` | 10 | 6 (rest left out) | 60 | 20 |
| `physionetmi` | 106 (88, 92, 100 left out) | 4 | 64 | 16 |
| `cho2017` | 52 | 2 | 64 | 16 |
| `schirrmeister2017` | 14 | 4 | 128 | 16 |
| `lee2019_mi` | 54 | 2 | 62 | no preset |

## Running

`MNE_DATA` is the MOABB download folder, `SPDNET_CACHE_DIR` the covariance
cache. From this folder:

```bash
# one dataset
spdnet-train --config-dir configs experiment=bnci2014_001
# every dataset, with the same overrides (options such as --multirun first)
bash run_eeg.sh --multirun seed=0,1,2 logging.rich_progress=false

# 1. Optuna search on the validation metrics, one study per dataset and batch
#    norm ("none": no batch norm). The best trial of each study becomes the
#    preset configs/experiment/<dataset>_<batch norm>_tuned.yaml
BATCHNORMS="none mean_only mean_var_scalar" bash run_optuna_eeg.sh
# 2. final runs with these presets, over several seeds (split and initialisation)
BATCHNORMS="none mean_only mean_var_scalar" bash run_eeg.sh --multirun \
    seed=0,1,2,3,4 logging.rich_progress=false
```

`spdnet-train` writes to `results/spdnet_<dataset>/<experiment name>/`, in the
current folder (`<date>` for a single run, `<date>_trial_<n>` in a multirun).
`run_optuna_eeg.sh` writes to `results_optuna/<dataset>_<batch norm>_<date>/`.
A study whose best trial failed writes no preset, and `run_eeg.sh` skips it.

A Hydra multirun cannot sweep the `dataset` or `experiment` group with the root
config of spdnet-training (its output folder reads `dataset.name` before the
sweep): the scripts run one Hydra call per dataset and batch norm.

## Other datasets and batch norms

Both scripts work for any dataset with a flat `dataset/<x>.yaml` and an
`experiment/<x>.yaml` preset in the config directory: set `CONFIGS` to that
directory and `DATASETS` to the list. `OPTUNA_CONFIG` names the search space.

A new batch norm of `SPDnet`, selected by a value of `model.batchnorm_type`,
only needs that value in `BATCHNORMS`. Its other arguments, if any, go in the
`model` block of the dataset presets. `spdnet-optimize --fix` accepts only the
entries of the search space and `batchnorm`.
