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
    seed=0,1,2,3,4,5,6,7,8,9 logging.rich_progress=false
```

`spdnet-train` writes to `results/spdnet_<dataset>/<experiment name>/`, in the
current folder (`<date>` for a single run, `<date>_trial_<n>` in a multirun).
`run_optuna_eeg.sh` writes to `results_optuna/<dataset>_<batch norm>_<date>/`.
A study whose best trial failed writes no preset, and `run_eeg.sh` skips it.

A Hydra multirun cannot sweep the `dataset` or `experiment` group with the root
config of spdnet-training (its output folder reads `dataset.name` before the
sweep): the scripts run one Hydra call per dataset and batch norm.

## Cross-subject protocol

The datasets `<dataset>_subjects` (`configs/dataset/` and `configs/experiment/`)
evaluate on subjects never seen in training, with the same model, search space and
scripts. They use the `subject_fold_<k>` splits of spdnet-datasets
(`subject_folds`), built like the `stratified_fold_<k>` splits of its nilearn
datasets:

- one fold per subject, the same in every run (`fold_seed`): fold `k` tests one
  subject, validates on the next two (`val_groups: 2`) and trains on the others.
  Every subject is tested once:

  | Dataset | Subjects | Sessions | Folds |
  |---|---|---|---|
  | `bnci2014_001_subjects` | 9 | 2 | 9 |
  | `weibo2014_subjects` | 10 | 1 | 10 |
  | `schirrmeister2017_subjects` | 14 | 1 | 14 |

- each session of a validation or test subject is cut in recording order: its
  first half is an unlabeled calibration pool, its second half is evaluated. The
  adaptation to a session only sees trials recorded before the evaluated ones, in
  the same session, so the protocol stays cross-subject for BNCI2014_001 and its
  two sessions. The labels are not used, so the loader must keep the recording
  order (MOABB stores Cho2017 class by class: no such protocol for it);
- each training batch holds trials of one session of one subject;
- at validation and test, the running statistics of the SPD batch norm are those of
  the evaluated session, computed on its calibration pool
  (`spdnet_training/cross_subject.py`): the whole pool for a validation session,
  its first `K` trials for a test session, for each `K` of `calib_sizes` (`0`: no
  adaptation: the running statistics of training, `null`: the whole pool). The
  batch norm therefore normalizes with its batch statistics in training
  (`batchnorm_norm_strategy=classical`), and its momentum, which only acts on
  `K = 0`, is fixed to 0.01 in these studies. Each run writes `target_metrics.json`:
  accuracy over the trials, mean over the test subjects and accuracy of each test
  subject, for each `K`. The usual test metrics use the whole pool.

Hyperparameters are tuned within each fold (nested cross-validation): one Optuna
study per fold and batch norm (40 trials by default, `N_TRIALS`), on the training
and validation subjects of that fold only, and one preset
`<dataset>_fold<k>_<batch norm>_tuned.yaml` (its last line names the study). Each
fold is then trained with its own preset. The score of a batch norm is the mean over
the folds: check that every fold has a preset for every batch norm (`run_eeg.sh`
skips a missing one with a message). The Optuna sampler is seeded
(`sampler_seed` in `eeg.yaml`).

```bash
S="bnci2014_001_subjects weibo2014_subjects schirrmeister2017_subjects"
# 1. Optuna, one study per fold and batch norm (FOLDS="0 1" restricts the folds)
DATASETS="$S" BATCHNORMS="none mean_only mean_var_scalar" bash run_optuna_eeg.sh
# 2. every fold with its preset, over 10 seeds (initialisation and batch order)
DATASETS="$S" BATCHNORMS="none mean_only mean_var_scalar" bash run_eeg.sh --multirun \
    seed=0,1,2,3,4,5,6,7,8,9 logging.rich_progress=false
```

## Other datasets and batch norms

Both scripts work for any dataset with a flat `dataset/<x>.yaml` and an
`experiment/<x>.yaml` preset in the config directory: set `CONFIGS` to that
directory and `DATASETS` to the list. `OPTUNA_CONFIG` names the search space.

A new batch norm of `SPDnet`, selected by a value of `model.batchnorm_type`, needs
that value in `BATCHNORMS`. Its other arguments, if any, go in the `model` block
of the dataset presets. `spdnet-optimize --fix` accepts only the entries of the
search space and `batchnorm`. For the cross-subject protocol, the layer must also
subclass `BatchNormSPDBase`, keep its running statistics in buffers, set them to
the batch statistics when its momentum is 1, and stay well defined on a few
trials (K = 4): add it to `BATCHNORM_TYPES` in `tests/test_cross_subject.py`,
whose test checks this. Tune and run every batch norm with the same version of
yetanotherspdnet.
