# Training a model

## What `spdnet-train` does

1. Assembles the configuration; in a multirun with several GPUs, assigns
   job $k$ to GPU `gpus[k % len(gpus)]`.
2. Builds the loaders with `DatasetManager.create_dataloaders(cfg.dataset)`
   and reads the number of classes and the input shape.
3. Builds {py:class}`~spdnet_training.SPDNetModule` from `cfg.model`, with the
   optimizer and scheduler of `cfg.trainer`.
4. Trains with a Lightning `Trainer`: validation every epoch, early stopping
   and checkpointing on `trainer.early_stopping.monitor` /
   `trainer.checkpoint.monitor` (validation accuracy by default).
5. Tests the **best checkpoint** on the test loader.

## Metrics

For each of `train/`, `val/`, `test/`: loss, accuracy, and macro-averaged
precision, recall and F1 (`torchmetrics`); a confusion matrix for validation
and test. Early stopping, checkpoints and the plateau schedulers monitor these
keys (`val/accuracy`, `val/loss`).

## Output directory

`results/<experiment.group>/<experiment.name>/<date>/` (one sub-directory per
job in a multirun):

| File | Content |
|---|---|
| `training.log` | Full log: configuration, dataset, model size, epochs |
| `metrics.csv` | One row per epoch (`CleanCSVMetricsLogger`) |
| `test_results.json` | Test metrics of the best checkpoint (`ResultsSaver`) |
| `spdnet_experiment_status.json` | Durations, energy and GPU memory of the run |
| `plots/` | Learning curves (`logging.save_plots`) |
| `covariance_analysis/` | Spectra of sample covariances (`logging.analyze_covariance`) |
| checkpoint | Best model (`trainer.checkpoint`) |

## Sweeps

```bash
# 3 seeds x 2 batch normalizations, one job each
spdnet-train --multirun --config-dir my_project/configs dataset=hdm05 \
    trainer=sgd_warmup_plateau seed=0,1,2 \
    model.batchnorm_method=arithmetic,geometric_arithmetic_harmonic
# spread over the GPUs listed in launcher/gpu_sweep.yaml
spdnet-train --multirun ... launcher=gpu_sweep
```
