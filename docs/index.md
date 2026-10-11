# spdnet-training

**spdnet-training** trains the SPD networks of
[yetanotherspdnet](https://github.com/Yet-Another-Research-Organisation/yetanotherspdnet)
on the datasets of
[spdnet-datasets](https://github.com/Yet-Another-Research-Organisation/spdnet-datasets):
a PyTorch Lightning module, a Hydra command line (`spdnet-train`), an Optuna
search (`spdnet-optimize`), callbacks, and CNN backbones for image inputs.

```text
Hydra config ─▶ spdnet-train ─▶ DatasetManager (spdnet-datasets) ─▶ loaders
                     │
                     └──────▶ SPDNetModule ─▶ SPDnet / BackboneSPDnet (yetanotherspdnet)
                                   │
                     Lightning Trainer (early stopping, best checkpoint) ─▶ test ─▶ results/
```

## Install

```bash
pip install "spdnet-training @ git+https://github.com/Yet-Another-Research-Organisation/spdnet-training"
# development
git clone https://github.com/Yet-Another-Research-Organisation/spdnet-training.git
cd spdnet-training && pip install -e ".[dev,test,docs]"
```

## Quick start

The package ships the model and trainer configurations; the dataset
configurations belong to the project using it (for instance
`sigpro_2026/configs/dataset/`), passed with `--config-dir`:

```bash
export DATA_ROOT=/DATA
spdnet-train --config-dir my_project/configs \
    dataset=hyperleaf trainer=sgd_warmup_plateau \
    model.hidden_layers_size=[184,158] model.eps=0.01 \
    trainer.scheduler.target_lr=0.05 dataset.batch_size=48
```

From Python, the Lightning module takes the same keys:

```python
from spdnet_training import SPDNetModule

module = SPDNetModule(
    output_dim=4, input_dim=204, hidden_layers_size=[184, 158], eps=0.01,
    batchnorm=True, batchnorm_method="arithmetic",
    optimizer={"name": "sgd", "lr": 0.05, "momentum": 0.9, "nesterov": True},
    scheduler={"name": "plateau", "mode": "max", "monitor": "val/accuracy"},
)
```

## Contents

| Page | For |
|---|---|
| {doc}`guide/configuration` | the Hydra configuration tree, model keys, optimizers and schedulers |
| {doc}`guide/training` | what `spdnet-train` does, metrics and output files |
| {doc}`guide/optimize` | hyperparameter search with `spdnet-optimize` (Optuna) |
| {doc}`guide/backbones` | image inputs: CNN backbone, covariance pooling, SPDnet |
| {doc}`reference/index` | API reference |

```{toctree}
:hidden:
:caption: Guide

guide/configuration
guide/training
guide/optimize
guide/backbones
```

```{toctree}
:hidden:
:caption: Reference

reference/index
```
