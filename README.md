# spdnet-training

Training infrastructure for SPDNet research: PyTorch Lightning module, callbacks, Hydra training loop, and utilities.

## Overview

This package contains the shared training infrastructure extracted from the monolithic spdnet-benchmarks codebase. It provides:

- **Lightning Module**: `SPDNetModule` - PyTorch Lightning wrapper for SPDNet models
- **Training Loop**: Hydra-based training script with comprehensive configuration
- **Callbacks**: Generic callbacks for plotting, metrics logging, covariance analysis, etc.
- **Utilities**: Learning rate schedulers, metrics writers, and helper functions
- **Config Files**: Trainer configurations (Adam, SGD, schedulers, etc.)

## Installation

Install from source:

```bash
cd spdnet-training
pip install -e .
```

Or install directly from git:

```bash
pip install git+https://github.com/Yet-Another-Research-Organisation/spdnet-training.git
```

## Usage

### As a Command-Line Tool

After installation, use the `spdnet-train` command:

```bash
spdnet-train dataset=hyperleaf trainer=adam_plateau
```

### As a Library

Import components directly in your code:

```python
from spdnet_training import SPDNetModule
from spdnet_training.callbacks import PlottingCallback, RichMetricsLogger
from spdnet_training.utils.scheduler import WarmupReduceLROnPlateau
```

## Package Structure

```
spdnet-training/
├── src/spdnet_training/
│   ├── __init__.py
│   ├── lightning_module.py   # SPDNetModule for PyTorch Lightning
│   ├── train.py              # Main training CLI with Hydra
│   ├── callbacks/            # Generic callbacks (plotting, logging, etc.)
│   ├── utils/                # Utilities (schedulers, metrics, etc.)
│   └── configs/              # Hydra config files
├── tests/                    # Unit tests
└── pyproject.toml
```

## Features

- **Flexible Configuration**: Hydra-based configuration system
- **Reproducibility**: Seed management and deterministic training
- **Monitoring**: Rich console output, CSV metrics logging, energy monitoring
- **Analysis**: Covariance analysis, plotting callbacks, confusion matrices
- **Extensibility**: Easy to extend with custom callbacks and configurations

## Dependencies

- Python >= 3.11
- PyTorch >= 2.0
- PyTorch Lightning >= 2.0
- Hydra >= 1.3
- yetanotherspdnet (from GitHub)
- spdnet-datasets (from GitHub)

## License

MIT License

## Authors

- Matthieu Gallet
- Ammar Mian <ammar.mian@univ-smb.fr>
