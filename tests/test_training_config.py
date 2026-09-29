"""Hydra config wiring: optimizer presence and the adaptive GAH initial t."""

import pytest
import torch
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from spdnet_training.lightning_module import SPDNetModule
from spdnet_training.train import check_optimization_config


def _compose(*overrides):
    with initialize_config_module("spdnet_training.configs", version_base=None):
        return compose("config", overrides=["+dataset.input_dim=8", *overrides])


def test_default_config_has_an_optimizer():
    check_optimization_config(_compose())


def test_trainer_without_optimizer_fails_fast():
    with pytest.raises(ValueError, match="trainer=adam_plateau"):
        check_optimization_config(_compose("trainer=default"))


@pytest.mark.parametrize("t_init", [0.2, 0.8])
def test_t_gah_init_reaches_the_adaptive_gah_mean(t_init):
    cfg = _compose(
        "model.batchnorm_method=adaptive_geometric_arithmetic_harmonic",
        f"model.batchnorm_t_gah_init={t_init}",
        "model.hidden_layers_size=[6,4]",
    )
    config = OmegaConf.to_container(cfg.model, resolve=True)
    config["output_dim"] = 3
    module = SPDNetModule(**config)
    t_values = [
        m.t_gah
        for m in module.modules()
        if isinstance(getattr(m, "t_gah", None), torch.Tensor)
    ]
    assert t_values
    for t in t_values:
        assert torch.isclose(t, torch.tensor(t_init, dtype=t.dtype))
