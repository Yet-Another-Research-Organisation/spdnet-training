"""spdnet-optimize: config lookup and hidden layers."""

import optuna
from omegaconf import OmegaConf

from spdnet_training.optimize import (
    CONFIGS_DIR,
    config_file,
    load_optuna_config,
    sample_hyperparameters,
)


def test_config_file_prefers_the_config_dir(tmp_path):
    (tmp_path / "dataset").mkdir()
    (tmp_path / "dataset" / "mine.yaml").write_text("name: mine\n")
    assert config_file("dataset", "mine", tmp_path) == tmp_path / "dataset" / "mine.yaml"
    assert config_file("model", "spdnet", tmp_path) == CONFIGS_DIR / "model" / "spdnet.yaml"


def test_hidden_layers_sampled_or_kept():
    space = load_optuna_config("kaggle_wheat_v2").search_space
    params = sample_hyperparameters(optuna.trial.FixedTrial(_first_choices(space)), space)
    assert params["_hidden_layers_size"] == [30]
    no_hidden = OmegaConf.create({"batch_size": {"type": "categorical", "choices": [32]}})
    params = sample_hyperparameters(optuna.trial.FixedTrial({"batch_size": 32}), no_hidden)
    assert params["_hidden_layers_size"] is None  # sizes of the model config


def _first_choices(space):
    out = {}
    for name, spec in space.items():
        out[name] = spec["choices"][0] if spec["type"] == "categorical" else spec["low"]
    return out


def test_fix_params_checks_names_and_drops_batchnorm_entries():
    import pytest

    from spdnet_training.optimize import fix_params, parse_fixed

    cfg = load_optuna_config("kaggle_wheat_v2")
    with pytest.raises(ValueError, match="not in the search space"):
        fix_params(cfg, {"batchnorm_typ": "mean_only"})
    with pytest.raises(ValueError, match="NAME=VALUE"):
        parse_fixed(["batchnorm_type"])
    assert parse_fixed(["batchnorm=false", "epsilon=0.1"]) == {"batchnorm": False,
                                                               "epsilon": 0.1}
    space = fix_params(cfg, {"batchnorm": False}).search_space
    assert "batchnorm_type" not in space and list(space["batchnorm"]["choices"]) == [False]


def test_apply_params_without_batchnorm_keeps_the_model_settings():
    from spdnet_training.optimize import apply_params, fix_params

    cfg = fix_params(load_optuna_config("kaggle_wheat_v2"), {"batchnorm": False})
    params = sample_hyperparameters(
        optuna.trial.FixedTrial(_first_choices(cfg.search_space)), cfg.search_space)
    model = {"batchnorm_type": "mean_only"}
    apply_params(params, 0, cfg, {}, model, {"optimizer": {}, "scheduler": {}})
    assert model["batchnorm"] is False and model["batchnorm_type"] == "mean_only"


def test_changes():
    from spdnet_training.optimize import _changes

    old = {"a": 1, "b": {"c": [1, 2], "d": None}, "e": 2}
    new = {"a": 1, "b": {"c": [1, 3], "d": None}, "e": 2, "f": 0}
    assert _changes(new, old) == {"b": {"c": [1, 3]}, "f": 0}


def test_no_preset_when_the_best_trial_failed(tmp_path):
    from spdnet_training.optimize import write_best_experiment

    study = optuna.create_study()
    trial = optuna.trial.create_trial(value=999.0, params={}, distributions={},
                                      user_attrs={"seed_results": [999.0]})
    study.add_trial(trial)
    path = tmp_path / "best.yaml"
    assert not write_best_experiment(path, study, load_optuna_config("kaggle_wheat_v2"),
                                     "missing")
    assert not path.exists()
