"""Backbones, covariance pooling and BackboneSPDnet (no weight download)."""

import pytest
import torch
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from spdnet_training.backbones import (
    BACKBONES,
    CovariancePooling,
    build_backbone,
    build_backbone_spdnet,
)
from spdnet_training.lightning_module import SPDNetModule

WIDTHS = {"efficientnet_b0": 40, "mobilenet_v2": 32, "resnet18": 128}


def _is_spd(matrices: torch.Tensor) -> bool:
    symmetric = torch.allclose(matrices, matrices.transpose(-1, -2))
    return symmetric and bool((torch.linalg.eigvalsh(matrices) > 0).all())


@pytest.mark.parametrize("name", sorted(BACKBONES))
@pytest.mark.parametrize("in_channels", [3, 5])
def test_backbone_width_and_stride(name, in_channels):
    backbone, n_channels = build_backbone(name, in_channels, pretrained=False)
    assert n_channels == WIDTHS[name]
    out = backbone(torch.randn(2, in_channels, 48, 352))
    assert out.shape == (2, n_channels, 6, 44)  # output stride 8


def test_first_conv_keeps_geometry():
    reference, _ = build_backbone("resnet18", 3, pretrained=False)
    adapted, _ = build_backbone("resnet18", 5, pretrained=False)
    old, new = reference[0], adapted[0]
    assert new.in_channels == 5
    assert (new.kernel_size, new.stride, new.padding) == (
        old.kernel_size,
        old.stride,
        old.padding,
    )


def test_unknown_backbone():
    with pytest.raises(ValueError):
        build_backbone("vgg16", pretrained=False)


@pytest.mark.parametrize("estimator", ["scm", "student"])
def test_pooling_returns_spd_float64(estimator):
    features = torch.randn(3, 8, 6, 44, requires_grad=True)
    covariance = CovariancePooling(8, estimator)(features)
    assert covariance.shape == (3, 8, 8) and covariance.dtype == torch.float64
    assert _is_spd(covariance)
    covariance.sum().backward()
    assert torch.isfinite(features.grad).all()


def test_scm_pooling_matches_sample_covariance():
    features = torch.randn(2, 4, 5, 7, dtype=torch.float64)
    covariance = CovariancePooling(4, "scm", eps=0.0)(features)
    for b in range(2):
        torch.testing.assert_close(covariance[b], torch.cov(features[b].flatten(1)))


def test_unknown_estimator():
    with pytest.raises(ValueError):
        CovariancePooling(4, "tyler")


@pytest.mark.parametrize("estimator", ["scm", "student"])
@pytest.mark.parametrize("batchnorm", [False, True])
def test_backbone_spdnet_trains_end_to_end(estimator, batchnorm):
    model = build_backbone_spdnet(
        "mobilenet_v2",
        output_dim=4,
        estimator=estimator,
        pretrained=False,
        hidden_layers_size=[16, 8],
        batchnorm=batchnorm,
        batchnorm_mean_type="geometric_arithmetic_harmonic",
    )
    logits = model(torch.randn(4, 3, 48, 352))
    assert logits.shape == (4, 4) and logits.dtype == torch.float64
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2, 3])).backward()
    for part in (model.backbone, model.spdnet):
        grads = [p.grad for p in part.parameters() if p.requires_grad]
        assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)


def test_lightning_module_builds_backbone_from_hydra_config():
    with initialize_config_module("spdnet_training.configs", version_base=None):
        cfg = compose(
            "config",
            overrides=[
                "model=backbone_spdnet",
                "model.backbone.backbone=resnet18",
                "model.backbone.pretrained=false",
                "+dataset.name=hyperleaf",
            ],
        )
    config = OmegaConf.to_container(cfg.model, resolve=True)
    config["output_dim"] = 4  # set from the dataset by train.py
    module = SPDNetModule(**config)
    assert module.model.spdnet.input_dim == WIDTHS["resnet18"]
    assert module(torch.randn(2, 3, 48, 352)).shape == (2, 4)


def test_student_pooling_with_dead_channel_needs_shrinkage():
    """A channel that is zero on every pixel (dead ReLU) makes the weighted
    sample covariance singular: the Student-t fixed point fails without
    shrinkage and stays SPD, with finite gradients, with it."""
    generator = torch.Generator().manual_seed(0)
    features = torch.randn(2, 6, 12, 12, generator=generator)
    features[:, 3] = 0.0  # dead channel
    with pytest.raises(torch.linalg.LinAlgError):
        CovariancePooling(6, "student")(features)
    features.requires_grad_(True)
    covariance = CovariancePooling(6, "student", shrinkage=0.999)(features)
    eigvals = torch.linalg.eigvalsh(covariance)
    assert covariance.dtype == torch.float64 and eigvals.min() > 0
    covariance.sum().backward()
    assert torch.isfinite(features.grad).all()
