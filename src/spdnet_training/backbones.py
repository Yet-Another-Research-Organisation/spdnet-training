"""CNN backbones + covariance pooling + SPDnet, for image inputs.

When only an image is available (e.g. a few bands of a hyperspectral cube),
there is no covariance to feed SPDnet directly: a truncated ImageNet backbone
extracts a feature map, the covariance of its channels over the pixels is
estimated (sample covariance or a robust M-estimator), and SPDnet classifies
that SPD matrix. The whole chain is trained end to end.

The backbone runs in float32 (speed); pooling and SPDnet run in the SPDnet
dtype (float64 by default, the library convention for eigendecompositions).
"""

from functools import partial

import torch
from torch import nn
from torchvision import models
from yetanotherspdnet.functions.m_estimators import sample_covariance, student_function
from yetanotherspdnet.model import SPDnet
from yetanotherspdnet.nn import MEstimation


def _resnet18_stem(net: nn.Module) -> nn.Sequential:
    return nn.Sequential(
        net.conv1, net.bn1, net.relu, net.maxpool, net.layer1, net.layer2
    )


# name -> (constructor, ImageNet weights, truncation at stride 8, first conv path)
# Stride 8 keeps enough pixels per covariance on small images (48x352 -> 6x44).
BACKBONES = {
    "efficientnet_b0": (
        models.efficientnet_b0,
        models.EfficientNet_B0_Weights.IMAGENET1K_V1,
        lambda net: net.features[:4],  # 40 channels
        (0, 0),
    ),
    "mobilenet_v2": (
        models.mobilenet_v2,
        models.MobileNet_V2_Weights.IMAGENET1K_V1,
        lambda net: net.features[:7],  # 32 channels
        (0, 0),
    ),
    "resnet18": (
        models.resnet18,
        models.ResNet18_Weights.IMAGENET1K_V1,
        _resnet18_stem,  # 128 channels
        (0,),
    ),
}


def build_backbone(
    name: str, in_channels: int = 3, pretrained: bool = True
) -> tuple[nn.Sequential, int]:
    """
    Truncated torchvision backbone (output stride 8).

    Parameters
    ----------
    name : str
        One of ``BACKBONES``: ``"efficientnet_b0"``, ``"mobilenet_v2"``,
        ``"resnet18"``

    in_channels : int, optional
        Number of input channels. If not 3, the first convolution is replaced
        (same kernel, stride and padding) and, when pretrained, initialized with
        the mean of the ImageNet RGB filters. Default is 3

    pretrained : bool, optional
        Load ImageNet weights (downloaded once by torchvision). Default is True

    Returns
    -------
    backbone : nn.Sequential
        Feature extractor mapping (B, in_channels, H, W) to (B, C, H/8, W/8)

    n_channels : int
        Number of output channels C
    """
    if name not in BACKBONES:
        raise ValueError(
            f"Unknown backbone {name!r}, expected one of {list(BACKBONES)}"
        )
    constructor, weights, truncate, conv_path = BACKBONES[name]
    backbone = truncate(constructor(weights=weights if pretrained else None))
    if in_channels != 3:
        parent = backbone
        for index in conv_path[:-1]:
            parent = parent[index]
        old = parent[conv_path[-1]]
        new = nn.Conv2d(
            in_channels,
            old.out_channels,
            old.kernel_size,
            old.stride,
            old.padding,
            bias=old.bias is not None,
        )
        if pretrained:
            with torch.no_grad():
                mean = old.weight.mean(dim=1, keepdim=True)
                new.weight.copy_(mean.expand(-1, in_channels, -1, -1))
        parent[conv_path[-1]] = new
    with torch.no_grad():
        n_channels = backbone.eval()(torch.zeros(1, in_channels, 32, 32)).shape[1]
    return backbone.train(), n_channels


class CovariancePooling(nn.Module):
    """
    Covariance of the channels of a feature map over its pixels.

    Maps (B, C, H, W) to SPD matrices (B, C, C) in ``dtype``: the pixels are the
    samples, then ``eps * I`` is added for numerical stability.

    Parameters
    ----------
    n_features : int
        Number of channels C of the feature map

    estimator : str, optional
        ``"scm"`` (sample covariance) or ``"student"`` (Student-t M-estimator,
        robust to outlying pixels, implicit fixed-point backward).
        Default is ``"scm"``

    nu : float, optional
        Degrees of freedom of the Student-t estimator. Default is 5.0

    n_iterations : int, optional
        Fixed-point iterations of the Student-t estimator. Default is 10

    eps : float, optional
        Diagonal loading. Default is 1e-5

    shrinkage : float | None, optional
        Student-t estimator only: each fixed-point iterate becomes
        ``shrinkage * F(Sigma) + (1 - shrinkage) * I`` (regularized
        M-estimator), which keeps it invertible. Needed when some channels can
        be identically zero, as after a ReLU (ResNet18 cut): the weighted
        sample covariance of the first iteration is then singular and the next
        Cholesky fails; ``eps`` is only added after the estimation.
        Default is None (no shrinkage)

    dtype : torch.dtype, optional
        Output dtype. Default is torch.float64
    """

    def __init__(
        self,
        n_features: int,
        estimator: str = "scm",
        nu: float = 5.0,
        n_iterations: int = 10,
        eps: float = 1e-5,
        shrinkage: float | None = None,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        if estimator not in ("scm", "student"):
            raise ValueError(f"estimator must be 'scm' or 'student', got {estimator!r}")
        self.estimator, self.nu, self.eps, self.dtype = estimator, nu, eps, dtype
        self.shrinkage = shrinkage
        self.m_estimation = None
        if estimator == "student":
            weight = partial(student_function, n_features=n_features, nu=nu)
            self.m_estimation = MEstimation(
                weight, n_iterations=n_iterations, shrinkage=shrinkage
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        pixels = features.flatten(2).transpose(1, 2).to(self.dtype)  # (B, HW, C)
        if self.m_estimation is None:
            covariance = sample_covariance(pixels)
        else:
            covariance = self.m_estimation(pixels)
        eye = torch.eye(pixels.shape[-1], dtype=self.dtype, device=pixels.device)
        return covariance + self.eps * eye

    def extra_repr(self) -> str:
        return (
            f"estimator={self.estimator}, nu={self.nu}, eps={self.eps}, "
            f"shrinkage={self.shrinkage}"
        )


class BackboneSPDnet(nn.Sequential):
    """Backbone -> CovariancePooling -> SPDnet, trained end to end."""

    def __init__(self, backbone: nn.Module, pooling: CovariancePooling, spdnet: SPDnet):
        super().__init__()
        self.backbone, self.pooling, self.spdnet = backbone, pooling, spdnet


def build_backbone_spdnet(
    backbone: str,
    output_dim: int,
    estimator: str = "scm",
    in_channels: int = 3,
    pretrained: bool = True,
    pooling_options: dict | None = None,
    **spdnet_kwargs,
) -> BackboneSPDnet:
    """
    Build a :class:`BackboneSPDnet`; SPDnet's ``input_dim`` is the backbone width.

    Parameters
    ----------
    backbone : str
        Backbone name, see :func:`build_backbone`

    output_dim : int
        Number of classes

    estimator : str, optional
        Covariance estimator, see :class:`CovariancePooling`. Default is ``"scm"``

    in_channels : int, optional
        Number of image channels (bands). Default is 3

    pretrained : bool, optional
        ImageNet initialization of the backbone. Default is True

    pooling_options : dict, optional
        Extra :class:`CovariancePooling` arguments (``nu``, ``n_iterations``,
        ``eps``)

    **spdnet_kwargs
        :class:`yetanotherspdnet.model.SPDnet` arguments (``hidden_layers_size``,
        ``batchnorm``, ``dtype``, ``device``...)

    Returns
    -------
    model : BackboneSPDnet
    """
    features, n_channels = build_backbone(backbone, in_channels, pretrained)
    dtype = spdnet_kwargs.get("dtype", torch.float64)
    pooling = CovariancePooling(
        n_channels, estimator, dtype=dtype, **(pooling_options or {})
    )
    spdnet = SPDnet(input_dim=n_channels, output_dim=output_dim, **spdnet_kwargs)
    device = spdnet_kwargs.get("device", torch.device("cpu"))
    return BackboneSPDnet(features.to(device), pooling, spdnet)
