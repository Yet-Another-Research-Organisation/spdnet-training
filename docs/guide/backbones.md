# Backbones: images → covariance → SPDnet

`spdnet_training.backbones` handles datasets where the input is an **image**
rather than a covariance matrix (for instance three bands of a hyperspectral
cube, with `spdnet-datasets`' HyperLeaf `mode="image"`):

```
image (B, bands, H, W)
  → truncated ImageNet backbone (float32)   feature map (B, C, H/8, W/8)
  → CovariancePooling (float64)             SPD matrix  (B, C, C)
  → SPDnet (yetanotherspdnet)               logits      (B, n_classes)
```

Everything is trained end to end. The code is a cleaned-up version of the
`backbone_dataset` research code (`EfficientNetB0SPDNet`, `ResNetSPDNet`,
`CovariancePooling`) and of the ResNet18 + M-estimator scripts of the
`mestimator` experiments.

## Backbones

| Name | Cut | Output width |
|---|---|---|
| `efficientnet_b0` | `features[:4]` | 40 |
| `mobilenet_v2` | `features[:7]` | 32 |
| `resnet18` | `conv1 … layer2` | 128 |

All backbones are cut at **output stride 8**. The pixels of the feature map
are the samples of the covariance, so they must outnumber the channels: a
48×352 HyperLeaf image gives 6×44 = 264 samples at stride 8, but only 66 at
stride 16 (fewer than EfficientNet-B0's 80 channels there, hence a singular
sample covariance).

With `pretrained=True`, the ImageNet weights are downloaded once by torchvision
(into `~/.cache/torch`). With `in_channels != 3`, the first convolution is
replaced with the same kernel, stride and padding, and initialized with the
mean of the RGB filters.

## Covariance pooling

`CovariancePooling(n_features, estimator)` casts the features to float64 and
then estimates:

- `"scm"`: sample covariance
  (`yetanotherspdnet.functions.m_estimators.sample_covariance`);
- `"student"`: Student-t M-estimator (`yetanotherspdnet.nn.MEstimation`,
  ν = 5, 10 fixed-point iterations by default). It down-weights outlying
  pixels, and its backward uses implicit differentiation at the fixed point.

`eps · I` (1e-5 by default) is added to the result.

## Usage

Python:

```python
from spdnet_training.backbones import build_backbone_spdnet

model = build_backbone_spdnet(
    "efficientnet_b0",
    output_dim=4,
    estimator="student",           # or "scm"
    in_channels=3,
    hidden_layers_size=[32, 16],   # any SPDnet argument
    batchnorm=True,
    batchnorm_mean_type="geometric_arithmetic_harmonic",
)
logits = model(images)             # images: float32 (B, 3, H, W)
```

Hydra (`spdnet-train`): select `model=backbone_spdnet` with an image dataset
config, for example

```bash
spdnet-train model=backbone_spdnet model.backbone.backbone=resnet18 \
    model.backbone.estimator=student dataset=<hyperleaf image config>
```

Through Lightning, `trainer.precision=64` (derived from `model.dtype`) casts
the whole model, backbone included, to float64. That is correct but slower
than the mixed float32/float64 execution of plain PyTorch loops (e.g.
`spdnet-benchmark-demo`).

## Tests

`tests/test_backbones.py` builds everything with `pretrained=False` (no
download). It checks:

- widths and output stride, and the first convolution adapted to 5 bands;
- that pooling gives SPD float64 matrices with finite gradients, and that SCM
  pooling equals `torch.cov`;
- end-to-end training with finite gradients in both the backbone and the
  SPDnet head;
- that the Hydra config is dispatched by `SPDNetModule`.
