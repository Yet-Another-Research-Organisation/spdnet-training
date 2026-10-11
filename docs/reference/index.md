# API reference

| Object | Role |
|---|---|
| {py:class}`~spdnet_training.SPDNetModule` | Lightning module wrapping an SPDnet (or a backbone + SPDnet) |
| {doc}`backbones` | `build_backbone_spdnet`, `CovariancePooling`, `BackboneSPDnet` |
| {doc}`callbacks` | Plotting, CSV metrics, results, rich console, covariance analysis, Optuna pruning |
| {doc}`utils` | Warm-up + plateau scheduler, metrics writer, precision helper |

```{eval-rst}
.. autoclass:: spdnet_training.SPDNetModule
   :members: forward
```

```{toctree}
:hidden:

backbones
callbacks
utils
```
