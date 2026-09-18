"""Vendored DeepMaxent network and losses.

Upstream: https://github.com/RYCKEWAERT/deepmaxent
Commit:   3587ad743b3c1898f61ac1c1c5f8b2884b750db4 (2026-02-28)
Files:    librairies/model.py, librairies/losses.py — copied verbatim so the
          Deep SDM here is the author's model, not a re-implementation.

The upstream training driver (librairies/train_models_elith.py) is tied to the
Elith/NCEAS folder layout, so it is not vendored. deepmaxent_training_usa_30s.py
drives these classes on the USA 1 km stack instead.
"""

from deepmaxent.losses import bce_loss, ce_loss, deepmaxent_loss, poisson_loss
from deepmaxent.model import deepmaxent_model

__all__ = [
    "deepmaxent_model",
    "deepmaxent_loss",
    "poisson_loss",
    "bce_loss",
    "ce_loss",
]
