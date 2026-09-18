"""Checkpoint format and inference for the USA 1 km DeepMaxent Deep SDM.

Kept apart from deepmaxent_training_usa_30s.py so recommend_usa_30s.py can read
the paths below without importing torch — the XGBoost path must stay free of a
torch dependency. ``DeepMaxentSDM.load`` imports torch on first use.

One checkpoint holds every species: DeepMaxent is a single network with one
output per species, unlike the per-species XGBoost boosters in
data/models_usa_30s/.

Scores
------
The network emits a log-intensity lambda_s(x) that is only defined up to a
per-species constant (the maxent loss normalises over cells). Training stores

    log_z_s = log mean_over_training_cells exp(lambda_s)

so an average target-group cell sits at lambda_s - log_z_s = 0, and

    p_s(x) = sigmoid(lambda_s(x) - log_z_s)

is the record-likeness of this cell against an average cell where some other
listed tree was recorded. That is the same question the XGBoost boosters answer
with their 1:1 presence/target-group-background sampling, so p from either model
is read on the same scale.
"""

from __future__ import annotations

import os
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.dirname(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from repo_paths import data, relpath

CHECKPOINT_FORMAT = "deepmaxent-usa-30s/1"
MODEL_DIR = data("models_deepmaxent_usa_30s")
CHECKPOINT_NAME = "deepmaxent_usa_30s.pt"
CHECKPOINT_PATH = os.path.join(MODEL_DIR, CHECKPOINT_NAME)
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.csv")
FEATURE_NAMES_PATH = os.path.join(MODEL_DIR, "feature_names.txt")

UPSTREAM = {
    "repo": "https://github.com/RYCKEWAERT/deepmaxent",
    "commit": "3587ad743b3c1898f61ac1c1c5f8b2884b750db4",
}

TRAIN_HINT = (
    "  python deepmaxent_training_usa_30s.py --full-list"
)


def save_checkpoint(
    path,
    *,
    state_dict,
    species,
    feature_names,
    hidden_size,
    hidden_nbr,
    scaler_mean,
    scaler_scale,
    log_z,
    training,
):
    """Write the one-file checkpoint. Tensors/lists only, so torch.load can
    stay on weights_only=True."""
    import torch

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "upstream": dict(UPSTREAM),
        "species": list(species),
        "feature_names": list(feature_names),
        "input_size": int(len(feature_names)),
        "output_size": int(len(species)),
        "hidden_size": int(hidden_size),
        "hidden_nbr": int(hidden_nbr),
        "state_dict": {k: v.detach().cpu() for k, v in state_dict.items()},
        "scaler_mean": torch.as_tensor(np.asarray(scaler_mean), dtype=torch.float32),
        "scaler_scale": torch.as_tensor(np.asarray(scaler_scale), dtype=torch.float32),
        "log_z": torch.as_tensor(np.asarray(log_z), dtype=torch.float32),
        "training": dict(training),
    }
    torch.save(payload, path)
    return path


class DeepMaxentSDM:
    """Trained network + the standardiser and log_z needed to read its output."""

    def __init__(self, payload, device="cpu"):
        import torch

        from deepmaxent import deepmaxent_model

        fmt = payload.get("format")
        if fmt != CHECKPOINT_FORMAT:
            raise SystemExit(
                f"Checkpoint format {fmt!r}, expected {CHECKPOINT_FORMAT!r}. "
                "Retrain with deepmaxent_training_usa_30s.py."
            )
        self.device = torch.device(device)
        self.species = list(payload["species"])
        self.feature_names = list(payload["feature_names"])
        self.index = {name: i for i, name in enumerate(self.species)}
        self.training = dict(payload.get("training") or {})
        self.upstream = dict(payload.get("upstream") or {})

        self.model = deepmaxent_model(
            int(payload["input_size"]),
            int(payload["hidden_size"]),
            int(payload["output_size"]),
            int(payload["hidden_nbr"]),
        )
        self.model.load_state_dict(payload["state_dict"])
        self.model.to(self.device).eval()

        self._mean = payload["scaler_mean"].to(self.device)
        self._scale = payload["scaler_scale"].to(self.device)
        self._log_z = payload["log_z"].to(self.device)

    @classmethod
    def load(cls, path=CHECKPOINT_PATH, device="cpu"):
        import torch

        if not os.path.isfile(path):
            raise SystemExit(
                f"No DeepMaxent checkpoint at {path}. Train with:\n{TRAIN_HINT}"
            )
        payload = torch.load(path, map_location="cpu", weights_only=True)
        return cls(payload, device=device)

    def assert_feature_order(self, names):
        """The 61-layer order is the training contract; a silent reorder would
        score every species against the wrong rasters."""
        if list(names) != self.feature_names:
            n_shown = 4
            raise SystemExit(
                "Predictor order does not match the DeepMaxent checkpoint.\n"
                f"  checkpoint: {self.feature_names[:n_shown]} … "
                f"({len(self.feature_names)} layers)\n"
                f"  caller:     {list(names)[:n_shown]} … ({len(names)} layers)\n"
                "Retrain after changing collect_predictor_paths()."
            )

    def _standardise(self, values):
        """Raw layer values -> z-scores. Layers with no value in this cell fall
        back to the training mean (z = 0); XGBoost instead learns its own
        default direction for missing values."""
        import torch

        x = torch.as_tensor(
            np.atleast_2d(np.asarray(values, dtype=np.float32)), device=self.device
        )
        z = (x - self._mean) / self._scale
        return torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

    def log_intensity(self, values):
        import torch

        with torch.no_grad():
            return self.model(self._standardise(values))

    def predict_proba(self, values):
        """(n, 61) raw layer values -> (n, n_species) record-likeness in [0, 1]."""
        import torch

        with torch.no_grad():
            lam = self.model(self._standardise(values))
            return torch.sigmoid(lam - self._log_z).cpu().numpy()

    def sensitivity(self, species, values):
        """|d lambda_s / d z_j| at this cell: which layers move this species'
        score here, per 1 SD of layer j. This is a local gradient, not the
        species-wide gain XGBoost reports."""
        import torch

        if species not in self.index:
            return None
        z = self._standardise(values).clone().requires_grad_(True)
        lam = self.model(z)[0, self.index[species]]
        (grad,) = torch.autograd.grad(lam, z)
        return np.abs(grad[0].detach().cpu().numpy())
