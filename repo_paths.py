"""Repo-root and data-directory locations.

USA 1 km scripts live at the repo root. The global 18 km POC lives in ``poc/``.
Downloaders live in ``data/scripts/``. Rasters, GBIF, and saved models live
under ``data/``. USA map HTML lives in ``maps/``; 18 km maps in ``poc/maps/``.

Importing this module puts the repo root on ``sys.path``. Paths returned here
are absolute so scripts work from any cwd.
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.dirname(__file__))
DATA = os.path.join(ROOT, "data")
MAPS = os.path.join(ROOT, "maps")
POC = os.path.join(ROOT, "poc")
POC_MAPS = os.path.join(POC, "maps")

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def data(*parts: str) -> str:
    return os.path.join(DATA, *parts)


def maps_dir(*parts: str) -> str:
    return os.path.join(MAPS, *parts)


def usa_maps(*parts: str) -> str:
    return os.path.join(MAPS, *parts)


def poc_maps(*parts: str) -> str:
    return os.path.join(POC_MAPS, *parts)


def relpath(path: str) -> str:
    """Store portable paths in metrics.csv (relative to the repo root)."""
    return os.path.relpath(path, ROOT)


def resolve(path: str) -> str:
    """Turn a metrics.csv path into an existing file, if possible."""
    if not isinstance(path, str) or not path.strip():
        return path
    if os.path.isfile(path):
        return os.path.abspath(path)
    joined = os.path.join(ROOT, path)
    if os.path.isfile(joined):
        return joined
    name = os.path.basename(path)
    for folder in (data("models"), data("models_usa_30s")):
        candidate = os.path.join(folder, name)
        if os.path.isfile(candidate):
            return candidate
    return path
