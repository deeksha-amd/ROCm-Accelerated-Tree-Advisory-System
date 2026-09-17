"""Global 18 km (10-arc-minute) planting POC.

USA 1 km scripts stay at the repo root. Run these from the project root:

    python poc/xgboost_training.py
    python poc/recommend.py --lat 51.51 --lon -0.13 --html poc/maps/suggest.html
    python poc/suitability_maps.py --species "Quercus robur"
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
