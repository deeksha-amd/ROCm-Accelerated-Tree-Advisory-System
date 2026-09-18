"""USA 1 km planting shortlist, from either SDM.

--model xgboost    per-species boosters in data/models_usa_30s/
--model deepmaxent one DeepMaxent network in data/models_deepmaxent_usa_30s/

Both read the same 61-layer 1 km stack and the same target-group training data,
so p means the same thing either way: record-likeness of this cell against an
average cell where some other listed tree was recorded. It is not planting
suitability. Invasive/naturalised checklist trees are dropped from the top-5.
Satellite filters come from data/country_data/USA/satellite_30s/ (1 km),
never from the 18 km data/satellite/ stack.

    python data/scripts/satellite_rasters_usa_30s.py --smoke
    python future_climate_usa_30s.py
    python recommend_usa_30s.py --lat 30.2672 --lon -97.7431 --html maps/austin.html
    python recommend_usa_30s.py --model deepmaxent --lat 30.2672 --lon -97.7431
    python suitability_maps_usa_30s.py --species "Quercus virginiana"
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import rasterio

_ROOT = os.path.abspath(os.path.dirname(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from poc.catalog import (
    MIN_AUC_DEFAULT,
    load_saved_models,
    load_traits_table,
    region_for_country,
)
from poc.recommend import (
    _pixel_value,
    bio_number,
    geocode,
    load_booster,
    native_label,
    reverse_geocode,
    sample_satellite,
    tag_cell_status,
)
from poc.recommend_html import write_html, _safe
from repo_paths import data
from clean_species_usa_30s import TEMPLATE, read_grid
from xgboost_training_usa_30s import MODEL_DIR as XGB_MODEL_DIR
from xgboost_training_usa_30s import collect_predictor_paths

import deepmaxent_sdm

TOP_N = 5
DEFAULT_MODEL = "xgboost"
USA_TRAITS_CSV = data("usa_tree_species_list.csv")
FUTURE_30S_DIR = data("country_data", "USA", "climate_future_2050_30s")
SATELLITE_DIR = data("country_data", "USA", "satellite_30s")
SATELLITE_FILES = {
    "exclusion": "planting_exclusion_mask_30s.tif",
    "water": "water_fraction_30s.tif",
    "snow": "snow_ice_fraction_30s.tif",
    "built": "built_up_fraction_30s.tif",
    "crop": "cropland_fraction_30s.tif",
    "tree": "tree_cover_fraction_30s.tif",
    "ndvi": "ndvi_mean_30s.tif",
}
DO_NOT_PLANT_RE = re.compile(
    r"\b(invasive|naturalised|naturalized|noxious)\b", re.I
)
SCORE_DISCLAIMER = (
    "p = record-likeness vs other listed trees. Not a permit. "
    "2050 = new BIO only. Land cover is a filter."
)
XGB_IMPORTANCE_CAPTION = (
    "Layers this tree's US model used most — not the reason for the score here."
)
DEEPMAXENT_IMPORTANCE_CAPTION = (
    "How this pin's score changes if each layer moves (local)."
)

BIO_PLAIN = {
    1: "yearly temperature",
    2: "day–night temperature range",
    3: "isothermality",
    4: "temperature seasonality",
    5: "hottest-month temperature",
    6: "coldest-month temperature",
    7: "yearly temperature range",
    8: "wet-season temperature",
    9: "dry-season temperature",
    10: "warm-quarter temperature",
    11: "cold-quarter temperature",
    12: "yearly rainfall",
    13: "wettest-month rainfall",
    14: "driest-month rainfall",
    15: "rainfall seasonality",
    16: "wet-quarter rainfall",
    17: "dry-quarter rainfall",
    18: "warm-quarter rainfall",
    19: "cold-quarter rainfall",
}
SOIL_PLAIN = {
    "ph_h2o": "soil pH",
    "sand": "sand",
    "silt": "silt",
    "clay": "clay",
    "bdod": "bulk density",
    "soc": "soil carbon",
    "socd": "carbon density",
    "nitrogen": "soil nitrogen",
    "cec": "cation exchange",
    "cfvo": "stones",
    "drainage": "drainage",
}
TERRAIN_PLAIN = {
    "elev_mean": "elevation",
    "elev_min": "lowest elevation in cell",
    "elev_max": "highest elevation in cell",
    "elev_range": "elevation range",
    "elev_std": "elevation variation",
    "slope": "slope",
    "steepness_q3": "upper-slope steepness",
    "steepness_max": "max steepness",
    "northness": "north-facing slopes",
    "eastness": "east-facing slopes",
    "terrain_vector_strength": "aspect consistency",
}

USE_TO_GOALS = {
    "street_park": "shade,beauty",
    "native_woodland": "shade,beauty",
    "timber": "shade,beauty",
    "fruit_nut": "food",
    "ornamental": "beauty",
    "agroforestry": "shade,food",
}
REGION_TO_NATIVE = {
    "north_america": "n_america",
    "europe": "europe",
    "asia": "e_asia",
    "oceania": "australia",
    "south_america": "s_america",
}
GRID_UI = {
    "label": "1 km",
    "marker": "Score is for this ~1 km cell, not a backyard.",
    "note": "One ~1 km cell. Yard shade, frost, and watering are not in the model.",
}


def load_usa_traits(species_list):
    """Catalog defaults, then the US plantable checklist (common name, use, invasive notes)."""
    table = load_traits_table(species_list).set_index("species")
    if not os.path.isfile(USA_TRAITS_CSV):
        return table.reset_index()
    extra = pd.read_csv(USA_TRAITS_CSV, comment="#", skipinitialspace=True)
    extra = extra.dropna(subset=["species"]).drop_duplicates("species")
    extra["species"] = extra["species"].astype(str).str.strip()
    extra = extra.set_index("species")
    for name, row in extra.iterrows():
        if name not in table.index:
            continue
        common = row.get("common_name")
        if pd.notna(common) and str(common).strip():
            table.at[name, "common_name"] = str(common).strip()
        use = row.get("use")
        if pd.notna(use) and str(use).strip() in USE_TO_GOALS:
            table.at[name, "goals"] = USE_TO_GOALS[str(use).strip()]
        region = row.get("region")
        if pd.notna(region) and str(region).strip() in REGION_TO_NATIVE:
            table.at[name, "native_regions"] = REGION_TO_NATIVE[str(region).strip()]
        notes = row.get("notes")
        if pd.notna(notes) and str(notes).strip():
            notes = str(notes).strip()
            if DO_NOT_PLANT_RE.search(notes):
                table.at[name, "warning"] = notes
                table.at[name, "invasive_in"] = "n_america"
            else:
                care = str(table.at[name, "care"] or "").strip()
                if care.lower() in {"", "nan", "none"}:
                    care = notes
                else:
                    care = f"{care} {notes}"
                table.at[name, "care"] = care
    return table.reset_index()


def do_not_plant(traits):
    """True when the US checklist marks the tree invasive/naturalised/noxious."""
    if str(traits.get("invasive_in") or "").strip():
        return True
    blob = " ".join(
        str(traits.get(key) or "")
        for key in ("warning", "care", "notes")
    )
    return bool(DO_NOT_PLANT_RE.search(blob))


def require_usa_rasters():
    """Clone-honest check: a git clone has no rasters (gitignored)."""
    if not os.path.isfile(TEMPLATE):
        raise SystemExit(
            "USA 1 km climate template missing:\n"
            f"  {TEMPLATE}\n"
            "A git clone does not include rasters (gitignored under "
            "data/country_data/). Copy climate_30s/, soil_30s/, and "
            "topography_30s/ from the machine that built the country stack."
        )


def confidence(p, auc, n_folds):
    if p >= 0.70 and auc >= 0.85 and n_folds >= 4:
        return "Looks like records"
    if p >= 0.50 and auc >= 0.75:
        return "Somewhat like records"
    return "Weak likeness"


def future_note(p, p_2050):
    if p_2050 is None or not np.isfinite(p_2050):
        return ""
    delta = p_2050 - p
    if p_2050 >= 0.70 and delta >= -0.05:
        return ""
    if delta <= -0.15:
        return f"Record-likeness drops {abs(delta):.0%} under 2050 BIO."
    if delta >= 0.10:
        return f"Record-likeness rises {delta:.0%} under 2050 BIO."
    return ""


def sample_predictors(lat, lon):
    layers = collect_predictor_paths()
    names = [n for n, _ in layers]
    values = np.array(
        [_pixel_value(path, lat, lon) for _, path in layers], dtype=np.float32
    )
    return names, values


def sample_future_usa(names, values_now, lat, lon, folder=FUTURE_30S_DIR):
    """Same 61-vector as today; 19 BIO swapped from 1 km 2050 delta rasters."""
    if not os.path.isdir(folder):
        return None, "missing 1 km 2050 folder — run python future_climate_usa_30s.py"
    out = np.array(values_now, dtype=np.float32, copy=True)
    n_swapped = 0
    for i, name in enumerate(names):
        k = bio_number(name)
        if k is None:
            continue
        path = os.path.join(folder, f"wc2.1_30s_bio_{k}.tif")
        if not os.path.isfile(path):
            return None, f"missing {path}"
        val = _pixel_value(path, lat, lon)
        if not np.isfinite(val):
            return None, f"nodata BIO{k}"
        out[i] = val
        n_swapped += 1
    if n_swapped != 19:
        return None, f"swapped {n_swapped} BIO layers, expected 19"
    return out, None


def plain_layer_name(filename):
    k = bio_number(filename)
    if k is not None:
        return BIO_PLAIN.get(k, f"climate BIO{k}")
    stem = os.path.basename(filename).replace(".tif", "")
    if stem in TERRAIN_PLAIN:
        return TERRAIN_PLAIN[stem]
    for key, label in SOIL_PLAIN.items():
        if stem.startswith(key + "_"):
            depth = "surface" if "0-30" in stem else "subsoil"
            return f"{label} ({depth})"
    return stem.replace("_", " ")


def importance_rows(weights, names, top_n=5):
    """Turn a per-layer weight vector into the top few shares the UI shows.
    What the weights mean is the backend's business — see the captions."""
    if weights is None:
        return []
    weights = np.asarray(weights, dtype=np.float64)
    if weights.size != len(names) or not np.isfinite(weights).any():
        return []
    weights = np.where(np.isfinite(weights), weights, 0.0)
    total = float(weights.sum())
    if total <= 0:
        return []
    rows = []
    for i in np.argsort(-weights)[:top_n]:
        share = float(weights[i] / total)
        if share < 0.01:
            continue
        rows.append({
            "name": names[i],
            "label": plain_layer_name(names[i]),
            "share": round(share, 4),
        })
    return rows


class XgboostBackend:
    """One gradient-boosted booster per species, loaded on demand."""

    kind = "xgboost"
    title = ""
    model_dir = XGB_MODEL_DIR
    metrics_path = os.path.join(XGB_MODEL_DIR, "metrics.csv")
    train_hint = "  python xgboost_training_usa_30s.py --full-list --skip-existing"
    importance_caption = XGB_IMPORTANCE_CAPTION
    missing_layer_note = (
        "The model skips those layers here; it does not fill in average soil."
    )

    def __init__(self):
        self._boosters = {}

    def check_runtime(self):
        if not os.path.isfile(self.metrics_path):
            raise SystemExit(
                f"Missing {self.metrics_path}. Train with:\n{self.train_hint}"
            )
        if not glob.glob(os.path.join(self.model_dir, "*.json")):
            raise SystemExit(
                f"No boosters in {self.model_dir}/*.json.\n"
                "metrics.csv is tracked; the JSON files are gitignored. "
                f"Copy them from the training machine, or train:\n{self.train_hint}"
            )

    def prepare(self, names):
        return None

    def _booster(self, rec):
        if rec.species not in self._boosters:
            self._boosters[rec.species] = load_booster(rec.model_path)
        return self._boosters[rec.species]

    def score(self, rec, x, tag="now"):
        return float(self._booster(rec).predict_proba(x)[0, 1])

    def importance(self, rec, names, values):
        try:
            gain = self._booster(rec).feature_importances_
        except Exception:
            return []
        return importance_rows(gain, names)


class DeepMaxentBackend:
    """One DeepMaxent network for every species, so a pin is a single forward
    pass; results are cached per predictor vector (today's, then 2050's)."""

    kind = "deepmaxent"
    title = ""
    model_dir = deepmaxent_sdm.MODEL_DIR
    metrics_path = deepmaxent_sdm.METRICS_PATH
    train_hint = deepmaxent_sdm.TRAIN_HINT
    importance_caption = DEEPMAXENT_IMPORTANCE_CAPTION
    missing_layer_note = (
        "The model fills those with typical US values."
    )

    def __init__(self, checkpoint_path=deepmaxent_sdm.CHECKPOINT_PATH):
        self.checkpoint_path = checkpoint_path
        self.sdm = None
        self._probabilities = {}

    def check_runtime(self):
        if not os.path.isfile(self.metrics_path):
            raise SystemExit(
                f"Missing {self.metrics_path}. Train with:\n{self.train_hint}"
            )
        if not os.path.isfile(self.checkpoint_path):
            raise SystemExit(
                f"No DeepMaxent checkpoint at {self.checkpoint_path}.\n"
                "metrics.csv is tracked; the .pt file is gitignored. "
                f"Copy it from the training machine, or train:\n{self.train_hint}"
            )

    def prepare(self, names):
        self.sdm = deepmaxent_sdm.DeepMaxentSDM.load(self.checkpoint_path)
        self.sdm.assert_feature_order(names)

    def _vector(self, x, tag):
        if tag not in self._probabilities:
            self._probabilities[tag] = self.sdm.predict_proba(x)[0]
        return self._probabilities[tag]

    def score(self, rec, x, tag="now"):
        column = self.sdm.index.get(rec.species)
        if column is None:
            return float("nan")
        return float(self._vector(x, tag)[column])

    def importance(self, rec, names, values):
        return importance_rows(self.sdm.sensitivity(rec.species, values), names)


def make_backend(kind):
    if kind == "xgboost":
        return XgboostBackend()
    if kind == "deepmaxent":
        return DeepMaxentBackend()
    raise SystemExit(f"Unknown --model {kind!r}; use xgboost or deepmaxent.")


def cell_geometry(lat, lon, path=TEMPLATE):
    info = {
        "west": None,
        "south": None,
        "east": None,
        "north": None,
        "status": "unknown",
        "color": "#5b6b76",
    }
    if not os.path.isfile(path):
        return info
    with rasterio.open(path) as src:
        row, col = src.index(lon, lat)
        if not (0 <= row < src.height and 0 <= col < src.width):
            return info
        west, north = src.xy(row, col, offset="ul")
        east, south = src.xy(row, col, offset="lr")
    info.update(
        west=float(west),
        south=float(south),
        east=float(east),
        north=float(north),
    )
    return info


def feat_match(names, token):
    token = token.lower()
    for i, name in enumerate(names):
        if token in name.lower():
            return i
    return None


def site_plain_language(names, values):
    bits = []

    def val(token):
        i = feat_match(names, token)
        if i is None or not np.isfinite(values[i]):
            return None
        return float(values[i])

    t = val("bio_1.tif")
    if t is not None:
        if t < 5:
            climate = "cold"
        elif t < 12:
            climate = "cool"
        elif t < 18:
            climate = "mild"
        elif t < 24:
            climate = "warm"
        else:
            climate = "hot"
        bits.append(f"{climate} (yearly average about {t:.0f}°C)")

    rain = val("bio_12.tif")
    if rain is not None:
        if rain < 400:
            wet = "dry"
        elif rain < 800:
            wet = "moderately wet"
        else:
            wet = "wet"
        bits.append(f"{wet} ({rain:.0f} mm rain/year)")

    ph = val("ph_h2o_0-30cm")
    if ph is not None:
        if ph < 5.5:
            soil = "acidic soil"
        elif ph < 7.3:
            soil = "roughly neutral soil"
        else:
            soil = "alkaline soil"
        bits.append(f"{soil} (pH {ph:.1f})")

    drain = val("drainage_0-30cm")
    if drain is not None:
        if drain < 3.5:
            bits.append("slow-draining ground")
        elif drain < 5.5:
            bits.append("average drainage")
        else:
            bits.append("well-drained ground")

    elev = val("elev_mean.tif")
    if elev is not None:
        bits.append(f"about {elev:.0f} m above sea level")

    if not bits:
        return "We could not read climate or soil at this pin."
    return "This location looks " + ", ".join(bits) + "."


def assert_conus(lat, lon):
    grid = read_grid()
    west, south, east, north = grid["bbox"]
    if not (south <= lat <= north and west <= lon <= east):
        raise SystemExit(
            f"Pin {lat:.4f}, {lon:.4f} is outside the lower-48 1 km grid "
            f"(bbox {west}, {south}, {east}, {north}). "
            "Use python poc/recommend.py for the global 18 km models."
        )


def recommend(lat, lon, goal="any", sun="any", top=TOP_N, min_auc=MIN_AUC_DEFAULT,
              country="", model=DEFAULT_MODEL):
    backend = model if hasattr(model, "kind") else make_backend(model)
    require_usa_rasters()
    backend.check_runtime()
    assert_conus(lat, lon)
    try:
        names, values = sample_predictors(lat, lon)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"{exc}\n"
            "A git clone does not include data/country_data/USA GeoTIFFs. "
            "Copy climate_30s/, soil_30s/, and topography_30s/ from the "
            "machine that built the country stack."
        ) from exc
    missing = [n for n, v in zip(names, values) if not np.isfinite(v)]
    n_bio_missing = sum(1 for n in missing if "bio_" in n)
    if n_bio_missing or len(missing) > 15:
        raise SystemExit(
            "No usable climate/soil/elevation at this pin "
            f"(missing {len(missing)} of {len(names)} USA 1 km layers"
            + (f", including {n_bio_missing} BIO" if n_bio_missing else "")
            + "). Try a nearby inland point."
        )
    missing_note = ""
    if missing:
        labels = [plain_layer_name(n) for n in missing[:6]]
        extra = "…" if len(missing) > 6 else ""
        missing_note = (
            f"{len(missing)} layer(s) have no value here "
            f"({', '.join(labels)}{extra}); "
            + backend.missing_layer_note
        )

    sat = sample_satellite(
        lat,
        lon,
        satellite_dir=SATELLITE_DIR,
        files=SATELLITE_FILES,
        missing_hint=(
            "No USA 1 km satellite filters yet — run "
            "python data/scripts/satellite_rasters_usa_30s.py. "
            "Skipping water/city checks."
        ),
    )
    if missing_note:
        sat.setdefault("notes", []).insert(0, missing_note)
    # Mix bar already shows built/crop fractions; keep only blockers and gaps.
    sat["notes"] = [
        n for n in sat.get("notes") or []
        if "18 km" not in n
        and "built-up city" not in n
        and "largely cropland" not in n
    ]
    site = site_plain_language(names, values)
    cell = tag_cell_status(cell_geometry(lat, lon), sat)

    saved = load_saved_models(min_auc=min_auc, metrics_path=backend.metrics_path)
    if saved.empty:
        raise SystemExit(
            f"No saved USA 1 km models in {backend.metrics_path} passed the AUC "
            f"gate (or their weights are missing under {backend.model_dir}).\n"
            f"Train with:\n{backend.train_hint}"
        )
    backend.prepare(names)

    traits_table = load_usa_traits(saved["species"].tolist())
    traits_table = traits_table.set_index("species")
    goal = (goal or "any").lower()
    sun = (sun or "any").lower()
    region = region_for_country(country) or "n_america"

    x = values.reshape(1, -1)
    values_2050, future_err = sample_future_usa(names, values, lat, lon)
    x2050 = None if values_2050 is None else values_2050.reshape(1, -1)
    scored = []
    avoided = []
    for rec in saved.itertuples(index=False):
        species = rec.species
        traits = (
            traits_table.loc[species].to_dict()
            if species in traits_table.index
            else {}
        )
        goals = {g.strip() for g in str(traits.get("goals", "")).split(",") if g.strip()}
        if goal != "any" and goal not in goals:
            continue
        sun_need = str(traits.get("sun", "full")).lower()
        if sun == "shade" and sun_need == "full":
            continue
        if sun == "full" and sun_need == "shade":
            continue

        p = backend.score(rec, x, tag="now")
        if not np.isfinite(p) or p < 0.30:
            continue
        if do_not_plant(traits):
            avoided.append({
                "species": species,
                "common_name": traits.get("common_name") or species,
                "p": p,
                "auc": float(rec.auc),
                "warning": (
                    "" if str(traits.get("warning") or "").lower() in {"", "nan", "none"}
                    else traits.get("warning")
                ),
            })
            continue
        p_2050 = None
        delta = None
        if x2050 is not None:
            p_2050 = backend.score(rec, x2050, tag="2050")
            delta = p_2050 - p
        scored.append({
            "species": species,
            "common_name": traits.get("common_name") or species,
            "p": p,
            "p_2050": p_2050,
            "delta": delta,
            "future_note": future_note(p, p_2050),
            "auc": float(rec.auc),
            "n_folds": int(rec.n_folds_usable),
            "rank_score": p * float(rec.auc),
            "confidence": confidence(p, float(rec.auc), int(rec.n_folds_usable)),
            "feature_importance": backend.importance(rec, names, values),
            "care": "" if str(traits.get("care") or "").lower() in {"", "nan", "none"} else traits.get("care"),
            "warning": "" if str(traits.get("warning") or "").lower() in {"", "nan", "none"} else traits.get("warning"),
            "native": native_label(traits, region),
            "goals": traits.get("goals") or "",
            "evergreen": traits.get("evergreen") or "",
        })

    scored.sort(key=lambda r: r["rank_score"], reverse=True)
    avoided.sort(key=lambda r: r["p"], reverse=True)
    return {
        "title": "Trees that look like recorded sites here",
        "lat": lat,
        "lon": lon,
        "country": country or "US",
        "site": site,
        "cell": cell,
        "grid": GRID_UI,
        "satellite": sat,
        "goal": goal,
        "sun": sun,
        "n_models": int(len(saved)),
        "model": backend.kind,
        "model_title": backend.title,
        "importance_caption": backend.importance_caption,
        "disclaimer": SCORE_DISCLAIMER,
        "future": {
            "available": values_2050 is not None,
            "path": FUTURE_30S_DIR,
            "period": "2041-2060",
            "ssp": "ssp245",
            "gcm": "MPI-ESM1-2-HR",
            "note": None if values_2050 is not None else (
                "No USA 1 km 2050 BIO — run python future_climate_usa_30s.py"
                + (f" ({future_err})" if future_err else "")
            ),
        },
        "picks": scored[:top],
        "avoid": avoided[:8],
    }


def print_report(result, address=None):
    print()
    print("=" * 64)
    print(
        "USA 1 km record-likeness shortlist"
        + (f"  ({result['model_title']})" if result.get("model_title") else "")
    )
    print("=" * 64)
    where = address or f"{result['lat']:.4f}, {result['lon']:.4f}"
    print(f"Place:  {where}")
    print(f"Lat/lon: {result['lat']:.4f}, {result['lon']:.4f}  (~1 km cell)")
    if result["country"]:
        print(f"Country code: {result['country']}")
    print(f"Goal:   {result['goal']}    Sun: {result['sun']}")
    print()
    print(result["site"])
    print(result["grid"]["note"])
    mix = result["satellite"].get("values") or {}
    mix_bits = []
    for key in ("water", "snow", "built", "crop", "tree"):
        val = mix.get(key)
        if val is not None and np.isfinite(val) and val >= 0.02:
            mix_bits.append(f"{key} {val:.0%}")
    if mix_bits:
        print("Land cover: " + ", ".join(mix_bits))
    for note in result["satellite"].get("notes", []):
        print(note)
    fut = result.get("future") or {}
    if fut.get("available"):
        print("2050: new BIO only; soil and terrain unchanged")
    else:
        print("2050: skipped — " + (fut.get("note") or ""))
    if result["satellite"].get("blocked"):
        print()
        print("No planting list — satellite/terrain says this pin is not plantable.")
        return

    if not result["picks"]:
        print()
        print("No species cleared the record-likeness + goal filters. Try --goal any or --min-auc 0.60.")
        print()
    else:
        print()
        print(f"Top {len(result['picks'])} of {result['n_models']} models (invasive/naturalised dropped):")
        print()
        caption = result.get("importance_caption") or ""
        if caption:
            print(caption)
        print()
        for i, pick in enumerate(result["picks"], 1):
            print(f"{i}. {pick['common_name']}  ({pick['species']})")
            print(
                f"   today {pick['p']:.0%}",
                end="",
            )
            if pick.get("p_2050") is not None:
                print(f"   2050 {pick['p_2050']:.0%}", end="")
            print(f"   model AUC {pick['auc']:.2f}")
            if pick.get("future_note"):
                print(f"   {pick['future_note']}")
            fi = pick.get("feature_importance") or []
            if fi:
                bits = [f"{row['label']} {row['share']:.0%}" for row in fi[:5]]
                print("   What this tree's model watches:")
                print("   " + " · ".join(bits))
            native_care = " ".join(
                part for part in (pick.get("native"), pick.get("care")) if part
            )
            if native_care:
                print("   Native range and care:")
                print(f"   {native_care}")
            if pick["warning"]:
                print(f"   Warning: {pick['warning']}")
            print()

    avoid = result.get("avoid") or []
    if avoid:
        print("Do not plant (cell looks like their records; listed invasive/naturalised):")
        for row in avoid:
            warn = f" — {row['warning']}" if row.get("warning") else ""
            print(
                f"   {row['common_name']} ({row['species']})  p={row['p']:.0%}{warn}"
            )
        print()

    print(result.get("disclaimer") or SCORE_DISCLAIMER)
    print()


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Recommend trees from the USA 1 km SDMs."
    )
    p.add_argument(
        "--model",
        choices=["xgboost", "deepmaxent"],
        default=DEFAULT_MODEL,
        help="which trained SDM to score with (default: %(default)s)",
    )
    p.add_argument("--lat", type=float)
    p.add_argument("--lon", type=float)
    p.add_argument("--address", type=str)
    p.add_argument(
        "--goal",
        choices=["any", "shade", "food", "beauty"],
        default="any",
    )
    p.add_argument(
        "--sun",
        choices=["any", "full", "part", "shade"],
        default="any",
    )
    p.add_argument("--top", type=int, default=TOP_N)
    p.add_argument("--min-auc", type=float, default=MIN_AUC_DEFAULT)
    p.add_argument("--json", action="store_true")
    p.add_argument(
        "--html",
        nargs="?",
        const="suggest_usa.html",
        default=None,
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    country = "US"
    address = args.address
    if args.address:
        lat, lon, country, display = geocode(args.address)
        address = display
        if country and country != "US":
            print(f"Note: geocode country is {country}; still scoring on the CONUS 1 km grid.")
    elif args.lat is not None and args.lon is not None:
        lat, lon = args.lat, args.lon
        country, display = reverse_geocode(lat, lon)
        if display:
            address = display
        country = country or "US"
    else:
        raise SystemExit("Pass --lat and --lon, or --address")

    result = recommend(
        lat,
        lon,
        goal=args.goal,
        sun=args.sun,
        top=args.top,
        min_auc=args.min_auc,
        country=country,
        model=args.model,
    )
    if args.html:
        path = write_html(result, args.html, address=address)
        print(f"Wrote {path}")
    if args.json:
        print(json.dumps(_safe(result), indent=2))
        return
    print_report(result, address=address)


if __name__ == "__main__":
    main()
