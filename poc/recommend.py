"""POC: turn saved SDM models into a non-technical planting shortlist.

Plug in more GBIF later
-----------------------
1. Retrain: python poc/xgboost_training.py   # writes data/models/*.json + metrics.csv
2. Optional: add a row in data/species_traits.csv (common name, goals, warnings)
3. Re-run this script. New models are picked up automatically.

Example
-------
python data/scripts/satellite_rasters.py                    # once; site filters
python poc/recommend.py --lat 51.51 --lon -0.13 --goal shade
python poc/recommend.py --lat 51.51 --lon -0.13 --goal shade --html poc/maps/suggest.html
python poc/recommend.py --address "Portland, Oregon" --goal beauty --sun part
python poc/recommend.py --lat 0 --lon -150                  # ocean → blocked
python poc/suitability_maps.py --species "Quercus robur"    # today vs 2050 map
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import xgboost as xgb

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from poc.catalog import (
    MIN_AUC_DEFAULT,
    load_saved_models,
    load_traits_table,
    region_for_country,
    write_traits_stub,
)
from poc.recommend_html import write_html
from poc.xgboost_training import collect_predictor_paths
from repo_paths import data

SATELLITE_DIR = data("satellite")
TOP_N = 5
FUTURE_CLIMATE_PATH = data(
    "climate_future_2050",
    "wc2.1_10m_bioc_MPI-ESM1-2-HR_ssp245_2041-2060.tif",
)
TEMPLATE_RASTER = data("climate_current", "wc2.1_10m_bio_1.tif")
BIO_RE = re.compile(r"bio_(\d+)\.tif$", re.I)

# Written by satellite_rasters.py. Substring fallback still works if you add layers.
SATELLITE_FILES = {
    "exclusion": "planting_exclusion_mask_10m.tif",
    "water": "water_fraction_10m.tif",
    "snow": "snow_ice_fraction_10m.tif",
    "built": "built_up_fraction_10m.tif",
    "crop": "cropland_fraction_10m.tif",
    "tree": "tree_cover_fraction_10m.tif",
    "ndvi": "ndvi_mean_10m.tif",
}


def geocode(address):
    """Nominatim lookup. Returns lat, lon, country_code, display_name."""
    query = urllib.parse.urlencode(
        {"q": address, "format": "json", "limit": 1, "addressdetails": 1}
    )
    url = f"https://nominatim.openstreetmap.org/search?{query}"
    req = urllib.request.Request(
        url, headers={"User-Agent": "hackathon-tree-poc/0.1 (research)"}
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        hits = json.loads(resp.read().decode("utf-8"))
    if not hits:
        raise SystemExit(f"Could not geocode: {address!r}")
    hit = hits[0]
    addr = hit.get("address") or {}
    country = (addr.get("country_code") or "").upper()
    return float(hit["lat"]), float(hit["lon"]), country, hit.get("display_name", address)


def _pixel_value(path, lat, lon):
    with rasterio.open(path) as src:
        row, col = src.index(lon, lat)
        if not (0 <= row < src.height and 0 <= col < src.width):
            return np.nan
        band = src.read(1, window=Window(col, row, 1, 1))
        val = float(band[0, 0])
        if src.nodata is not None and val == src.nodata:
            return np.nan
        if abs(val) > 1e30 or not np.isfinite(val):
            return np.nan
        return val


def reverse_geocode(lat, lon):
    """Best-effort country for native/invasive labels when only lat/lon is given."""
    query = urllib.parse.urlencode(
        {"lat": lat, "lon": lon, "format": "json", "addressdetails": 1}
    )
    url = f"https://nominatim.openstreetmap.org/reverse?{query}"
    req = urllib.request.Request(
        url, headers={"User-Agent": "hackathon-tree-poc/0.1 (research)"}
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            hit = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return "", ""
    addr = hit.get("address") or {}
    country = (addr.get("country_code") or "").upper()
    return country, hit.get("display_name", "")


def sample_predictors(lat, lon):
    """Read one cell from each training layer. Does not load full rasters."""
    layers = collect_predictor_paths()
    names = [n for n, _ in layers]
    values = np.array(
        [_pixel_value(path, lat, lon) for _, path in layers], dtype=np.float32
    )
    return names, values


def bio_number(name):
    m = BIO_RE.search(name)
    return int(m.group(1)) if m else None


def sample_future_values(names, values_now, lat, lon, path=FUTURE_CLIMATE_PATH):
    """Same 42-vector as today, but BIO swapped from the 2050 19-band GeoTIFF.

    Band k is BIO k. Training order is filename sort (bio_10 before bio_2).
    Soil and elevation are copied from values_now. Missing file → (None, reason).
    """
    if not os.path.isfile(path):
        return None, "missing file"
    with rasterio.open(path) as src:
        row, col = src.index(lon, lat)
        if not (0 <= row < src.height and 0 <= col < src.width):
            return None, "out of bounds"
        win = Window(col, row, 1, 1)
        cube = src.read(window=win)
        nodata = src.nodata
    out = np.array(values_now, dtype=np.float32, copy=True)
    for i, name in enumerate(names):
        k = bio_number(name)
        if k is None:
            continue
        if k < 1 or k > cube.shape[0]:
            return None, f"BIO{k} past band count {cube.shape[0]}"
        val = float(cube[k - 1, 0, 0])
        if nodata is not None and val == nodata:
            return None, "nodata"
        if abs(val) > 1e30 or not np.isfinite(val):
            return None, "nodata"
        out[i] = val
    return out, None


def cell_geometry(lat, lon, path=TEMPLATE_RASTER):
    """1/6° WorldClim cell that contains the pin, plus a plantable/city/blocked flag."""
    info = {"west": None, "south": None, "east": None, "north": None,
            "status": "unknown", "color": "#5b6b76"}
    if not os.path.isfile(path):
        return info
    with rasterio.open(path) as src:
        row, col = src.index(lon, lat)
        if not (0 <= row < src.height and 0 <= col < src.width):
            return info
        west, north = src.xy(row, col, offset="ul")
        east, south = src.xy(row, col, offset="lr")
    info.update(west=float(west), south=float(south),
                east=float(east), north=float(north))
    return info


def tag_cell_status(cell, sat):
    if sat.get("blocked"):
        cell["status"] = "blocked"
        cell["color"] = "#9b2c2c"
        return cell
    vals = sat.get("values") or {}
    built = vals.get("built")
    crop = vals.get("crop")
    if built is not None and np.isfinite(built) and built >= 0.4:
        cell["status"] = "city"
        cell["color"] = "#b7791f"
    elif crop is not None and np.isfinite(crop) and crop >= 0.5:
        cell["status"] = "farm"
        cell["color"] = "#b7791f"
    else:
        cell["status"] = "plantable"
        cell["color"] = "#1f6f4a"
    return cell


def future_note(p, p_2050):
    if p_2050 is None or not np.isfinite(p_2050):
        return ""
    delta = p_2050 - p
    if p_2050 >= 0.70 and delta >= -0.05:
        return "Still a strong climate match in 2050 (ssp245)."
    if delta <= -0.15:
        return f"Climate match drops by {abs(delta):.0%} by 2050 — same soil, warmer climate."
    if delta >= 0.10:
        return f"Climate match rises by {delta:.0%} by 2050 as this cell warms."
    return "2050 climate is close enough that the ranking barely moves."


def _as_fraction(val):
    """WorldCover outputs 0–1; some products store percent 0–100."""
    if not np.isfinite(val):
        return val
    if val > 1.5:
        return float(val) / 100.0
    return float(val)


def sample_satellite(
    lat,
    lon,
    satellite_dir=None,
    files=None,
    missing_hint=None,
    grain_note=None,
):
    """Optional site filter. Missing folder → no-op (POC still runs).

    Never feeds pixels into XGBoost. Default reads data/satellite/*.tif on the
    shared 10-arc-minute grid. Pass satellite_dir/files for another grid
    (USA 1 km lives in data/country_data/USA/satellite_30s/).
    """
    satellite_dir = satellite_dir or SATELLITE_DIR
    files = files or SATELLITE_FILES
    info = {"blocked": False, "notes": [], "values": {}}
    if not os.path.isdir(satellite_dir):
        info["notes"].append(
            missing_hint
            or (
                "No data/satellite/ folder yet — run python data/scripts/satellite_rasters.py, "
                "then re-run. Skipping water/city/exclusion checks."
            )
        )
        return info

    tifs = []
    for root, _, names in os.walk(satellite_dir):
        if os.path.sep + "raw" + os.path.sep in root + os.path.sep:
            continue
        for name in names:
            if name.endswith(".tif"):
                tifs.append(os.path.join(root, name))

    def find(key, extra_parts=()):
        exact = os.path.join(satellite_dir, files.get(key, ""))
        if files.get(key) and os.path.isfile(exact):
            return exact
        parts = (key,) + tuple(extra_parts)
        parts = tuple(p.lower() for p in parts)
        for path in tifs:
            low = os.path.basename(path).lower()
            if key == "tree" and ("broadleaf" in low or "needle" in low):
                continue
            if all(p in low for p in parts):
                return path
        return None

    def read_layer(key, extra_parts=()):
        path = find(key, extra_parts)
        if not path:
            return None
        val = _as_fraction(_pixel_value(path, lat, lon))
        if np.isfinite(val):
            info["values"][key] = val
            return val
        return None

    if not tifs:
        info["notes"].append(
            missing_hint
            or (
                "data/satellite/ exists but has no GeoTIFFs — "
                "run python data/scripts/satellite_rasters.py."
            )
        )
        return info

    excl = read_layer("exclusion", ("planting_exclusion",))
    if excl is not None and excl > 0.5:
        info["blocked"] = True
        info["notes"].append(
            "Satellite exclusion mask says this cell is not plantable "
            "(water, ice, wetland, or unmapped ocean)."
        )

    for label, key, thresh, msg in (
        ("water", "water", 0.5, "This cell is mostly water."),
        ("snow", "snow", 0.5, "This cell is mostly snow or ice."),
        ("built-up", "built", 0.4, "This cell is largely built-up city."),
        ("cropland", "crop", 0.5, "This cell is largely cropland (open ground, not forest)."),
    ):
        val = read_layer(key)
        if val is None:
            continue
        if val >= thresh:
            if label in {"water", "snow"}:
                info["blocked"] = True
            info["notes"].append(f"{msg} (satellite {label} ≈ {val:.0%})")

    tree_val = read_layer("tree")
    if tree_val is not None and tree_val >= 0.4:
        info["notes"].append(
            f"This cell already has substantial tree cover (≈ {tree_val:.0%}). "
            "Scores say what *could* grow, not that you should clear forest."
        )
    else:
        ndvi_val = read_layer("ndvi")
        if ndvi_val is not None and ndvi_val >= 0.6:
            info["notes"].append(
                "Satellite greenness is high here — the cell is already vegetated."
            )
    if tifs and not info["values"]:
        info["notes"].append(
            "Satellite rasters exist but this cell is unmapped "
            "(partial/smoke run, or nodata)."
        )
    elif grain_note and info["values"]:
        info["notes"].insert(0, grain_note)
    return info


def feat_index(names, suffix):
    for i, name in enumerate(names):
        if name.endswith(suffix):
            return i
    return None


def site_plain_language(names, values):
    """Translate the 42 numbers into a short place description."""
    bits = []

    def val(suffix):
        i = feat_index(names, suffix)
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

    ph = val("ph_h2o_0-30cm_10m.tif")
    if ph is not None:
        if ph < 5.5:
            soil = "acidic soil"
        elif ph < 7.3:
            soil = "roughly neutral soil"
        else:
            soil = "alkaline soil"
        bits.append(f"{soil} (pH {ph:.1f})")

    drain = val("drainage_0-30cm_10m.tif")
    if drain is not None:
        if drain < 3.5:
            bits.append("slow-draining ground")
        elif drain < 5.5:
            bits.append("average drainage")
        else:
            bits.append("well-drained ground")

    elev = val("gedtm30_v1.2_elev_10m.tif")
    if elev is not None:
        bits.append(f"about {elev:.0f} m above sea level")

    if not bits:
        return "We could not read climate or soil at this pin."
    return "This location looks " + ", ".join(bits) + "."


def reason_for(species, traits, p, names, values):
    """One or two sentences a non-technical user can use."""
    common = traits.get("common_name") or species
    t = None
    i = feat_index(names, "bio_1.tif")
    if i is not None and np.isfinite(values[i]):
        t = float(values[i])
    rain = None
    i = feat_index(names, "bio_12.tif")
    if i is not None and np.isfinite(values[i]):
        rain = float(values[i])
    drain = None
    i = feat_index(names, "drainage_0-30cm_10m.tif")
    if i is not None and np.isfinite(values[i]):
        drain = float(values[i])

    climate_bit = "your climate"
    if t is not None and rain is not None:
        if t >= 18 and rain < 500:
            climate_bit = "your warm, dry climate"
        elif t >= 18:
            climate_bit = "your warm climate"
        elif t < 8:
            climate_bit = "your cool climate"
        elif rain < 450:
            climate_bit = "your relatively dry climate"
        else:
            climate_bit = "your climate"

    drain_bit = ""
    if drain is not None:
        if drain >= 5.5:
            drain_bit = " and well-drained ground"
        elif drain < 3.5:
            drain_bit = ", though the soil here holds water"

    if p >= 0.70:
        lead = f"{common} is a strong climate-and-soil match for {climate_bit}{drain_bit}."
    elif p >= 0.50:
        lead = f"{common} can work in {climate_bit}{drain_bit}, with some care."
    else:
        lead = f"{common} is only a borderline match for {climate_bit}{drain_bit}."
    return lead


def confidence(p, auc, n_folds):
    if p >= 0.70 and auc >= 0.85 and n_folds >= 4:
        return "Great match"
    if p >= 0.50 and auc >= 0.75:
        return "Good match"
    return "Risky"


def native_label(traits, region):
    native = {x.strip() for x in str(traits.get("native_regions", "")).split(",") if x.strip()}
    invasive = {x.strip() for x in str(traits.get("invasive_in", "")).split(",") if x.strip()}
    if region and region in invasive:
        return "Not native here — often invasive. Check before you plant."
    if region and region in native:
        return "Fits this region’s native range (coarse check)."
    if region:
        return "Not native to this region — fine as an ornamental if local rules allow."
    return "Native-range check skipped (no country on this pin)."


def load_booster(path, device="cpu"):
    model = xgb.XGBClassifier()
    model.load_model(path)
    # Pin scoring is one 61-vector — keep CPU. Grid maps pass device=cuda.
    try:
        model.set_params(device=device)
    except Exception:
        pass
    return model


def recommend(lat, lon, goal="any", sun="any", top=TOP_N, min_auc=MIN_AUC_DEFAULT,
              country=""):
    names, values = sample_predictors(lat, lon)
    if np.isnan(values).any():
        missing = [n for n, v in zip(names, values) if not np.isfinite(v)]
        raise SystemExit(
            "No complete climate/soil/elevation at this pin "
            f"(missing {len(missing)} layers). Try a nearby inland point."
        )

    sat = sample_satellite(lat, lon)
    site = site_plain_language(names, values)
    cell = tag_cell_status(cell_geometry(lat, lon), sat)
    values_2050, future_err = sample_future_values(names, values, lat, lon)
    x2050 = None if values_2050 is None else values_2050.reshape(1, -1)

    saved = load_saved_models(min_auc=min_auc)
    if saved.empty:
        raise SystemExit("No saved models passed the AUC gate. Train first.")

    traits_table = load_traits_table(saved["species"].tolist())
    traits_table = traits_table.set_index("species")
    goal = (goal or "any").lower()
    sun = (sun or "any").lower()
    region = region_for_country(country)

    x = values.reshape(1, -1)
    scored = []
    for rec in saved.itertuples(index=False):
        species = rec.species
        traits = traits_table.loc[species].to_dict() if species in traits_table.index else {}
        goals = {g.strip() for g in str(traits.get("goals", "")).split(",") if g.strip()}
        if goal != "any" and goal not in goals:
            continue
        sun_need = str(traits.get("sun", "full")).lower()
        if sun == "shade" and sun_need == "full":
            continue
        if sun == "full" and sun_need == "shade":
            continue

        model = load_booster(rec.model_path)
        p = float(model.predict_proba(x)[0, 1])
        if p < 0.30:
            continue
        p_2050 = None
        delta = None
        if x2050 is not None:
            p_2050 = float(model.predict_proba(x2050)[0, 1])
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
            "reason": reason_for(species, traits, p, names, values),
            "care": traits.get("care") or "",
            "warning": traits.get("warning") or "",
            "native": native_label(traits, region),
            "goals": traits.get("goals") or "",
            "evergreen": traits.get("evergreen") or "",
        })

    scored.sort(key=lambda r: r["rank_score"], reverse=True)
    return {
        "lat": lat,
        "lon": lon,
        "country": country,
        "site": site,
        "cell": cell,
        "satellite": sat,
        "goal": goal,
        "sun": sun,
        "n_models": int(len(saved)),
        "future": {
            "available": values_2050 is not None,
            "path": FUTURE_CLIMATE_PATH,
            "period": "2041-2060",
            "ssp": "ssp245",
            "gcm": "MPI-ESM1-2-HR",
            "note": None if values_2050 is not None else (
                "No 2050 raster — run python data/scripts/future_climate_rasters.py"
                + (f" ({future_err})" if future_err else "")
            ),
        },
        "picks": scored[:top],
    }


def print_report(result, address=None):
    print()
    print("=" * 64)
    print("Tree suggestions for this location")
    print("=" * 64)
    where = address or f"{result['lat']:.4f}, {result['lon']:.4f}"
    print(f"Place:  {where}")
    if result["country"]:
        print(f"Country code: {result['country']}")
    print(f"Goal:   {result['goal']}    Sun: {result['sun']}")
    print()
    print(result["site"])
    print(
        "Note: this is an ~18 km climate cell, not a single backyard. "
        "Your yard’s shade and watering still matter."
    )
    mix = result["satellite"].get("values") or {}
    mix_bits = []
    for key in ("water", "snow", "built", "crop", "tree"):
        val = mix.get(key)
        if val is not None and np.isfinite(val):
            mix_bits.append(f"{key} {val:.0%}")
    if mix_bits:
        print("Satellite mix: " + ", ".join(mix_bits))
    for note in result["satellite"].get("notes", []):
        print(f"Satellite: {note}")
    fut = result.get("future") or {}
    if not fut.get("available"):
        print("2050: skipped — " + (fut.get("note") or "future climate file missing"))
    if result["satellite"].get("blocked"):
        print()
        print("No planting list — satellite/terrain says this pin is not plantable.")
        return

    if not result["picks"]:
        print()
        print("No species cleared the match + goal filters. Try --goal any.")
        return

    print()
    print(f"Top {len(result['picks'])} of {result['n_models']} climate-vetted trees:")
    print()
    for i, pick in enumerate(result["picks"], 1):
        print(f"{i}. {pick['common_name']}  ({pick['species']})")
        print(f"   {pick['confidence']}   today {pick['p']:.0%}", end="")
        if pick.get("p_2050") is not None:
            print(f"   2050 {pick['p_2050']:.0%}   model AUC {pick['auc']:.2f}")
        else:
            print(f"   model AUC {pick['auc']:.2f}")
        print(f"   {pick['reason']}")
        if pick.get("future_note"):
            print(f"   2050: {pick['future_note']}")
        native_care = " ".join(
            part for part in (pick.get("native"), pick.get("care")) if part
        )
        if native_care:
            print("   Native range and care:")
            print(f"   {native_care}")
        if pick["warning"]:
            print(f"   Warning: {pick['warning']}")
        print()


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Recommend trees from saved SDM models (POC)."
    )
    p.add_argument("--lat", type=float, help="Latitude")
    p.add_argument("--lon", type=float, help="Longitude")
    p.add_argument("--address", type=str, help="Geocode this instead of lat/lon")
    p.add_argument(
        "--goal",
        choices=["any", "shade", "food", "beauty"],
        default="any",
        help="Filter species cards by use",
    )
    p.add_argument(
        "--sun",
        choices=["any", "full", "part", "shade"],
        default="any",
        help="Yard sun (the 18 km model cannot see this)",
    )
    p.add_argument("--top", type=int, default=TOP_N)
    p.add_argument("--min-auc", type=float, default=MIN_AUC_DEFAULT)
    p.add_argument("--json", action="store_true", help="Machine-readable output")
    p.add_argument(
        "--html",
        nargs="?",
        const="poc/maps/suggest.html",
        default=None,
        help="Write a Leaflet demo page (default path: poc/maps/suggest.html)",
    )
    p.add_argument(
        "--write-traits-stub",
        action="store_true",
        help="Write data/species_traits.csv for the current saved models and exit",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.write_traits_stub:
        saved = load_saved_models(min_auc=0.0)
        path = write_traits_stub(saved["species"].tolist())
        print(f"Wrote {path} ({len(saved)} species). Edit goals/care, then re-run.")
        return

    country = ""
    address = args.address
    if args.address:
        lat, lon, country, display = geocode(args.address)
        address = display
    elif args.lat is not None and args.lon is not None:
        lat, lon = args.lat, args.lon
        country, display = reverse_geocode(lat, lon)
        if display:
            address = display
    else:
        raise SystemExit("Pass --lat and --lon, or --address")

    result = recommend(
        lat, lon,
        goal=args.goal,
        sun=args.sun,
        top=args.top,
        min_auc=args.min_auc,
        country=country,
    )
    if args.html:
        path = write_html(result, args.html, address=address)
        print(f"Wrote {path}")
    if args.json:
        from recommend_html import _safe
        print(json.dumps(_safe(result), indent=2))
        return
    print_report(result, address=address)


if __name__ == "__main__":
    main()
