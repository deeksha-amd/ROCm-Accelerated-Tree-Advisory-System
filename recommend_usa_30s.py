"""USA 1 km planting shortlist from data/models_usa_30s/.

Does not import or overwrite poc/recommend.py (the global 18 km POC).

    python future_climate_usa_30s.py
    python recommend_usa_30s.py --lat 30.2672 --lon -97.7431 --html maps/austin.html
    python suitability_maps_usa_30s.py --species "Quercus virginiana"
"""

from __future__ import annotations

import argparse
import json
import os
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
    confidence,
    future_note,
    geocode,
    load_booster,
    native_label,
    reverse_geocode,
    sample_satellite,
    tag_cell_status,
)
from poc.recommend_html import write_html
from repo_paths import data
from clean_species_usa_30s import TEMPLATE, read_grid
from xgboost_training_usa_30s import MODEL_DIR, collect_predictor_paths

TOP_N = 5
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.csv")
USA_TRAITS_CSV = data("usa_tree_species_list.csv")
FUTURE_30S_DIR = data("country_data", "USA", "climate_future_2050_30s")

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
    "marker": "Score is for this ~1 km climate cell, not a backyard.",
    "note": (
        "The box on the map is one 30-arc-second cell (~1 km). "
        "Yard shade, frost pockets and watering are still not in the model."
    ),
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
            low = notes.lower()
            if "invasive" in low:
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


def feature_importance_rows(model, names, top_n=5):
    """Global gain share for this species — not a local explanation of the pin."""
    try:
        gain = np.asarray(model.feature_importances_, dtype=np.float64)
    except Exception:
        return []
    if gain.size != len(names) or not np.isfinite(gain).any():
        return []
    total = float(gain.sum())
    if total <= 0:
        return []
    order = np.argsort(-gain)[:top_n]
    rows = []
    for i in order:
        share = float(gain[i] / total)
        if share < 0.01:
            continue
        rows.append({
            "name": names[i],
            "label": plain_layer_name(names[i]),
            "share": round(share, 4),
        })
    return rows


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


def reason_for(species, traits, p, names, values):
    common = traits.get("common_name") or species
    t = None
    i = feat_match(names, "bio_1.tif")
    if i is not None and np.isfinite(values[i]):
        t = float(values[i])
    rain = None
    i = feat_match(names, "bio_12.tif")
    if i is not None and np.isfinite(values[i]):
        rain = float(values[i])
    drain = None
    i = feat_match(names, "drainage_0-30cm")
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


def assert_conus(lat, lon):
    grid = read_grid()
    west, south, east, north = grid["bbox"]
    if not (south <= lat <= north and west <= lon <= east):
        raise SystemExit(
            f"Pin {lat:.4f}, {lon:.4f} is outside the lower-48 1 km grid "
            f"(bbox {west}, {south}, {east}, {north}). "
            "Use recommend.py for the global 18 km models."
        )


def recommend(lat, lon, goal="any", sun="any", top=TOP_N, min_auc=MIN_AUC_DEFAULT,
              country=""):
    assert_conus(lat, lon)
    names, values = sample_predictors(lat, lon)
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
        missing_note = (
            f"{len(missing)} soil/terrain layer(s) have no value in this 1 km cell "
            f"({', '.join(missing[:6])}{'…' if len(missing) > 6 else ''}); "
            "XGBoost treats those as missing rather than as average soil."
        )

    sat = sample_satellite(lat, lon)
    if missing_note:
        sat.setdefault("notes", []).insert(0, missing_note)
    site = site_plain_language(names, values)
    cell = tag_cell_status(cell_geometry(lat, lon), sat)

    saved = load_saved_models(min_auc=min_auc, metrics_path=METRICS_PATH)
    if saved.empty:
        raise SystemExit(
            f"No saved USA 1 km models in {METRICS_PATH} passed the AUC gate. "
            "Train with: python xgboost_training_usa_30s.py --full-list --skip-existing"
        )

    traits_table = load_usa_traits(saved["species"].tolist())
    traits_table = traits_table.set_index("species")
    goal = (goal or "any").lower()
    sun = (sun or "any").lower()
    region = region_for_country(country) or "n_america"

    x = values.reshape(1, -1)
    values_2050, future_err = sample_future_usa(names, values, lat, lon)
    x2050 = None if values_2050 is None else values_2050.reshape(1, -1)
    scored = []
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
            "feature_importance": feature_importance_rows(model, names),
            "care": "" if str(traits.get("care") or "").lower() in {"", "nan", "none"} else traits.get("care"),
            "warning": "" if str(traits.get("warning") or "").lower() in {"", "nan", "none"} else traits.get("warning"),
            "native": native_label(traits, region),
            "goals": traits.get("goals") or "",
            "evergreen": traits.get("evergreen") or "",
        })

    scored.sort(key=lambda r: r["rank_score"], reverse=True)
    return {
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
    }


def print_report(result, address=None):
    print()
    print("=" * 64)
    print("USA 1 km tree suggestions")
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
        if val is not None and np.isfinite(val):
            mix_bits.append(f"{key} {val:.0%}")
    if mix_bits:
        print("Satellite mix: " + ", ".join(mix_bits))
    for note in result["satellite"].get("notes", []):
        print(f"Satellite: {note}")
    fut = result.get("future") or {}
    if fut.get("available"):
        print("2050: 1 km BIO delta (ssp245 2041–2060); soil and terrain unchanged")
    else:
        print("2050: skipped — " + (fut.get("note") or ""))
    if result["satellite"].get("blocked"):
        print()
        print("No planting list — satellite/terrain says this pin is not plantable.")
        return

    if not result["picks"]:
        print()
        print("No species cleared the match + goal filters. Try --goal any or --min-auc 0.60.")
        return

    print()
    print(f"Top {len(result['picks'])} of {result['n_models']} climate-vetted trees:")
    print()
    for i, pick in enumerate(result["picks"], 1):
        print(f"{i}. {pick['common_name']}  ({pick['species']})")
        print(
            f"   {pick['confidence']}   today {pick['p']:.0%}",
            end="",
        )
        if pick.get("p_2050") is not None:
            print(f"   2050 {pick['p_2050']:.0%}", end="")
        print(f"   model AUC {pick['auc']:.2f}")
        print(f"   {pick['reason']}")
        if pick.get("future_note"):
            print(f"   2050: {pick['future_note']}")
        fi = pick.get("feature_importance") or []
        if fi:
            bits = [f"{row['label']} {row['share']:.0%}" for row in fi[:5]]
            print("   Model uses: " + ", ".join(bits))
        print(f"   {pick['native']}")
        if pick["care"]:
            print(f"   Care: {pick['care']}")
        if pick["warning"]:
            print(f"   Warning: {pick['warning']}")
        print()


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Recommend trees from USA 1 km XGBoost models (not the 18 km POC)."
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
