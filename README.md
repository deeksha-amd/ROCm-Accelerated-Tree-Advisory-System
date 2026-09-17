# ROCm tree advisory (USA 1 km)

One XGBoost model per tree species: given a location’s climate, soil, and terrain, how much does it look like places where that species has been recorded? Latitude/longitude only look up the raster cell. They are not features.

The **main path is the contiguous USA at 1 km** (30 arcsec). An older global **18 km** demo lives in `poc/`.

Run every command from this directory (`hackathon_2026/`), with the venv on:

```bash
source venv/bin/activate
```

Training needs a GPU (`device=cuda`, ROCm on MI300X). Recommend and maps run on CPU.

---

## Layout

| Path | Role |
|---|---|
| `xgboost_training_usa_30s.py` | Train USA 1 km models → `data/models_usa_30s/` |
| `clean_species_usa_30s.py` | Clean + thin US GBIF onto the 1 km grid |
| `recommend_usa_30s.py` | Pin → top 5 USA trees (today + 2050) |
| `future_climate_usa_30s.py` | Build honest 1 km 2050 BIO (once) |
| `suitability_maps_usa_30s.py` | One-species today vs 2050 map |
| `poc/` | Global 18 km train / recommend / oak–Seville maps |
| `data/` | Rasters, GBIF, species lists, saved models |
| `data/scripts/` | Downloaders (climate, soil, topography, satellite, GBIF) |
| `maps/` | USA HTML (`austin.html`, live oak, …) |
| `repo_paths.py` | Shared `data/` locations (imported, not run) |

How models are trained and how recommend uses them: `DEVELOPER_GUIDE.md`.  
Where each raster came from and what not to train on: `DATA_OVERVIEW.md`.

---

## USA 1 km — recommend (models already on disk)

About 249 species are saved under `data/models_usa_30s/`. You do not need to retrain to demo a pin.

```bash
# Austin
python recommend_usa_30s.py --lat 30.2672 --lon -97.7431 --goal shade --html maps/austin.html

# Or geocode
python recommend_usa_30s.py --address "Austin, Texas" --goal shade --html maps/austin.html
```

Open `maps/austin.html`. The page shows lat/lon, a 1 km cell, today vs 2050 bars, and which climate/soil layers the models use most.

CONUS only. Pins outside the lower-48 envelope are rejected.

If 2050 bars are missing, build the 1 km future BIO once (does not retrain):

```bash
python future_climate_usa_30s.py
python recommend_usa_30s.py --lat 30.2672 --lon -97.7431 --html maps/austin.html
```

One-species map (default live oak, south-central US crop):

```bash
python suitability_maps_usa_30s.py --species "Quercus virginiana"
```

Open `maps/Quercus_virginiana_usa_30s.html`.

---

## USA 1 km — train

Occurrences must be cleaned onto the same 1 km grid the rasters use.

```bash
# Short list first (~32 well-known trees)
python clean_species_usa_30s.py
python xgboost_training_usa_30s.py

# Full US checklist (keeps JSON already on disk)
python clean_species_usa_30s.py --species-list data/usa_tree_species_list.csv
python xgboost_training_usa_30s.py --full-list --skip-existing
```

`--skip-existing` does not overwrite a saved booster. Drop it only if you intend to retrain.

Needs ~6 GB RAM for the 61-layer stack. Writes `data/models_usa_30s/*.json` and `data/models_usa_30s/metrics.csv`. Does not touch `poc/` or `data/models/`.

---

## 18 km global POC

Separate contract: 42 layers on a 2160×1080 WorldClim grid, models in `data/models/`.

```bash
python data/scripts/satellite_rasters.py          # site filters, not XGBoost features
python poc/recommend.py --lat 51.51 --lon -0.13 --goal shade --html poc/maps/suggest.html
python poc/suitability_maps.py --species "Quercus robur"
```

Retrain that fleet with `python poc/xgboost_training.py`. Do not point it at `data/country_data/USA`.

---

## Downloaders

Only if a raster folder is missing. Outputs land under `data/`.

```bash
python data/scripts/download_species.py
python data/scripts/current_climate_rasters.py
python data/scripts/future_climate_rasters.py
python data/scripts/soil_rasters.py
python data/scripts/topography_rasters.py
python data/scripts/satellite_rasters.py
```

USA 1 km climate/soil/terrain are already in `data/country_data/USA/`. Satellite vegetation is a **filter after scoring**, never an XGBoost input.
