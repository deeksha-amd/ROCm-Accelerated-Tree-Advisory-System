# ROCm tree advisory (USA 1 km)

One XGBoost model per tree species. A pin only looks up the raster cell;
latitude and longitude are **not** features. The score `p` is **record-likeness**:
how much that 1 km cell looks like GBIF records of the species *versus other
listed trees* (target-group background). It is not a planting permit, survival
odds, or a backyard shade model.

The **main path is the contiguous USA at 1 km** (30 arcsec). An older global
**18 km** demo lives in `poc/`. A separate **deep-learning SDM** pipeline lives
in `deep_learning_sdm/` (see `deep_learning_sdm/COMMANDS.md`).

Run every command from this directory (`hackathon_2026/`), with the venv on:

```bash
source venv/bin/activate
```

Train with `--device cuda` on MI300X (or `--device cpu`). A pin is one 61-vector —
`recommend_usa_30s.py` scores on CPU. Maps default to CPU; pass `--device cuda` to
batch-score a bbox.

---

## A git clone cannot run Austin

`metrics.csv` and `feature_names.txt` are in git. The rest of the USA runtime
is **gitignored** (GeoTIFFs are huge; JSON boosters are many):

| Needed locally | Typical path | If missing |
|---|---|---|
| 1 km climate / soil / terrain | `data/country_data/USA/{climate,soil,topography}_30s/` | copy from the training machine |
| Saved boosters | `data/models_usa_30s/*.json` | copy, or train (below) |
| Optional 2050 BIO | `data/country_data/USA/climate_future_2050_30s/` | `python future_climate_usa_30s.py` |
| Optional 1 km satellite filters | `data/country_data/USA/satellite_30s/` | `python data/scripts/satellite_rasters_usa_30s.py` |

`recommend_usa_30s.py` exits with that list rather than scoring on empty folders.
Do **not** point USA recommend at `data/satellite/` (the 18 km global filters).

---

## Layout

| Path | Role |
|---|---|
| `xgboost_training_usa_30s.py` | Train USA 1 km models → `data/models_usa_30s/` |
| `clean_species_usa_30s.py` | Clean + thin US GBIF onto the 1 km grid |
| `recommend_usa_30s.py` | Pin → top 5 plantable trees (today + 2050) |
| `future_climate_usa_30s.py` | Build honest 1 km 2050 BIO (once) |
| `suitability_maps_usa_30s.py` | One-species today vs 2050 record-likeness map |
| `poc/` | Global 18 km train / recommend / oak–Seville maps |
| `data/` | Rasters, GBIF, species lists, saved models |
| `data/scripts/` | Downloaders (climate, soil, topography, satellite, GBIF) |
| `deep_learning_sdm/` | Deep-learning SDM + advisory (France 1 km served run) |
| `maps/` | USA HTML (`austin.html`, live oak, …) |
| `repo_paths.py` | Shared `data/` locations (imported, not run) |

How models are trained and how recommend uses them: `DEVELOPER_GUIDE.md`.  
Where each raster came from and what not to train on: `DATA_OVERVIEW.md`.

---

## USA 1 km — recommend (needs local models + rasters)

When the table above is on disk:

```bash
# Austin
python recommend_usa_30s.py --lat 30.2672 --lon -97.7431 --goal shade --html maps/austin.html

# Or geocode
python recommend_usa_30s.py --address "Austin, Texas" --goal shade --html maps/austin.html
```

Each booster is loaded with `load_booster(..., device="cpu")`. For that pin the
61-vector is scored as `p = predict_proba(x)[0, 1]` (sigmoid of the tree-sum
log-odds). Species with `p < 0.30` are dropped; the rest rank by `p * auc`.
Invasive and naturalised trees (chinaberry, tree-of-heaven, …) are trained so
the model knows them, then **dropped from the top-5** and listed under “do not
plant”. Default recommend gate is AUC **≥ 0.70**.

Open `maps/austin.html`. The page shows lat/lon, a 1 km cell, today vs 2050
record-likeness bars, and **species-wide** XGBoost gain (not a local explanation
of the pin). The demo reads the **1:1** JSON in `data/models_usa_30s/` (255
saved, mean AUC 0.888) — not a `--shared-universe` speed-run.

CONUS only. Pins outside the lower-48 envelope are rejected.

If 2050 bars are missing, build the 1 km future BIO once (does not retrain):

```bash
python future_climate_usa_30s.py
python recommend_usa_30s.py --lat 30.2672 --lon -97.7431 --html maps/austin.html
```

1 km satellite mix (WorldCover on the same cell as the map box):

```bash
python data/scripts/satellite_rasters_usa_30s.py --smoke   # Austin + Portland tiles
# python data/scripts/satellite_rasters_usa_30s.py         # full CONUS, slower
```

One-species map (default live oak, south-central US crop — a bbox, not the
full 7020×3060 stack):

```bash
python suitability_maps_usa_30s.py --species "Quercus virginiana"
# optional GPU batches (host NumPy → inplace_predict; CPU is often faster for one booster)
# python suitability_maps_usa_30s.py --species "Quercus virginiana" --device cuda --batch-rows 1000000
```

Open `maps/Quercus_virginiana_usa_30s.html`.

---

## USA 1 km — train

Occurrences must be cleaned onto the same 1 km grid the rasters use.

There are **two protocols**. Austin scores the **1:1** fleet in
`data/models_usa_30s/`. Do **not** overwrite that folder with
`--shared-universe` JSON.

### Demo fleet (1:1 absences)

Each species gets its presence cells plus `min(n_presence, 25000)` other
listed-tree cells. Typical X is a few thousand rows. Shipped: **255 saved**,
mean spatial-block AUC **0.888**. `--device` defaults to `cuda`.

```bash
# Short list first (~32 well-known trees)
python clean_species_usa_30s.py
python xgboost_training_usa_30s.py

# Full US checklist (keeps JSON already on disk)
python clean_species_usa_30s.py --species-list data/usa_tree_species_list.csv
python xgboost_training_usa_30s.py --full-list --skip-existing
```

`--skip-existing` keeps a booster only when the JSON exists **and**
`metrics.csv` says `status=saved`. Drop the flag only if you intend to retrain.

### Speed-run (`--shared-universe`)

One X of unique 1 km tree cells (**245,276** after dropping NaN rows). Each
species is a 0/1 label on that table, with `scale_pos_weight`. X is
histogram-binned once (`QuantileDMatrix`); folds reuse those cuts, on CPU and
GPU alike. GPU histogram work is real; 1:1 samples are launch-bound. Clocked on
MI300X vs 8 CPU threads: fit **458 s vs 2414 s (5.3×)**, wall 490 s vs 2446 s,
mean spatial-block AUC **0.876** GPU / **0.877** CPU (**248 saved**). That AUC
is **not** comparable to the 1:1 fleet. Write to a **different** `--model-dir`.

```bash
python xgboost_training_usa_30s.py --full-list --shared-universe --device cuda \
  --model-dir data/models_usa_30s_shared
python xgboost_training_usa_30s.py --full-list --shared-universe --device cpu --n-jobs 8 \
  --model-dir data/models_usa_30s_shared_cpu
```

`--n-jobs 8` is the workstation comparison; `0` (default) uses all visible CPUs.
The trainer prints Wall / peak RSS.

Needs ~6 GB RAM for the 61-layer stack. Does not touch `poc/` or `data/models/`.

---

## 18 km global POC

Separate contract: 42 layers on a 2160×1080 WorldClim grid, models in `data/models/`.

```bash
python data/scripts/satellite_rasters.py          # 18 km site filters, not XGBoost features
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
python data/scripts/satellite_rasters.py            # 18 km POC filters
python data/scripts/satellite_rasters_usa_30s.py    # USA 1 km filters
```

USA 1 km climate/soil/terrain live in `data/country_data/USA/` (gitignored
GeoTIFFs). Satellite vegetation is a **filter after scoring**, never an XGBoost
input. The deep-learning pipeline uses a separate `country_data/` tree at the
repo root.
