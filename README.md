# ROCm tree advisory (USA 1 km)

Two SDMs over the same 1 km data, picked with `--model`:

| `--model` | What it is | Weights |
|---|---|---|
| `xgboost` (default) | one gradient-boosted booster per species | `data/models_usa_30s/*.json` |
| `deepmaxent` | one [DeepMaxent](https://github.com/RYCKEWAERT/deepmaxent) network with an output per species | `data/models_deepmaxent_usa_30s/*.pt` |

A pin only looks up the raster cell; latitude and longitude are **not**
features. The score `p` is **record-likeness**: how much that 1 km cell looks
like GBIF records of the species *versus other listed trees* (target-group
background). It is not a planting permit, survival odds, or a backyard shade
model. Both models are trained on the same occurrences and the same 61 layers,
so `p` reads on the same scale either way.

The **main path is the contiguous USA at 1 km** (30 arcsec). An older global
**18 km** demo lives in `poc/`. A separate **deep-learning SDM** pipeline lives
in `deep_learning_sdm/` (see `deep_learning_sdm/COMMANDS.md`).

Run every command from this directory (`hackathon_2026/`), with the venv on:

```bash
source venv/bin/activate
```

Training needs a GPU (`device=cuda`, ROCm on MI300X). Recommend and maps run on CPU.

---

## Which model to pick

Both gate on 5-fold spatial-block CV. Mean AUC over the species each one gates:

| Model | Species gated | Mean AUC |
|---|---|---|
| XGBoost | 32 | 0.8451 |
| DeepMaxent | 243 | 0.8534 |

Those two numbers are **not** comparable: DeepMaxent's 0.8534 spans 243 species,
many of them sparse and hard, against 32 well-sampled ones for XGBoost. On the 32
both gate, XGBoost is still ahead — **0.8451 vs 0.8235**. DeepMaxent does not win
on accuracy. The protocols also differ in difficulty, unresolved so far: XGBoost
blocks each species' own points, so held-out ground still lies inside that
species' range, while DeepMaxent blocks all 245k CONUS cells into
continental-scale chunks (per-fold 0.887 / 0.888 / 0.818 / 0.867 / 0.804). Part
of the gap is likely protocol rather than model quality, but that is unproven.

Coverage is the sharper difference. DeepMaxent gates 243 of the 285 names it
trains on against XGBoost's 32, and the default `--min-auc 0.70` leaves 236
eligible at a pin (XGBoost: 31), so shortlists come from a much wider pool.
Retraining cuts the other way: `--skip-existing` makes XGBoost incremental, so
adding 20 species costs 20 species of training, while DeepMaxent fits every
species jointly and adding one means refitting all of them (~5-6 min). XGBoost
ships 255 JSON files, DeepMaxent one 0.8 MB checkpoint.

Scoring is CPU-only by design for both. A pin costs ~2.6 s with `--model
deepmaxent` against ~1.4 s with `--model xgboost`, nearly all of it importing
torch rather than the model. The GPU is skipped deliberately: it needs 2.25 s of
warm-up, more than the 0.74 s CPU forward pass over all 245k cells.

---

## A git clone cannot run Austin

`metrics.csv` and `feature_names.txt` are in git. The rest of the USA runtime
is **gitignored** (GeoTIFFs are huge; JSON boosters are many):

| Needed locally | Typical path | If missing |
|---|---|---|
| 1 km climate / soil / terrain | `data/country_data/USA/{climate,soil,topography}_30s/` | copy from the training machine |
| Saved boosters | `data/models_usa_30s/*.json` | copy, or train (below) |
| DeepMaxent checkpoint | `data/models_deepmaxent_usa_30s/deepmaxent_usa_30s.pt` | copy, or train (below); only for `--model deepmaxent` |
| Optional 2050 BIO | `data/country_data/USA/climate_future_2050_30s/` | `python future_climate_usa_30s.py` |
| Optional 1 km satellite filters | `data/country_data/USA/satellite_30s/` | `python data/scripts/satellite_rasters_usa_30s.py` |

`recommend_usa_30s.py` exits with that list rather than scoring on empty folders.
Do **not** point USA recommend at `data/satellite/` (the 18 km global filters).

---

## Layout

| Path | Role |
|---|---|
| `xgboost_training_usa_30s.py` | Train USA 1 km boosters → `data/models_usa_30s/` |
| `deepmaxent_training_usa_30s.py` | Train the USA 1 km Deep SDM → `data/models_deepmaxent_usa_30s/` |
| `deepmaxent/` | DeepMaxent network + losses, vendored verbatim from upstream |
| `deepmaxent_sdm.py` | Checkpoint format and inference for the Deep SDM |
| `clean_species_usa_30s.py` | Clean + thin US GBIF onto the 1 km grid |
| `recommend_usa_30s.py` | Pin → top 5 plantable trees (today + 2050), either model |
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

# Same pin, Deep SDM instead of the boosters
python recommend_usa_30s.py --model deepmaxent --lat 30.2672 --lon -97.7431 --goal shade
```

Open `maps/austin.html`. The page shows lat/lon, a 1 km cell, today vs 2050
record-likeness bars, and a per-layer breakdown that depends on the model:
`--model xgboost` reports **species-wide** gain (not a local explanation of the
pin), while `--model deepmaxent` reports a **local sensitivity at that pin** —
`|d lambda / d z|`, the score change per 1 SD of each layer. Invasive and
naturalised trees (chinaberry, tree-of-heaven, …) are trained so the model
knows them, then **dropped from the top-5** and listed under “do not plant”.

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

## USA 1 km — train the Deep SDM

Same cleaned occurrences, same 61 layers, same target-group background, same
spatial-block CV gate. DeepMaxent fits every species jointly in one network, so
this is a single run rather than one per species.

```bash
python deepmaxent_training_usa_30s.py --smoke        # wiring check, ~1 min
python deepmaxent_training_usa_30s.py --full-list    # 5 CV folds + final fit, ~5-6 min
```

Needs ~6 GB RAM for the same 61-layer stack as the boosters. Of the ~5-6 min on
one MI300X, the final fit is 68 s; the rest is the five CV folds.

Writes `data/models_deepmaxent_usa_30s/deepmaxent_usa_30s.pt`, `metrics.csv`
and `feature_names.txt`; leaves the boosters alone. The defaults reproduce the
shipped checkpoint. Architecture and epochs are upstream's, but the optimiser
settings are retuned: `--weight-decay 2e-5` because upstream's `3e-4` was tuned
on a far smaller benchmark and over-regularises this dataset at a cost of ~4.5
AUC points, and `--learning-rate 1e-3 --batch-size 4096` purely for speed, at
the same accuracy. Override with `--epochs`, `--batch-size`,
`--learning-rate`, `--hidden-size`, `--hidden-nbr`, `--weight-decay`, `--loss`.

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
