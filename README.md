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
model. Both models use the same occurrences and the same 61 layers, so `p`
reads on the same scale either way.

The **main path is the contiguous USA at 1 km** (30 arcsec). An older global
**18 km** demo lives in `poc/`.

Train with `--device cuda` on MI300X (or `--device cpu`). A pin is one
61-vector — `recommend_usa_30s.py` scores on CPU. Maps default to CPU; pass
`--device cuda` to batch-score a bbox.

How models are trained and how recommend uses them: `DEVELOPER_GUIDE.md`.  
Where each raster came from and what not to train on: `DATA_OVERVIEW.md`.

---

## Setup

Run every command from this repository root, with the venv on:

```bash
python3 -m venv venv
source venv/bin/activate
pip install scikit-learn rasterio pandas numpy requests tqdm
pip install amd_xgboost --extra-index-url=https://pypi.amd.com/rocm-7.1.1/simple
```

`--model deepmaxent` needs the ROCm PyTorch image below (do not `pip install`
a CUDA wheel from PyPI).

---

## Which model to pick

Both gate on 5-fold spatial-block CV. Mean AUC over the species each one
**saves**:

| Model | Species saved | Mean AUC | AUC ≥ 0.70 at a pin |
|---|---|---|---|
| XGBoost (1:1 fleet) | 255 | 0.888 | 253 |
| DeepMaxent | 243 | 0.8529 | 236 |

Those AUCs are **not** comparable. Protocols differ: XGBoost blocks each
species' own points, so held-out ground still lies inside that species' range,
while DeepMaxent blocks all ~245k CONUS tree cells into continental-scale
chunks. Retraining also differs: `--skip-existing` makes XGBoost incremental;
DeepMaxent fits every species jointly, so adding one name means refitting all
of them (~2 min). XGBoost ships one JSON per saved species; DeepMaxent ships
one ~0.8 MB checkpoint.

Scoring is CPU-only by design for both. A pin costs ~2.6 s with `--model
deepmaxent` against ~1.4 s with `--model xgboost`, nearly all of it importing
torch rather than the model. The GPU is skipped deliberately: it needs 2.25 s
of warm-up, more than the 0.74 s CPU forward pass over all 245k cells.

---

## A git clone cannot run Austin

`metrics.csv` and `feature_names.txt` are in git. The rest of the USA runtime
is **gitignored** (GeoTIFFs are huge; JSON boosters are many):

| Needed locally | Typical path | If missing |
|---|---|---|
| 1 km climate / soil / terrain | `data/country_data/USA/{climate,soil,topography}_30s/` | copy from the training machine |
| Saved boosters | `data/models_usa_30s/*.json` | copy, or train (XGBoost below) |
| DeepMaxent checkpoint | `data/models_deepmaxent_usa_30s/deepmaxent_usa_30s.pt` | copy, or train (DeepMaxent below) |
| Optional 2050 BIO | `data/country_data/USA/climate_future_2050_30s/` | `python future_climate_usa_30s.py` |
| Optional 1 km satellite filters | `data/country_data/USA/satellite_30s/` | `python data/scripts/satellite_rasters_usa_30s.py` |

`recommend_usa_30s.py` exits with that list rather than scoring on empty
folders. Do **not** point USA recommend at `data/satellite/` (the 18 km global
filters).

Optional 2050 bars (does not retrain):

```bash
python future_climate_usa_30s.py
```

Optional 1 km satellite mix (WorldCover on the same cell as the map box):

```bash
python data/scripts/satellite_rasters_usa_30s.py --smoke   # Austin + Portland tiles
# python data/scripts/satellite_rasters_usa_30s.py         # full CONUS, slower
```

---

## Layout

| Path | Role |
|---|---|
| `xgboost_training_usa_30s.py` | Train USA 1 km boosters → `data/models_usa_30s/` |
| `deepmaxent_training_usa_30s.py` | Train the USA 1 km Deep SDM → `data/models_deepmaxent_usa_30s/` |
| `clean_species_usa_30s.py` | Clean + thin US GBIF onto the 1 km grid |
| `recommend_usa_30s.py` | Pin → top 5 plantable trees (today + 2050), either model |
| `suitability_maps_usa_30s.py` | One-species today vs 2050 map (saved JSON boosters) |
| `future_climate_usa_30s.py` | Build honest 1 km 2050 BIO (once) |
| `poc/` | Global 18 km train / recommend / maps |
| `poc/archive/` | Unused France deep SDM snapshot (not DeepMaxent) |
| `data/` | Rasters, GBIF, species lists, saved models |
| `data/scripts/` | Downloaders (climate, soil, topography, satellite, GBIF) |
| `maps/` | USA HTML (`austin.html`, live oak, …) |
| `repo_paths.py` | Shared `data/` locations (imported, not run) |

Occurrences must be cleaned onto the same 1 km grid the rasters use before
either trainer:

```bash
python clean_species_usa_30s.py                                          # seed CSV
python clean_species_usa_30s.py --species-list data/usa_tree_species_list.csv  # full US checklist
```

Needs ~6 GB RAM for the 61-layer stack. Does not touch `poc/` or `data/models/`.

---

## XGBoost

Austin scores the **1:1** fleet in `data/models_usa_30s/`. Do **not** overwrite
that folder with `--shared-universe` JSON.

### Train (1:1 absences)

Each species gets its presence cells plus `min(n_presence, 25000)` other
listed-tree cells. Typical X is a few thousand rows. Shipped: **255 saved**,
mean spatial-block AUC **0.888**. `--device` defaults to `cuda`.

```bash
python xgboost_training_usa_30s.py
python xgboost_training_usa_30s.py --full-list --skip-existing
```

`--skip-existing` keeps a booster only when the JSON exists **and**
`metrics.csv` says `status=saved`. Drop the flag only if you intend to retrain.

### Train (`--shared-universe`)

One X of unique 1 km tree cells (**245,276** after dropping NaN rows). Each
species is a 0/1 label on that table, with `scale_pos_weight`. X is
histogram-binned once (`QuantileDMatrix`); folds reuse those cuts, on CPU and
GPU alike. Clocked on MI300X vs 8 CPU threads: fit **458 s vs 2414 s (5.3×)**,
wall 490 s vs 2446 s, mean spatial-block AUC **0.876** GPU / **0.877** CPU
(**248 saved**). That AUC is **not** comparable to the 1:1 fleet. Write to a
**different** `--model-dir`.

```bash
python xgboost_training_usa_30s.py --full-list --shared-universe --device cuda \
  --model-dir data/models_usa_30s_shared
python xgboost_training_usa_30s.py --full-list --shared-universe --device cpu --n-jobs 8 \
  --model-dir data/models_usa_30s_shared_cpu
```

`--n-jobs 8` is the workstation comparison; `0` (default) uses all visible
CPUs. The trainer prints Wall / peak RSS.

### Recommend

```bash
python recommend_usa_30s.py --lat 30.2672 --lon -97.7431 --goal shade --html maps/austin.html
python recommend_usa_30s.py --address "Austin, Texas" --goal shade --html maps/austin.html
```

Each booster is loaded with `load_booster(..., device="cpu")`. For that pin the
61-vector is scored as `p = predict_proba(x)[0, 1]`. Species with `p < 0.30`
are dropped; the rest rank by `p * auc`. Invasive and naturalised trees are
trained so the model knows them, then **dropped from the top-5** and listed
under “do not plant”. Default recommend gate is AUC **≥ 0.70**.

Open `maps/austin.html`. The page shows lat/lon, a 1 km cell, today vs 2050
bars, and **species-wide** gain (not a local explanation of the pin). This
path reads the **1:1** JSON in `data/models_usa_30s/` (255 saved) — not a
`--shared-universe` speed-run.

CONUS only. Pins outside the lower-48 envelope are rejected.

### Suitability map

Default live oak, south-central US crop — a bbox, not the full 7020×3060
stack:

```bash
python suitability_maps_usa_30s.py --species "Quercus virginiana"
# python suitability_maps_usa_30s.py --species "Quercus virginiana" --device cuda --batch-rows 1000000
```

Open `maps/Quercus_virginiana_usa_30s.html`.

---

## DeepMaxent

Same cleaned occurrences, same 61 layers, same target-group background, same
spatial-block CV gate. One network with an output per species — a single run
rather than one booster per name.

Train and score DeepMaxent inside this ROCm PyTorch container (GPU devices
`/dev/kfd` and `/dev/dri`, home mounted so the repo is visible):

```bash
docker run --cap-add=SYS_PTRACE --ipc=host --privileged=true --shm-size=128GB \
  --network=host --device=/dev/kfd --device=/dev/dri --group-add video -it \
  -v $HOME:$HOME --name ${LOGNAME}_rocm \
  rocm/pytorch:rocm7.1_ubuntu24.04_py3.12_pytorch_release_2.9.1
```

Do not replace the image's torch with a CUDA wheel from PyPI.

### Train

```bash
python deepmaxent_training_usa_30s.py --smoke        # wiring check, ~1 min
python deepmaxent_training_usa_30s.py --full-list    # 5 CV folds + final fit, ~2 min
```

Of the ~2 min on one MI300X, the final fit is 18 s; the rest is the five CV
folds. Writes `data/models_deepmaxent_usa_30s/deepmaxent_usa_30s.pt`,
`metrics.csv` and `feature_names.txt`; leaves the boosters alone.

Architecture and epochs are upstream's; optimiser settings are retuned:
`--weight-decay 2e-5` (upstream `3e-4` over-regularises this dataset),
`--learning-rate 1e-3 --batch-size 4096` for speed at the same accuracy.
Override with `--epochs`, `--batch-size`, `--learning-rate`, `--hidden-size`,
`--hidden-nbr`, `--weight-decay`, `--loss`.

### Recommend

```bash
python recommend_usa_30s.py --model deepmaxent --lat 30.2672 --lon -97.7431 --goal shade --html maps/austin.html
```

Same pin contract as XGBoost (`p`, AUC gate, invasive drop). The per-layer
panel is a **local sensitivity at that pin** — `|d lambda / d z|`, the score
change per 1 SD of each layer.

### Suitability map

`suitability_maps_usa_30s.py` scores saved JSON boosters, not the DeepMaxent
checkpoint. Use the XGBoost map command above, or score a pin with
`--model deepmaxent`.

---

## 18 km global POC

Separate contract: 42 layers on a 2160×1080 WorldClim grid, models in
`data/models/`. Do not point it at `data/country_data/USA`.

```bash
python data/scripts/satellite_rasters.py
python poc/recommend.py --lat 51.51 --lon -0.13 --goal shade --html poc/maps/suggest.html
python poc/suitability_maps.py --species "Quercus robur"
python poc/xgboost_training.py
```

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
GeoTIFFs). Satellite vegetation is a **filter after scoring**, never an
XGBoost input.
