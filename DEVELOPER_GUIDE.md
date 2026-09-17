# Developer guide: tree SDM training → recommendations

This is the engineering map of the hackathon POC: how models are trained, what files they produce, how a non-technical recommendation is built, and how to plug in **satellite filters**, **2050 climate**, and **maps**.

If you only need data provenance (sources, licences, why GEDTM30), read `DATA_OVERVIEW.md`. This document is the **code path**.

---

## 1. What the system is

Each saved file in `models/` is a **binary XGBoost classifier for one tree species**.

It does **not** answer “which of 171 species lives here?” It answers, independently:

> Given this cell’s climate, soil, and elevation, how much does it look like places where *this* species has been recorded?

A location is never a feature. Latitude/longitude are only used to **look up** a 42-number vector on a shared 10-arc-minute grid (~**18.5 km** cells, 2160 × 1080 globally). Two yards in the same cell get the same score.

```
GBIF points + rasters          XGBoost (one model / species)         App
─────────────────────          ────────────────────────────          ───
occurrences (y=1)              spatial-block CV + early stop         recommend.py
pseudo-absences (y=0)    →     final model on all rows         →     top 5 cards
42 env numbers at each cell    models/*.json + metrics.csv           (JSON or text)
```

---

## 2. Repository map

| Path | What a developer uses it for |
|---|---|
| `download_species.py` | Pull GBIF occurrences → `gbif_500_species.csv` |
| `current_climate_rasters.py` | WorldClim BIO 1970–2000 → `climate_current/` |
| `recent_climate_rasters.py` | Derived 2015–2024 BIO (better match to GBIF dates; optional retrain) |
| `future_climate_rasters.py` | CMIP6 ssp245 2041–2060, **one 19-band GeoTIFF** |
| `soil_rasters.py` / `topography_rasters.py` | Align soil + elevation to the same grid |
| `satellite_rasters.py` | Stream ESA WorldCover → `satellite/*.tif` filters (**not** XGBoost features) |
| `xgboost_training.py` | Train, gate, save models |
| `catalog.py` | Genus/species cards (common name, goals, care) |
| `data/species_traits.csv` | Editable overlay on those cards |
| `recommend.py` | Location → top 5 plain-language picks (`--html` writes a Leaflet page) |
| `recommend_html.py` | HTML template for the pin + 18 km cell + today/2050 bars |
| `suitability_maps.py` | One-species today vs 2050 GeoTIFF/PNG/HTML (default: English oak) |
| `models/` | `*.json` boosters, `metrics.csv`, `feature_names.txt` |
| `DATA_OVERVIEW.md` | Why each dataset, what not to train on |

**Run training and recommend from the project root** (`hackathon_2026/`). Paths are relative. GPU is required for training (`device="cuda"` / ROCm on MI300X); recommend runs on CPU.

```bash
source venv/bin/activate
python xgboost_training.py          # hours-scale over ~300 taxa; writes models/
python recommend.py --lat 51.51 --lon -0.13 --goal shade --html suggest.html
```

---

## 3. Training flow (`xgboost_training.py`)

### 3.1 Load predictors once

`collect_predictor_paths()` stacks, in this order (must stay stable):

1. All `climate_current/*.tif` (19 BIO layers, **sorted by filename** — so `bio_10` comes before `bio_2`)
2. `soil_data/aligned_10m/*.tif` except `source_flag_10m.tif` (22 layers)
3. `topography/gedtm30/gedtm30_v1.2_elev_10m.tif`

That list is written to `models/feature_names.txt`. Inference **must** sample layers in the same order.

Do **not** train on `soil_data/soilgrids_5km/` (wrong grid) or vegetation satellite layers (circular: they measure existing trees).

### 3.2 Per species

For each unique name in `gbif_500_species.csv`:

1. **Keep trees only.** Binomial whose genus is in `TREE_GENERA`. Drops moths, birds, bacteria that matched a GBIF genus search (`Carduelis pinus`, `Erigeron acer`, …).
2. **Snap to unique grid cells.** Many GBIF rows share one 18 km pixel. Count unique cells, not records.
3. **Skip sparse ranges.** `< MIN_UNIQUE_CELLS` (80) → `skip_few_cells`. Spatial CV is noise on a handful of cells.
4. **Pseudo-absences.** Random land pixels inside a padded bounding box of the occurrences, only where climate+soil+elev are all valid. Equal count to presences (or fewer if the box is small). Global background is avoided on purpose (it made AUC `nan` or ~0.99).
5. **Labels.** Presence `y=1`, background `y=0`. Features = 42 raster values at those coordinates. Rows with any NaN dropped.
6. **Spatial CV.** K-means on map coordinates → ~10 geographic blocks. `StratifiedGroupKFold` holds whole blocks out (5 folds). A fold is used only if train and test both contain 0s and 1s. ROC AUC needs both classes.
7. **Fit.** `XGBClassifier`, histogram trees, GPU, early stopping on the held-out block. Regularization is lighter when `n` is small (`min_child_weight` 2 / 4 / 8).
8. **Final model.** Retrain on all rows with `n_estimators = median(best_iteration)` from CV (floor 10). That is the file we save.
9. **Save gate.** Write `models/{Genus_species}.json` only if AUC is finite **and** ≥ 3 mixed folds. Weak 2-fold AUCs (including lucky 0.96s) stay in `metrics.csv` as `skip_few_folds`.

### 3.3 What “AUC” means here

It is **spatial-block ROC AUC**, not random-split accuracy. Typical usable range is about **0.70–0.90**. Very high scores on well-sampled, widely planted species (`Pinus radiata`) can mean “easy climate envelope / plantations,” not a native woodland.

`recommend.py` ignores saved models with AUC `< 0.70` (`catalog.MIN_AUC_DEFAULT`).

### 3.4 Training outputs

| File | Contents |
|---|---|
| `models/Quercus_robur.json` | XGBoost booster (sklearn `load_model`) |
| `models/metrics.csv` | One row per GBIF name: `status`, `auc`, `n_unique_cells`, `n_folds_usable`, `model_path` |
| `models/feature_names.txt` | 42 raster filenames in training order |

`metrics.csv` `status` values:

| status | Meaning |
|---|---|
| `saved` | Model written; safe to score if AUC ≥ your gate |
| `skip_not_tree` | Name is not a tree genus |
| `skip_few_cells` | &lt; 80 unique cells |
| `skip_no_folds` | Every CV fold was one-class |
| `skip_few_folds` | Trained but &lt; 3 mixed folds; **not saved** |
| `skip_no_background` / `skip_train_error` | Data or fit failure |

### 3.5 Adding more GBIF data later

1. Replace or grow `gbif_500_species.csv` (same columns: `species`, `latitude`, `longitude`).
2. Re-run `python xgboost_training.py`.
3. New `models/*.json` appear; `recommend.py` loads whatever `metrics.csv` marks `saved`.
4. Optionally add a row to `data/species_traits.csv` for common name / shade-food-beauty. Missing rows fall back to genus defaults in `catalog.py`.

You do **not** need to change `recommend.py` for new species.

---

## 4. Recommendation flow (`recommend.py`)

```
address or lat/lon
    → geocode (Nominatim) + country code
    → read 42 pixels (windowed GeoTIFF reads, not full stacks)
    → optional satellite site filter
    → describe site in plain language (cool/warm, wet/dry, pH, drainage)
    → for each saved model with AUC ≥ 0.70:
          p = P(presence class | x)
    → filter by --goal (shade|food|beauty) and --sun using the traits table
    → drop p < 0.30
    → rank by p × AUC
    → top 5 cards: Great / Good / Risky, reason, native hint, care, warning
```

Confidence (site fit × model quality):

| Label | Rule |
|---|---|
| Great match | `p ≥ 0.70` and AUC ≥ 0.85 and ≥ 4 folds |
| Good match | `p ≥ 0.50` and AUC ≥ 0.75 |
| Risky | otherwise (still shown only if `p ≥ 0.30`) |

`--json` prints the same structure for a UI.

`data/species_traits.csv` is **not** from GBIF. It is generated from `catalog.py` (`python recommend.py --write-traits-stub`): genus defaults plus a hand list of well-known common names. Obscure taxa get names like “Griffithii oak”. Edit the CSV; `recommend.py` prefers those cells over code defaults.

---

## 5. Satellite data — use as a **site filter**, not a training feature

`DATA_OVERVIEW.md` measured this: NDVI / tree-cover alone can look like a strong predictor (AUC ~0.79) and then **collapse on farmland** (the land we advise on). Those layers answer “are there trees here already?”, not “what can establish?”

### Do not put in `x` (the 42 features)

- Tree / broadleaf / needleleaf / shrub / grass fractions
- All NDVI layers

### Download (aligned to the shared grid)

Do **not** download the 10 m WorldCover mosaic (~120 GB). `satellite_rasters.py` streams COG overviews over HTTP and averages class fractions onto `climate_current/wc2.1_10m_bio_1.tif` (2160×1080), the same pattern as `soil_rasters.py`.

```bash
python satellite_rasters.py --smoke     # 4 tiles: London, Portland, Lyon, Pacific
python satellite_rasters.py             # global, ~10 min with 8 workers
python satellite_rasters.py --bbox -10,40,10,60 --workers 8
```

Writes `satellite/*.tif` (gitignored). No XGBoost retrain. `recommend.py` reads those files on the next run.

`--smoke` / `--bbox` leave the rest of the globe as NaN so other cities are skipped, not marked as ocean. A full run treats unmapped cells as water.

### Do apply **after** `predict_proba`

`recommend.py` looks for `satellite/` (same 2160×1080 grid). If the folder is missing, it prints a skip note and continues. Exact names from `satellite_rasters.py` are preferred; substring match is the fallback:

| File / filename contains | Behaviour today |
|---|---|
| `planting_exclusion_mask_10m.tif` | Block recommendations if cell &gt; 0.5 |
| `water_fraction_10m.tif` | Block if fraction ≥ 0.5 |
| `snow_ice_fraction_10m.tif` | Block if fraction ≥ 0.5 |
| `built_up_fraction_10m.tif` | Warn (city cell, 18 km average) |
| `cropland_fraction_10m.tif` | Context: open farmland, not forest |
| `tree_cover_fraction_10m.tif` | Warn if already wooded (≥ 0.4) — UI only |
| `ndvi` | Warn if very green (≥ 0.6) — UI only; not produced by default |

### Optional extra predictor (retrain required)

`soilmoisture_mean` / `soilmoisture_amplitude` are microwave soil moisture, not greenness. They cover only ~79% of GBIF points (holes under dense canopy). To use them:

1. Drop the rasters into the aligned 10 m grid.
2. Append them in `collect_predictor_paths()` **and retrain every species** (feature count must match `feature_names.txt`).
3. Do not mix old 42-feature JSON files with a 44-feature sampler.

Until you retrain, keep moisture as a **text hint** on the card (“soil stays wet”), not as an XGBoost input.

---

## 6. Future climate (2050) — same trees, different BIO vector

File on disk:

`climate_future_2050/wc2.1_10m_bioc_MPI-ESM1-2-HR_ssp245_2041-2060.tif`

One GeoTIFF, **19 bands** = BIO1…BIO19 (band 1 = BIO1). Soil and elevation stay as they are (static-soil assumption).

### Pairing rule (easy to get wrong)

This 2050 file was bias-corrected against **WorldClim 1970–2000** (`climate_current/`). Compare future scores to models trained on `climate_current/`, which is what we have now.

If you later retrain on `climate_recent/` (2015–2024), **do not** subtract that from this 2050 product — you would understate warming. You would need a future layer bias-corrected to the same recent baseline.

### Feature-order rule (also easy to get wrong)

Training order is **filename sort**, not BIO1–BIO19:

`bio_1, bio_10, bio_11, …, bio_19, bio_2, …, bio_9`

Future band `k` is BIO`k`. When building `x_2050`, copy soil+elev from today and insert BIO values in **`feature_names.txt` order**, not band order 1..19.

Sketch:

```python
import rasterio
import numpy as np

names = [ln.strip() for ln in open("models/feature_names.txt")]
# x_now: 42 values in that order (recommend.sample_predictors)
with rasterio.open("climate_future_2050/wc2.1_10m_bioc_MPI-ESM1-2-HR_ssp245_2041-2060.tif") as src:
    # src.read(k) is BIO k at 1-based band index
    bio = {k: src.read(k)[row, col] for k in range(1, 20)}

x_future = []
for name in names:
    if "bio_" in name:
        # "...bio_12.tif" -> 12
        k = int(name.split("bio_")[1].split(".")[0])
        x_future.append(bio[k])
    else:
        x_future.append(x_now[names.index(name)])  # soil / elev unchanged
```

Then `p_now = model.predict_proba(x_now)` and `p_2050 = model.predict_proba(x_future)`.

`recommend.py` does this at the pin automatically when `climate_future_2050/wc2.1_10m_bioc_MPI-ESM1-2-HR_ssp245_2041-2060.tif` is present (`--html` shows paired bars). `suitability_maps.py` does it for every land cell of one species.

### Product ideas

| UX | How |
|---|---|
| “Still a match in 2050?” | Same top 5, show `p_now` vs `p_2050` |
| “Climate winners / losers” | Rank by `p_2050 - p_now` (range shift at this cell) |
| Suitability map 2050 | Same as §7 maps, but BIO from the 19-band file |

Do **not** retrain a second XGBoost on 2050 rasters. There are no 2050 occurrence labels. Transfer the **current** niche model to a new climate.

---

## 7. Showing output visually

The CLI is the POC. Everything below consumes `recommend.py --json` or the same `predict_proba` call.

### 7.1 Pin + cards (product UI)

```bash
python recommend.py --lat 51.51 --lon -0.13 --goal shade --html suggest.html
```

`suggest.html` is a Leaflet page: map, the **18 km cell** as a rectangle (green plantable / amber city or farm / red blocked), satellite mix bar, top-5 cards with **today vs 2050** bars. Needs the network for Leaflet and Esri tiles (OSM/CARTO block or watermark `file://` pages).

Zoom is capped (`maxZoom` 8 on fit). The caption states the score is for the cell, not a backyard.

### 7.2 Top-5 bar chart

Those bars live on the HTML cards (`p` and `p_2050`). The CLI prints the same pair. A confusion matrix is the wrong slide.

### 7.3 Suitability map for one species (research / poster)

```bash
python future_climate_rasters.py          # once, if the 19-band 2050 file is missing
python suitability_maps.py --species "Quercus robur"
```

Writes `maps/Quercus_robur_{now,2050,delta}.png` plus GeoTIFFs and `maps/Quercus_robur.html` (Europe crop by default: `--bbox -15,35,40,72`). Ocean/ice masked with `planting_exclusion_mask_10m.tif`. Same models, BIO swap only.

Do this for a handful of flagship species, not all 171.

### 7.4 City / country dashboard

Aggregate is more honest than a 18 km “yard” pin:

- Geocode a city → take the cell (or mean `p` over cells intersecting a polygon).
- Table: species × (now, 2050, native?, goal tags).
- Filter rows with the satellite exclusion mask so rivers and sea do not get oaks.

### 7.5 What not to visualize

- Raw 42-feature heatmaps (unreadable).
- AUC as a map (AUC is per species, not per cell).
- NDVI as “planting suitability.”

---

## 8. Mental model for new developers

```
                    ┌─ satellite exclusion / water / city  →  block or warn
User pin ─► 42 numbers
                    ├─ XGBoost × N species  →  p ∈ [0,1]   →  rank + badges
                    └─ (optional) swap 19 BIO with 2050    →  p_2050

Traits CSV ──────────► goals, care, native/invasive text
GBIF + rasters ──────► training only (not the live UI)
```

**Training** learns a climate–soil niche from presence vs background.  
**Recommend** evaluates that niche at one cell and translates numbers into language.  
**Satellite** decides whether the cell is a place a person can plant.  
**2050** asks the same model a what-if, without new labels.

---

## 9. Quick commands

```bash
# Train (GPU node)
python xgboost_training.py

# Satellite site filters (CPU, HTTP). Not a training input.
python satellite_rasters.py

# Refresh editable species cards after a new training run
python recommend.py --write-traits-stub

# Human-readable POC + HTML pin page
python recommend.py --lat 51.51 --lon -0.13 --goal shade --html suggest.html

# English oak today vs 2050 (Europe crop)
python future_climate_rasters.py
python suitability_maps.py --species "Quercus robur"

# For a frontend
python recommend.py --lat 45.76 --lon 4.84 --goal shade --json
```
