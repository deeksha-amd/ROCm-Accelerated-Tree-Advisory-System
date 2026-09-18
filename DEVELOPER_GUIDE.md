# Developer guide: tree SDM training → recommendations

Start with **`README.md`** for the current folder layout and copy-paste run commands.

This file is the engineering map: how models are trained, what they write, how a pin becomes a shortlist, and how **satellite filters**, **2050 climate**, and **maps** plug in.

**USA 1 km at the repo root is the main product.** The global 18 km demo lives in `poc/`. Section 3–8 below still describe that 18 km path in detail; section 10 is the USA 1 km contract.

If you only need data provenance (sources, licences, why GEDTM30), read `DATA_OVERVIEW.md`.

---

## 1. What the system is

Each saved file in `data/models/` is a **binary XGBoost classifier for one tree species**.

It does **not** answer “which of 171 species lives here?” It answers, independently:

> Given this cell’s climate, soil, and elevation, how much does it look like places where *this* species has been recorded?

A location is never a feature. Latitude/longitude are only used to **look up** a 42-number vector on a shared 10-arc-minute grid (~**18.5 km** cells, 2160 × 1080 globally). Two yards in the same cell get the same score.

```
GBIF points + rasters          XGBoost (one model / species)         App
─────────────────────          ────────────────────────────          ───
occurrences (y=1)              spatial-block CV + early stop         poc/recommend.py
pseudo-absences (y=0)    →     final model on all rows         →     top 5 cards
42 env numbers at each cell    data/models/*.json + metrics.csv      (JSON or text)
```

---

## 2. Repository map

| Path | What a developer uses it for |
|---|---|
| `README.md` | Run steps (start here) |
| `xgboost_training_usa_30s.py` | Train USA 1 km models → `data/models_usa_30s/` |
| `clean_species_usa_30s.py` | Clean + thin US GBIF onto the **1 km** CONUS grid |
| `recommend_usa_30s.py` | Location → top 5 from **USA 1 km** models |
| `future_climate_usa_30s.py` | Honest 1 km 2050 BIO (change-factor delta) |
| `suitability_maps_usa_30s.py` | USA today vs 2050 maps → `maps/` |
| `poc/` | Global **18 km** POC (train, recommend, catalog, oak/Seville maps) |
| `data/` | Rasters, GBIF tables, species lists, and saved models |
| `data/scripts/` | Download / align climate, soil, topography, satellite, GBIF |
| `maps/` | USA pin/map HTML |
| `data/usa_tree_species_seed.csv` | Short plantable US list (train this first) |
| `data/usa_tree_species_list.csv` | Full US checklist (grow later) |
| `data/species_traits.csv` | Editable overlay on species cards |
| `data/models/` | Global 18 km `*.json` boosters |
| `data/models_usa_30s/` | USA 1 km `*.json` boosters |
| `DATA_OVERVIEW.md` | Why each dataset, what not to train on |

**Run from the project root** (`hackathon_2026/`). `repo_paths.py` resolves `data/` from the repo root. GPU is required for training (`device="cuda"` / ROCm on MI300X); recommend runs on CPU.

USA rasters and `data/models_usa_30s/*.json` are **gitignored**. A clone must copy them or train; see `README.md`.

```bash
source venv/bin/activate
python xgboost_training_usa_30s.py --full-list --skip-existing
python data/scripts/satellite_rasters_usa_30s.py --smoke
python recommend_usa_30s.py --lat 30.2672 --lon -97.7431 --goal shade --html maps/austin.html
```

---

## 3. Training flow (`poc/xgboost_training.py`)

### 3.1 Load predictors once

`collect_predictor_paths()` stacks, in this order (must stay stable):

1. All `data/climate_current/*.tif` (19 BIO layers, **sorted by filename** — so `bio_10` comes before `bio_2`)
2. `data/soil_data/aligned_10m/*.tif` except `source_flag_10m.tif` (22 layers)
3. `data/topography/gedtm30/gedtm30_v1.2_elev_10m.tif`

That list is written to `data/models/feature_names.txt`. Inference **must** sample layers in the same order.

Do **not** train on `data/soil_data/soilgrids_5km/` (wrong grid) or vegetation satellite layers (circular: they measure existing trees).

### 3.2 Per species

For each unique name in `data/gbif_500_species.csv`:

1. **Keep trees only.** Binomial whose genus is in `TREE_GENERA`. Drops moths, birds, bacteria that matched a GBIF genus search (`Carduelis pinus`, `Erigeron acer`, …).
2. **Snap to unique grid cells.** Many GBIF rows share one 18 km pixel. Count unique cells, not records.
3. **Skip sparse ranges.** `< MIN_UNIQUE_CELLS` (80) → `skip_few_cells`. Spatial CV is noise on a handful of cells.
4. **Pseudo-absences.** Random land pixels inside a padded bounding box of the occurrences, only where climate+soil+elev are all valid. Equal count to presences (or fewer if the box is small). Global background is avoided on purpose (it made AUC `nan` or ~0.99).
5. **Labels.** Presence `y=1`, background `y=0`. Features = 42 raster values at those coordinates. Rows with any NaN dropped.
6. **Spatial CV.** K-means on map coordinates → ~10 geographic blocks. `StratifiedGroupKFold` holds whole blocks out (5 folds). A fold is used only if train and test both contain 0s and 1s. ROC AUC needs both classes.
7. **Fit.** `XGBClassifier`, histogram trees, GPU, early stopping on the held-out block. Regularization is lighter when `n` is small (`min_child_weight` 2 / 4 / 8).
8. **Final model.** Retrain on all rows with `n_estimators = median(best_iteration)` from CV (floor 10). That is the file we save.
9. **Save gate.** Write `data/models/{Genus_species}.json` only if AUC is finite **and** ≥ 3 mixed folds. Weak 2-fold AUCs (including lucky 0.96s) stay in `metrics.csv` as `skip_few_folds`.

### 3.3 What “AUC” means here

It is **spatial-block ROC AUC**, not random-split accuracy. Typical usable range is about **0.70–0.90**. Very high scores on well-sampled, widely planted species (`Pinus radiata`) can mean “easy climate envelope / plantations,” not a native woodland.

`poc/recommend.py` ignores saved models with AUC `< 0.70` (`catalog.MIN_AUC_DEFAULT`).

### 3.4 Training outputs

| File | Contents |
|---|---|
| `data/models/Quercus_robur.json` | XGBoost booster (sklearn `load_model`) |
| `data/models/metrics.csv` | One row per GBIF name: `status`, `auc`, `n_unique_cells`, `n_folds_usable`, `model_path` |
| `data/models/feature_names.txt` | 42 raster filenames in training order |

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

1. Replace or grow `data/gbif_500_species.csv` (same columns: `species`, `latitude`, `longitude`).
2. Re-run `python poc/xgboost_training.py`.
3. New `data/models/*.json` appear; `poc/recommend.py` loads whatever `metrics.csv` marks `saved`.
4. Optionally add a row to `data/species_traits.csv` for common name / shade-food-beauty. Missing rows fall back to genus defaults in `poc/catalog.py`.

You do **not** need to change `poc/recommend.py` for new species.

---

## 4. Recommendation flow (`poc/recommend.py`)

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

`data/species_traits.csv` is **not** from GBIF. It is generated from `poc/catalog.py` (`python poc/recommend.py --write-traits-stub`): genus defaults plus a hand list of well-known common names. Obscure taxa get names like “Griffithii oak”. Edit the CSV; `poc/recommend.py` prefers those cells over code defaults.

---

## 5. Satellite data — use as a **site filter**, not a training feature

`DATA_OVERVIEW.md` measured this: NDVI / tree-cover alone can look like a strong predictor (AUC ~0.79) and then **collapse on farmland** (the land we advise on). Those layers answer “are there trees here already?”, not “what can establish?”

### Do not put in `x` (the 42 features)

- Tree / broadleaf / needleleaf / shrub / grass fractions
- All NDVI layers

### Download (aligned to the shared grid)

Do **not** download the 10 m WorldCover mosaic (~120 GB). `satellite_rasters.py` streams COG overviews over HTTP and averages class fractions onto `data/climate_current/wc2.1_10m_bio_1.tif` (2160×1080), the same pattern as `soil_rasters.py`.

```bash
python data/scripts/satellite_rasters.py --smoke     # 4 tiles: London, Portland, Lyon, Pacific
python data/scripts/satellite_rasters.py             # global, ~10 min with 8 workers
python data/scripts/satellite_rasters.py --bbox -10,40,10,60 --workers 8
```

Writes `data/satellite/*.tif` (gitignored). No XGBoost retrain. `poc/recommend.py` reads those files on the next run.

`--smoke` / `--bbox` leave the rest of the globe as NaN so other cities are skipped, not marked as ocean. A full run treats unmapped cells as water.

### USA 1 km filters (do not use the 18 km stack)

`recommend_usa_30s.py` reads `data/country_data/USA/satellite_30s/*_30s.tif` on the same 7020×3060 cell as the map box. It never falls back to `data/satellite/`.

```bash
python data/scripts/satellite_rasters_usa_30s.py --smoke   # Austin + Portland
python data/scripts/satellite_rasters_usa_30s.py           # CONUS
```

Same WorldCover source and class rules as the 18 km script; different destination grid. Vegetation is still **not** an XGBoost input (`DO_NOT_TRAIN_ON_THIS.md` in that folder; trainer refuses the `satellite` path token).

### Do apply **after** `predict_proba`

`poc/recommend.py` looks for `data/satellite/` (same 2160×1080 grid). If the folder is missing, it prints a skip note and continues. Exact names from `satellite_rasters.py` are preferred; substring match is the fallback:

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

`data/climate_future_2050/wc2.1_10m_bioc_MPI-ESM1-2-HR_ssp245_2041-2060.tif`

One GeoTIFF, **19 bands** = BIO1…BIO19 (band 1 = BIO1). Soil and elevation stay as they are (static-soil assumption).

### Pairing rule (easy to get wrong)

This 2050 file was bias-corrected against **WorldClim 1970–2000** (`data/climate_current/`). Compare future scores to models trained on `data/climate_current/`, which is what we have now.

If you later retrain on `data/climate_recent/` (2015–2024), **do not** subtract that from this 2050 product — you would understate warming. You would need a future layer bias-corrected to the same recent baseline.

### Feature-order rule (also easy to get wrong)

Training order is **filename sort**, not BIO1–BIO19:

`bio_1, bio_10, bio_11, …, bio_19, bio_2, …, bio_9`

Future band `k` is BIO`k`. When building `x_2050`, copy soil+elev from today and insert BIO values in **`feature_names.txt` order**, not band order 1..19.

Sketch:

```python
import rasterio
import numpy as np

names = [ln.strip() for ln in open("data/models/feature_names.txt")]
# x_now: 42 values in that order (recommend.sample_predictors)
with rasterio.open("data/climate_future_2050/wc2.1_10m_bioc_MPI-ESM1-2-HR_ssp245_2041-2060.tif") as src:
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

`poc/recommend.py` does this at the pin automatically when `data/climate_future_2050/wc2.1_10m_bioc_MPI-ESM1-2-HR_ssp245_2041-2060.tif` is present (`--html` shows paired bars). `poc/suitability_maps.py` does it for every land cell of one species.

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
python poc/recommend.py --lat 51.51 --lon -0.13 --goal shade --html poc/maps/suggest.html
```

`poc/maps/suggest.html` is a Leaflet page: map, the **18 km cell** as a rectangle (green plantable / amber city or farm / red blocked), satellite mix bar, top-5 cards with **today vs 2050** bars. Needs the network for Leaflet and Esri tiles (OSM/CARTO block or watermark `file://` pages).

Zoom is capped (`maxZoom` 8 on fit). The caption states the score is for the cell, not a backyard.

### 7.2 Top-5 bar chart

Those bars live on the HTML cards (`p` and `p_2050`). The CLI prints the same pair. A confusion matrix is the wrong slide.

### 7.3 Suitability map for one species (research / poster)

```bash
python data/scripts/future_climate_rasters.py          # once, if the 19-band 2050 file is missing
python poc/suitability_maps.py --species "Quercus robur"
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

Same list, with more context, in `README.md`.

```bash
# USA 1 km train (GPU node)
python xgboost_training_usa_30s.py --full-list --skip-existing

# USA 1 km Deep SDM train (same data, one network for every species)
python deepmaxent_training_usa_30s.py --full-list

# USA 1 km recommend
python future_climate_usa_30s.py
python data/scripts/satellite_rasters_usa_30s.py --smoke
python recommend_usa_30s.py --lat 30.2672 --lon -97.7431 --goal shade --html maps/austin.html
python recommend_usa_30s.py --model deepmaxent --lat 30.2672 --lon -97.7431 --goal shade
python suitability_maps_usa_30s.py --species "Quercus virginiana"

# grow the list after the seed run (keeps models already on disk):
python clean_species_usa_30s.py --species-list data/usa_tree_species_list.csv
python xgboost_training_usa_30s.py --full-list --skip-existing

# 18 km POC (does not touch data/models_usa_30s/)
python poc/xgboost_training.py
python data/scripts/satellite_rasters.py
python poc/recommend.py --write-traits-stub
python poc/recommend.py --lat 51.51 --lon -0.13 --goal shade --html poc/maps/suggest.html
python data/scripts/future_climate_rasters.py
python poc/suitability_maps.py --species "Quercus robur"
python poc/recommend.py --lat 45.76 --lon 4.84 --goal shade --json
```

---

## 10. USA 1 km path (separate from the 10-arc-minute POC)

`poc/xgboost_training.py` stays on the **global 10-arc-minute** grid (`data/models/`).
Do not point it at `data/country_data/USA`.

The 1 km CONUS trainer is a different contract:

| Piece | Path |
|---|---|
| Scripts | repo root (`xgboost_training_usa_30s.py`, `recommend_usa_30s.py`, …) |
| Rasters | `data/country_data/USA/{climate,soil,topography}_30s/` (7020×3060, ~1 km) |
| Seed species | `data/usa_tree_species_seed.csv` (~32 well-known trees) |
| Full checklist | `data/usa_tree_species_list.csv` (grow later) |
| Raw GBIF | `data/species_occurrences/US/gbif_trees_US_raw.csv` |
| Cleaner | `clean_species_usa_30s.py` |
| Thinned table | `data/species_occurrences/US/gbif_trees_US_30s_thinned.csv` |
| Trainer | `xgboost_training_usa_30s.py` |
| Deep SDM trainer | `deepmaxent_training_usa_30s.py` |
| Recommend | `recommend_usa_30s.py` (CONUS 1 km; not `poc/recommend.py`) |
| 2050 BIO | `future_climate_usa_30s.py` → `data/country_data/USA/climate_future_2050_30s/` |
| Satellite filters | `data/scripts/satellite_rasters_usa_30s.py` → `data/country_data/USA/satellite_30s/` |
| Record-likeness maps | `suitability_maps_usa_30s.py` → `maps/` (default live oak, south-central US) |
| Models | `data/models_usa_30s/*.json` (gitignored; `metrics.csv` is tracked) |

Differences from the 10-arc-minute run that matter:

- **BIO order is numeric** (`bio_1` … `bio_19`), not filename sort.
- **61 predictors**: 19 BIO + 9 climate extras + 22 soil + 11 terrain. No satellite, no `source_flag`, no WorldClim elevation duplicate, no raw aspect degrees.
- **Background is target-group** (other list-tree cells), not random land. Recommend copy says **record-likeness**, not planting suitability.
- Invasive / naturalised checklist trees are **trained**, then **dropped from the top-5**.
- **Gate is 150 unique 1 km cells** and ≥ 3 mixed spatial folds.
- **2050 BIO** is `python future_climate_usa_30s.py` (10-arcmin ssp245 anomaly onto the 1 km training climate). Soil/terrain stay put. Do not upsample the global 10m cube.
- **Satellite filters** must be the 1 km `satellite_30s/` stack. A git clone has neither rasters nor JSON boosters.
- Needs ~6 GB RAM to hold the raster stack during training. `poc/recommend.py` still reads the **10-arc-minute** models; USA pins use `recommend_usa_30s.py`.

---

## 11. The second USA 1 km model: DeepMaxent (`--model deepmaxent`)

`deepmaxent_training_usa_30s.py` trains a Deep SDM on **exactly** the inputs
section 10 describes. It imports `collect_predictor_paths`, `Usa30sPredictors`,
`load_occurrence_table`, `spatial_block_ids` and the gate constants straight
from `xgboost_training_usa_30s.py` rather than restating them, so the two
models cannot drift onto different data. The network and its maximum-entropy
loss are vendored verbatim in `deepmaxent/` from
[RYCKEWAERT/deepmaxent](https://github.com/RYCKEWAERT/deepmaxent).

What is genuinely different, and why:

| | XGBoost | DeepMaxent |
|---|---|---|
| Shape | one booster per species | one network, one output per species |
| Training rows | per-species presence + sampled background | one row per target-group **cell**, all species at once |
| Background | 1:1 sample of other-tree cells | every other-tree cell, via the maxent normaliser |
| Missing layer at a cell | native missing-value branch | cell dropped in training; read as the US mean at score time |
| Artifacts | 255 `.json` | one `.pt` |

**Why the cell table.** The maxent loss is `-(y * logsoftmax(λ, dim=cells))`:
it normalises each species' log-intensity **over cells**, so a row has to be a
cell and a column has to be a species. The per-species presence/background
draws XGBoost uses cannot express that. The cell universe is still exactly the
target-group idea — a cell is in the table if any listed tree was recorded
there — so the evidence is unchanged, only its shape.

**Reading the score.** The network's λ is only defined up to a per-species
constant. Training stores `log_z_s = log mean_cells exp(λ_s)` and recommend
reports `p = sigmoid(λ_s − log_z_s)`, so `p = 0.5` is an average target-group
cell. That is the same question the boosters answer with their 1:1 sampling,
which is why the two models share thresholds and one `metrics.csv` schema.

**Gating is unchanged**: 150 unique 1 km cells, spatial-block CV, ≥ 3 usable
folds, and `load_saved_models` reads the same
`species,auc,n_folds_usable,model_path,status` columns. A species that fails
the fold gate still has an output in the network; it is simply not offered.

**Weight decay is the accuracy fix; learning rate and batch size are speed.**
Upstream's `3e-4` was set on the much smaller Elith/NCEAS benchmark and
over-regularises 245,276 target-group cells × 255 species. At `2e-5` the gated
mean AUC goes from 0.808 to 0.853, and from 0.793 to 0.824 on the 32 species
XGBoost also gates, where the boosters score 0.845; every one of the five
spatial folds improved. The optimum is sharp rather than monotonic — `3e-3`
collapses to 0.72 and `0` is also poor, with a broad plateau between 1e-5 and
5e-5 — so this is not "less regularisation is better". The other two changed
defaults, `--learning-rate 1e-3` and `--batch-size 4096`, buy roughly 4x
throughput and nothing else: a 100x learning-rate range and a 33x batch range
each moved AUC by under 0.005. Extra capacity helped only while the weight
decay was wrong, so the architecture stays upstream's.

**Feature order is the contract.** `DeepMaxentSDM.assert_feature_order` refuses
to score if the caller's 61 layers are not the checkpoint's 61 layers, because
a silent reorder would score every species against the wrong rasters.

The per-pin "which layers mattered" panel differs by necessity: XGBoost reports
species-wide gain, DeepMaxent reports `|∂λ_s/∂z_j|` at that pin (a local
gradient, per 1 SD of each layer). The captions say which one you are reading.
