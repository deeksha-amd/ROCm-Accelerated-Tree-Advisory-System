# Data Overview

What each dataset is, where it came from, and what we assume when using it.
Copy-paste run commands are in `README.md`.

**Layout:** USA 1 km XGBoost scripts live at the repo root. The global 18 km
XGBoost POC lives under `poc/`. Rasters, GBIF, models, and XGBoost downloaders
live under `data/` (`data/scripts/` for downloads). The deep-learning SDM
pipeline lives under `deep_learning_sdm/` and uses `country_data/` at the repo
root (manifests committed; GeoTIFFs gitignored).

**Shared grid:** every raster uses the same 2160 x 1080 global grid. One cell is
1/6 degree, about **18.5 km** across. ~808,000 of 2.3M cells are land. So cell
(row 500, col 300) is the same place in every file.

---

## The files

| Folder / file | What it is | Source | Year |
|---|---|---|---|
| `data/gbif_500_species.csv` | 392,725 "species seen here" records, 293 species, 143 countries | GBIF | 1500-2026, median **2024** |
| `data/climate_current/` | 19 climate variables (temperature, rainfall, seasonality) | WorldClim 2.1 | **1970-2000** average |
| `data/climate_recent/` | Same 19, **recommended for training** | Derived from WorldClim monthly | **2015-2024** |
| `data/climate_2000_2015/` | Same 19, backup window | Derived from WorldClim monthly | **2000-2015** |
| `data/climate_future_2050/` | Same 19, projected. One file, 19 layers | WorldClim / CMIP6, ssp245 | **2041-2060** |
| `data/soil_data/aligned_10m/` | 11 soil properties x 2 depths = 22 layers | OpenLandMap + SoilGrids | 2020-2022 / 2020 |
| `data/soil_data/soilgrids_5km/` | Raw reference copies. **Do not train on these** | SoilGrids 2.0 | 2020 |
| `data/topography/` | Elevation on the shared grid. Nothing else | GEDTM30 v1.2 | 2006-2015 data |
| `data/satellite/` | Land-cover **filters** for `poc/recommend.py` (water, built-up, crop, snow, tree, exclusion). Vegetation is **not** an XGBoost input. Built by `data/scripts/satellite_rasters.py` | ESA WorldCover 10 m (2021 v200; 2020 fallback) | 2021 |
| `data/country_data/USA/` | CONUS 1 km climate, soil, terrain (and USA 2050 BIO) | WorldClim / SoilGrids / GEDTM30 | mixed |
| `data/species_occurrences/US/` | Cleaned + thinned US GBIF for the 1 km trainer | GBIF | mixed |
| `data/models/` | Global 10-arc-minute XGBoost JSON | this repo | — |
| `data/models_usa_30s/` | CONUS 1 km XGBoost JSON | this repo | — |
| `data/validation/` | Evidence for why these sources were picked. **Not a model input** | various | — |
| `country_data/{FRA,ESP,USA}/` | 1 km country stacks for the deep-learning pipeline (manifests in git) | WorldClim / SoilGrids / GEDTM30 | mixed |
| `deep_learning_sdm/` | Deep-learning SDM train + advisory (separate from XGBoost) | this repo | — |

---

## Climate windows — which to use

`climate_recent/` and `climate_2000_2015/` were **derived**, not downloaded:
WorldClim publishes ready-made bioclim variables only for 1970-2000, so these
were computed from its monthly temperature and rainfall grids (downscaled from
CRU TS 4.09). The derivation was checked by recomputing 1970-2000 and
reproducing WorldClim's published rasters **bit-for-bit across all 808,053 land
cells**. Filenames match `climate_current/`, so switching is a one-line path
change.

**2024 is the last year available** — no observational product runs to 2026.

**The climate really has moved.** At the GBIF occurrence points, annual mean
temperature is **+1.28°C** warmer in 2015-2024 than in 1970-2000, and warmer at
**99.92%** of points. That is comparable to the entire warming in the 2050
projection. Rainfall shifted much less (+2.6%) but redistributed regionally.

**Use `climate_recent/`** — it matches when the occurrences were actually
recorded. `climate_2000_2015/` is the backup: same source, same code, and its
16-year span smooths interannual noise better than the 10-year recent window
(which contains two strong El Niño events).

> Note: `climate_future_2050/` was bias-corrected against the **1970-2000**
> baseline, so it pairs with `climate_current/`, not with the new windows.
> Comparing it to `climate_recent/` would understate future warming.

---

## Soil properties

| Code | Plain meaning |
|---|---|
| `ph_h2o` | Acidity / alkalinity |
| `sand` / `silt` / `clay` | Texture — grain size mix, sums to 100% |
| `bdod` | How compacted; high means hard for roots |
| `soc` / `socd` | Organic carbon (content / per volume) |
| `nitrogen` | Main growth nutrient |
| `cec` | How well soil holds nutrients against rain |
| `cfvo` | % stones and gravel |
| `drainage` | How fast water drains |

**Depths:** each property is averaged over two slabs below the surface —
`0-30cm` is the topsoil holding the nutrients and feeder roots that decide
whether a sapling establishes, `30-60cm` is the subsoil holding structural roots
and the moisture reserve a mature tree draws on in drought.

**Provenance differs by property:**

- **OpenLandMap** (published Feb 2026, data 2020-2022): `bdod`, `clay`,
  `ph_h2o`, `sand`, `silt`, `soc`, `socd`
- **SoilGrids only** (2020 release): `cec`, `nitrogen`, `cfvo` — OpenLandMap
  does not publish these
- **Derived, not measured:** `drainage`, calculated from sand and clay. Treat as
  a rough ranking.

OpenLandMap excludes hot deserts and stops at 76°N, so **69,500 cells (~12%)**
were backfilled from SoilGrids. `source_flag_10m.tif` marks which
(1 = OpenLandMap, 2 = SoilGrids), but it is a single global file describing the
seven OpenLandMap properties — it is misleading if read for `cec`, `nitrogen`
or `cfvo`.

---

## Satellite — filters, not predictors

**Do not train on the satellite layers** — but the reason is not the one first
recorded here. The original global finding (vegetation-only AUC 0.791 vs 0.711,
collapsing to +0.001 on cleared farmland; tree cover separating 90% of species in
one direction) **did not reproduce** on the France 1 km data, which is finer and
better sampled. Re-measured there, across 9 feature sets and 5 tree-cover strata:

- Satellite-only scores **0.706 vs 0.742** for the environmental set — 0.036
  *worse*, not 0.080 better.
- Adding satellite gains **+0.012**, and that gain **does not collapse** on
  treeless land (+0.015) or cropland (+0.011).
- Tree-cover direction agreement is **60.9%** vs **78.3%** for temperature — the
  global ordering reversed.

The exclusion still stands, because the gain is **not ecology**. Two thirds of it
is one layer, built-up fraction, absorbing residual survey bias: presence cells
average **9.9% built-up vs 2.8%** for background, and **21.9% vs 5.3%** on
treeless land. Its value is +0.016 on treeless land and **−0.001 in forest** —
the inverse of an ecological signal. The remainder is land *use*, which encodes
the advisory's own decision variable: down-ranking a field because trees are not
recorded in fields answers "is this land in use?", not "will a tree grow here?".

- **Do not use as predictors:** all of it. Tree cover and NDVI are worth only
  +0.007 and are nearly null in France (27.0% vs 25.3% presence/background).
  `soilmoisture_*` — previously called the one defensible layer — is worth
  **exactly +0.000** across every stratum.
- **Filters only:** built-up, water, snow/ice, land and cropland fractions, plus
  `planting_exclusion_mask_10m.tif`. Apply to model **output**, never input.
  46.6% of land carries no exclusion. Built-up is the strongest single
  contributor and the least defensible — it is already used correctly, as a hard
  output exclusion.

**Caveat on the re-measurement:** the background is effort-weighted only at
18.5 km, which is the exact channel built-up fraction exploits at 1 km. A 1 km
effort surface would likely shrink the satellite gain further.

**Coverage:** land cover **99.93%** of GBIF points (matches climate); NDVI
**93.4%** (gaps are desert and ice, masked at source); soil moisture **79.2%**.

**Known limit:** tree fraction understates open woodland — Iberia reads 8% vs
22% in 10 m data — but agrees within a few points in closed tropical forest.

For the XGBoost POC, the filter stack is built by
`python data/scripts/satellite_rasters.py`: ESA WorldCover 10 m class maps,
streamed as COG overviews and averaged onto the shared 10-arc-minute grid.
`poc/recommend.py` (and USA recommend, for the same 18 km filters at a pin)
reads those GeoTIFFs **after** scoring, never as `x`.

- **Do not use as XGBoost features:** tree / broadleaf / needleleaf / shrub /
  grass fractions, and any NDVI layer.
- **Optional extra predictor (retrain required):** `soilmoisture_mean` /
  `soilmoisture_amplitude` — microwave readings, not greenness. Not produced by
  the default downloader; they cover only ~79% of GBIF points, with holes under
  dense canopy.

---

## Why GEDTM30 and not SRTM or Copernicus

- **SRTM** has no data above 60°N — would silently drop 8,151 treeline records.
- **Copernicus** measures the **forest canopy**, not the ground: +15.4 m error
  over Amazon rainforest vs +0.3 m over tundra. That error tracks forest
  density, so it leaks what we are predicting into a predictor.

GEDTM30 is **bare-earth** and covers 85°N-65°S.

The rival DEMs and sample tiles those numbers were measured on live in
`data/validation/topography/`, not in `data/topography/`, which holds only the
production elevation layer.

---

## Assumptions

1. **A species' climate preference is fixed** — in space today, and in 2050.
   This underpins all species distribution modelling.
2. **The climate window matches the sightings.** Only 12.5% of records fall in
   the 1970-2000 window, against a median record year of 2024 — which is why
   `climate_recent/` exists. Using `climate_current/` instead assumes a tree
   recorded today germinated decades ago under that older climate: defensible
   for long-lived trees, but an assumption, and one that now costs 1.28°C.
3. **Soil and terrain are static.** Safe for texture, stones, elevation. Weaker
   for **nitrogen and carbon**, which shift over decades. Treat high nitrogen
   importance with caution.
4. **An 18.5 km cell describes a place.** One cell can hold a mountain and a
   valley; the model sees the average.
5. **No record ≠ absent.** GBIF is presence-only and biased toward well-surveyed
   countries (US alone: 109,513 records). Pseudo-absences may sit on unsurveyed
   ground where the species does grow.
6. **Averaging fine data up is acceptable** — true for elevation, **false for
   slope**, which must be computed at fine resolution first (otherwise it is
   understated ~21x).

---

## Known gaps

- **Antarctica:** no soil (none exists) and no terrain (source stops at 65°S).
  The only 3 records below that line are corrupt — a Himalayan pine tagged in
  Antarctica.
- **484 records (0.12%)** have no soil — narrow coastlines and small islands.
- **14,430 cells** have some soil properties but not all.
- **The derived climate windows cover 578,888 land cells** vs 808,053 in
  `climate_current/`. 97% of that difference is Antarctica; at the GBIF points
  the cost is only 412 records (99.83% valid vs 99.93%).
- **Usable for training: 392,121 of 392,725 (99.85%)** with climate, soil and
  terrain all valid.

---

## Licence note

The climate data derives from WorldClim 2.1, which is free for academic and
non-commercial use; redistribution or commercial use needs permission from
worldclim.org. That restriction carries over to `climate_recent/` and
`climate_2000_2015/` — **do not redistribute them**. Soil (CC-BY 4.0) and
terrain (CC-BY 4.0) are permissively licensed.
