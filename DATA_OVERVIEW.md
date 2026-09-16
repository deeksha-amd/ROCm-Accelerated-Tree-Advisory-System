# Data Overview

What each dataset is, where it came from, and what we assume when using it.

**Shared grid:** every raster uses the same 2160 x 1080 global grid. One cell is
1/6 degree, about **18.5 km** across. ~808,000 of 2.3M cells are land. So cell
(row 500, col 300) is the same place in every file.

---

## The files

| Folder / file | What it is | Source | Year |
|---|---|---|---|
| `gbif_500_species.csv` | 392,725 "species seen here" records, 293 species, 143 countries | GBIF | 1500-2026, median **2024** |
| `climate_current/` | 19 climate variables (temperature, rainfall, seasonality) | WorldClim 2.1 | **1970-2000** average |
| `climate_recent/` | Same 19, **recommended for training** | Derived from WorldClim monthly | **2015-2024** |
| `climate_2000_2015/` | Same 19, backup window | Derived from WorldClim monthly | **2000-2015** |
| `climate_future_2050/` | Same 19, projected. One file, 19 layers | WorldClim / CMIP6, ssp245 | **2041-2060** |
| `soil_data/aligned_10m/` | 11 soil properties x 2 depths = 22 layers | OpenLandMap + SoilGrids | 2020-2022 / 2020 |
| `soil_data/soilgrids_5km/` | Raw reference copies. **Do not train on these** | SoilGrids 2.0 | 2020 |
| `topography/` | Elevation on the shared grid. Nothing else | GEDTM30 v1.2 | 2006-2015 data |
| `satellite/` | Land-cover **filters** for `recommend.py` (water, built-up, crop, snow, tree, exclusion). Vegetation is **not** an XGBoost input. Built by `satellite_rasters.py` | ESA WorldCover 10 m (2021 v200; 2020 fallback) | 2021 |
| `validation/` | Evidence for why these sources were picked. **Not a model input** | various | — |

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

**Do not train on the vegetation layers.** NDVI and tree cover measure where
trees *already* grow, so predicting where trees *can* grow from them is circular.
Measured: vegetation layers alone — no climate, soil or terrain — reach
cross-validated AUC **0.791** vs **0.711** for the entire genuine environmental
set. But that +0.080 lead falls to **+0.001 on cleared farmland**, which is the
land we actually advise on. Tree cover also separates **90% of 40 species in the
same direction** (temperature: 52%) — it answers "are there trees here?", not
"what niche is this?".

The POC filter stack is built by `python satellite_rasters.py`: ESA WorldCover
10 m class maps, streamed as COG overviews and averaged onto the shared 10-arc-
minute grid. `recommend.py` reads those GeoTIFFs **after** scoring, never as `x`.

- **Filters only:** built-up, water, snow/ice, land and cropland fractions, plus
  `planting_exclusion_mask_10m.tif`. Apply to model **output**, never input.
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
`validation/topography/`, not in `topography/`, which holds only the production
elevation layer.

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
