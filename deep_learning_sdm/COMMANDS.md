# Commands

Every command below is run **from the repository root**, not from inside
`deep_learning_sdm/`. The scripts live in `deep_learning_sdm/`; the data trees
they read and write live at the root. Each data path in `sdm_features.py`
resolves relative to the working directory, so the root is where you stand:

```bash
cd /path/to/ROCm-Accelerated-Tree-Advisory-System
python deep_learning_sdm/deep_sdm_training.py --smoke
```

On this host the interpreter is the project venv, which already carries a ROCm
build of PyTorch, so substitute `./venv/bin/python` for `python` throughout.

---

## 0. What ships, and what you have to rebuild

The repository carries the code, the docs, the occurrence and background tables
the model trains on, and the checkpoint the advisory serves. It does **not**
carry the rasters: 274 MB for France and 2.4 GB for the USA, all of it derived
from public sources and all of it rebuilt by one command.

| In the repo | Rebuild with |
|---|---|
| `species_occurrences/gbif_trees_{clean,thinned}.csv` | `prepare_species_data.py` |
| `sampling_effort/background_FR*.csv`, effort GeoTIFFs | `sampling_effort.py --surface --background` |
| `deep_sdm_output/fra_30s_96sp_bg1km/` checkpoint + metrics | `deep_sdm_training.py` |
| `country_data/*/**/MANIFEST.json` (provenance only) | — |
| *not* the `country_data/` rasters | `prepare_country.py FRA` |
| *not* the global raster blocks | `soil_rasters.py`, `topography_rasters.py`, `recent_climate_rasters.py` |

**On a fresh clone, run this first.** Nothing else works until it finishes,
because both training and serving read the 1 km French rasters, and the
boundary polygon the advisory clips to is downloaded as part of it:

```bash
python deep_learning_sdm/prepare_country.py FRA
```

It is resumable, so an interrupted run picks up where it stopped.

---

## 1. Install dependencies

```bash
pip install -r deep_learning_sdm/requirements.txt
```

Or the bare minimum:

```bash
pip install rasterio numpy pandas tqdm affine shapely requests flask torch
```

**Do not `pip install torch` on a ROCm machine that already has it.** The PyPI
default is the CUDA build and it will silently lose the GPU.
`requirements.txt` leaves torch out for that reason.

Check the GPU is visible before anything else:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# True AMD Instinct MI210
```

The model runs on CPU too — pass `--device cpu` — it is just slower.

---

## 2. Download and prepare a country

One command builds a country's whole data tree, matching `country_data/FRA/`.

```bash
python deep_learning_sdm/prepare_country.py USA --plan   # extent and stages, no downloads
python deep_learning_sdm/prepare_country.py USA          # run it
python deep_learning_sdm/prepare_country.py --list       # countries with a profile
```

| Flag | Why |
|---|---|
| `--stage soil` | run one stage only (repeatable) |
| `--skip species` | rasters only, no GBIF download |
| `--jobs 4` | run independent raster blocks in parallel |
| `--force` | redo stages whose output already exists |
| `--bbox W,S,E,N` | override the extent |
| `--quiet` | stage output to `logs/` only |

Stages: `grid → worldclim → climate → clip10m → soil → terrain → satellite →
provenance → species_plan → species_download → species_clean →
species_effort → verify`.

Resumable — re-running skips finished stages, and the species download picks
up from checkpoints, which matters because GBIF throttles after sustained use.

**Which trees get downloaded** lives in `deep_learning_sdm/tree_species_list.csv`.
Edit that to change the recommendations; it is a product decision, not code.

Verify the output:

```bash
python deep_learning_sdm/verify_country_alignment.py FRA
```

---

## 3. Train the model

```bash
python deep_learning_sdm/deep_sdm_training.py --profile fra_30s --epochs 400   # 1 km, ~2.5 min
python deep_learning_sdm/deep_sdm_training.py --profile fra_10m --epochs 400   # 18.5 km
python deep_learning_sdm/deep_sdm_training.py --smoke                          # ~16 s sanity check
```

Check the inputs agree first:

```bash
python deep_learning_sdm/verify_sdm_inputs.py
```

Useful flags: `--occurrences`, `--background`, `--epochs`, `--hidden`,
`--folds`, `--device {auto,cuda,cpu}`, `--output-dir`, `--predict`.

Results land in `deep_sdm_output/<profile>/`: `run_summary.json`,
`cv_metrics_by_species.csv`, `cv_metrics_by_fold.csv`.

**Use `--epochs 400`.** The default of 60 cuts the 18.5 km run off before it
converges and makes it look worse than it is.

---

## 4. Get recommendations in the terminal

```bash
python deep_learning_sdm/advisory_service.py --point 43.5,5.4
```

| Flag | Effect |
|---|---|
| `--top 10` | show more than 5 options |
| `--no-metrics` | drop the numbers block at the end |
| `--no-nearby` | don't suggest nearby spots |
| `--self-test` | run the 5 built-in regression points |
| `--device cuda` | force the GPU |

---

## 5. Launch the web app

```bash
python deep_learning_sdm/advisory_app.py --host 0.0.0.0 --port 8000
```

Then open `http://localhost:8000`. Use `--host 127.0.0.1` (the default) to keep
it local to the machine.

JSON API, same data as the page:

```bash
curl "http://127.0.0.1:8000/api/recommend?lat=43.5&lon=5.4"
curl "http://127.0.0.1:8000/api/recommend?lat=43.5&lon=5.4&top=20"
curl "http://127.0.0.1:8000/api/health"
```

The app picks the run to serve from `deep_sdm_output/`, preferring
`fra_30s_96sp_bg1km` — the one committed here. On first start it draws a
reference score distribution and caches it in `advisory_cache/`, which takes
about 40 seconds; later starts are warm.

---

## 6. Validation and experiments

```bash
python deep_learning_sdm/validate_soil_alignment.py
python deep_learning_sdm/validate_topography_alignment.py
python deep_learning_sdm/validate_recent_climate.py
python deep_learning_sdm/validate_satellite_alignment.py
python deep_learning_sdm/verify_source_urls.py
```

The satellite circularity experiment, which is why no vegetation layer is a
predictor:

```bash
python deep_learning_sdm/deep_sdm_satellite_experiment.py
```

Its result summaries are committed under `satellite_experiment/results/`; see
`MODEL.md` and `../DATA_OVERVIEW.md` for what they mean.
