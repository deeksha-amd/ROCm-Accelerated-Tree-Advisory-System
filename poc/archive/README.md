# Archive — France deep SDM (not DeepMaxent)

Unused by the USA 1 km product (`xgboost_training_usa_30s.py`,
`deepmaxent_training_usa_30s.py`, `recommend_usa_30s.py`). Kept here as a
snapshot of the older France 1 km MLP + Flask advisory.

| Folder | What it was |
|---|---|
| `deep_learning_sdm/` | Trainers, downloaders, `advisory_app.py` |
| `deep_sdm_output/` | Served France checkpoint |
| `sampling_effort/` | Target-group backgrounds / effort rasters |
| `species_occurrences/` | Global/France GBIF (not `data/species_occurrences/US/`) |
| `satellite_experiment/` | NDVI circularity tables |

Scripts still assume **repo-root cwd** and paths like `species_occurrences/`
and `country_data/`. They will not run from this folder without path edits.
Do not confuse with USA DeepMaxent under `deepmaxent/` at the repo root.
