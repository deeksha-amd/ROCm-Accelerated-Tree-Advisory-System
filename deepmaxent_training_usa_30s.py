"""
DeepMaxent Deep SDM for the contiguous USA on the 1 km (30 arcsec) country grid.

A second model next to xgboost_training_usa_30s.py, trained on exactly the same
data. Everything about the inputs is imported from that module rather than
restated here, so the two models cannot drift apart:

  predictors   collect_predictor_paths()  — the same 61 layers in the same order
               (19 BIO, 9 climate extras, 22 soil, 11 terrain) on the USA
               7020 x 3060 grid, with the same DO_NOT_TRAIN_ON_THIS.md refusal
  occurrences  the same cleaned + thinned 1 km table from clean_species_usa_30s
  background   the same target-group idea: cells where some listed tree was
               recorded, never uniform land
  gating       the same MIN_UNIQUE_CELLS / spatial-block CV constants

Model: github.com/RYCKEWAERT/deepmaxent, vendored verbatim in deepmaxent/.

What differs from the XGBoost run, because DeepMaxent is a different estimator:

  One network, not 256 boosters. DeepMaxent has one output per species and a
  maximum-entropy loss that normalises over cells, so every species is fitted
  jointly on one shared set of target-group cells. Rows of the cell table are
  cells, not per-species presence/background draws, and the per-species
  presence/absence caps in the XGBoost trainer have nothing to cap here.

  Cells with a missing layer are dropped from training (the MLP has no
  native missing-value branch the way XGBoost does).

Outputs, all under data/models_deepmaxent_usa_30s/ so the XGBoost boosters in
data/models_usa_30s/ are untouched:

  deepmaxent_usa_30s.pt  one checkpoint: weights, species order, layer order,
                         standardiser, and per-species log_z
  metrics.csv            same species/auc/n_folds_usable/model_path/status
                         columns recommend_usa_30s.py already reads
  feature_names.txt      the 61-layer training contract

    python deepmaxent_training_usa_30s.py --smoke
    python deepmaxent_training_usa_30s.py --full-list
    python recommend_usa_30s.py --model deepmaxent --lat 30.2672 --lon -97.7431
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import rasterio
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

_ROOT = os.path.abspath(os.path.dirname(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from repo_paths import relpath
from clean_species_usa_30s import FULL_LIST, SEED_LIST, load_species_list, read_grid
from deepmaxent_sdm import (
    CHECKPOINT_PATH,
    FEATURE_NAMES_PATH,
    METRICS_PATH,
    MODEL_DIR,
    UPSTREAM,
    save_checkpoint,
)

# The data contract, imported so it stays identical to the XGBoost trainer.
from xgboost_training_usa_30s import (
    MIN_UNIQUE_CELLS,
    MIN_USABLE_FOLDS,
    N_SPATIAL_BLOCKS,
    N_SPATIAL_FOLDS,
    THINNED_CSV,
    Usa30sPredictors,
    collect_predictor_paths,
    load_occurrence_table,
    spatial_block_ids,
)

# Upstream's architecture and schedule (main_example.py, ConfigArgs); the
# optimiser settings below are retuned for this dataset, and these defaults are
# the ones the shipped checkpoint was trained with.
DEFAULT_HIDDEN_SIZE = 250
DEFAULT_HIDDEN_NBR = 2
# Learning rate and batch size are raised for speed, not accuracy: a 100x
# learning-rate range and a 33x batch range each moved AUC by under 0.005, but
# batch 4096 finishes a screen in ~52 s against ~184 s at upstream's 250.
DEFAULT_LEARNING_RATE = 1e-3
# Weight decay is the accuracy fix. Upstream's 3e-4 was tuned on the small
# Elith/NCEAS dataset and over-regularises 245k target-group cells x 255
# species; 2e-5 is worth ~4.5 AUC points on 5-fold spatial CV. It is a sharp
# optimum, not a trend — 3e-3 collapses to 0.72 and 0 is also poor — with a
# broad plateau between 1e-5 and 5e-5.
DEFAULT_WEIGHT_DECAY = 2e-5
DEFAULT_BATCH_SIZE = 4096
DEFAULT_EPOCHS = 100
DEFAULT_LOSS = "deepmaxent"
LOSS_CHOICES = ("deepmaxent", "poisson", "bce", "ce")

SEED = 42
# log_softmax over the batch is the maxent normaliser; a 1-cell batch is a
# constant and teaches nothing.
MIN_BATCH_CELLS = 2


def set_seed(seed):
    """Upstream librairies/utils.set_seed."""
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_criterion(loss_option):
    from deepmaxent import bce_loss, ce_loss, deepmaxent_loss, poisson_loss

    table = {
        "deepmaxent": deepmaxent_loss,
        "poisson": poisson_loss,
        "bce": bce_loss,
        "ce": ce_loss,
    }
    if loss_option not in table:
        raise ValueError("Loss option not recognized")
    return table[loss_option]()


class CellTable:
    """The target-group cell universe: one row per 1 km cell that holds a record
    of any listed tree, its 61 layer values, and which species were seen there.

    This is the same evidence the XGBoost trainer uses — presences from the
    thinned table, background from cells where *other* listed trees were
    recorded — reshaped from per-species point sets into the single
    cells x species count matrix DeepMaxent's loss expects.
    """

    def __init__(self, occ, predictors, species_order, use_record_counts=False):
        coords = occ[["latitude", "longitude"]].to_numpy(dtype=np.float64)
        rows, cols = predictors._rowcol(coords)
        on_grid = (
            (rows >= 0)
            & (rows < predictors.height)
            & (cols >= 0)
            & (cols < predictors.width)
        )
        rows, cols = rows[on_grid], cols[on_grid]
        occ = occ.loc[on_grid].reset_index(drop=True)

        keys = rows.astype(np.int64) * predictors.width + cols.astype(np.int64)
        uniq, inverse = np.unique(keys, return_inverse=True)
        cell_rows = (uniq // predictors.width).astype(np.int64)
        cell_cols = (uniq % predictors.width).astype(np.int64)

        features = predictors.stack[:, cell_rows, cell_cols].T
        complete = np.isfinite(features).all(axis=1)
        n_dropped = int((~complete).sum())

        remap = np.full(len(uniq), -1, dtype=np.int64)
        remap[complete] = np.arange(int(complete.sum()), dtype=np.int64)
        cell_of_record = remap[inverse]
        kept_record = cell_of_record >= 0

        self.n_dropped_cells = n_dropped
        self.names = list(predictors.names)
        self.species = list(species_order)
        self.X = np.ascontiguousarray(features[complete], dtype=np.float32)

        lon, lat = rasterio.transform.xy(
            predictors.transform, cell_rows[complete], cell_cols[complete]
        )
        self.coords = np.column_stack(
            [np.asarray(lat, dtype=np.float64), np.asarray(lon, dtype=np.float64)]
        )

        column_of = {name: i for i, name in enumerate(self.species)}
        species_column = (
            occ["species"].map(column_of).to_numpy(dtype=np.float64)
        )
        scored = kept_record & np.isfinite(species_column)
        if use_record_counts and "n_records" in occ.columns:
            weight = occ["n_records"].to_numpy(dtype=np.float32)
        else:
            weight = np.ones(len(occ), dtype=np.float32)

        self.Y = np.zeros((len(self.X), len(self.species)), dtype=np.float32)
        self.Y[
            cell_of_record[scored], species_column[scored].astype(np.int64)
        ] = weight[scored]

        self.presence = self.Y > 0
        self.n_presence = self.presence.sum(axis=0).astype(np.int64)

    @property
    def n_cells(self):
        return len(self.X)


def train_network(
    X,
    Y,
    *,
    device,
    hidden_size,
    hidden_nbr,
    learning_rate,
    weight_decay,
    batch_size,
    epochs,
    loss_option,
    seed,
    label,
    log_every,
):
    """Upstream train_deepmodel: Adam, keep the best-loss epoch. Batching runs on
    tensors already resident on the GPU instead of through a DataLoader —
    268k x 61 floats fit easily and the copies dominated the step otherwise."""
    import torch
    import torch.optim as optim

    from deepmaxent import deepmaxent_model

    set_seed(seed)
    device = torch.device(device)
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    Yt = torch.as_tensor(Y, dtype=torch.float32, device=device)

    model = deepmaxent_model(X.shape[1], hidden_size, Y.shape[1], hidden_nbr)
    model = model.to(device)
    criterion = make_criterion(loss_option).to(device)
    optimizer = optim.Adam(
        [
            {
                "params": model.parameters(),
                "lr": learning_rate,
                "weight_decay": weight_decay,
            }
        ]
    )

    n_cells = Xt.shape[0]
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    best_loss = float("inf")
    best_state = None
    best_epoch = -1
    history = []
    started = time.time()

    for epoch in range(epochs):
        model.train()
        order = torch.randperm(n_cells, device=device, generator=generator)
        running = torch.zeros((), device=device)
        n_batches = 0
        for start in range(0, n_cells, batch_size):
            index = order[start : start + batch_size]
            if index.numel() < MIN_BATCH_CELLS:
                continue
            optimizer.zero_grad(set_to_none=True)
            outputs = model(Xt[index])
            loss = criterion(outputs, Yt[index])
            loss.backward()
            optimizer.step()
            running += loss.detach()
            n_batches += 1

        epoch_loss = float(running.item() / max(n_batches, 1))
        history.append(epoch_loss)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
        if log_every and ((epoch + 1) % log_every == 0 or epoch == 0):
            print(
                f"    {label}  epoch {epoch + 1:4d}/{epochs}  "
                f"loss = {epoch_loss:.6f}  best = {best_loss:.6f}  "
                f"[{time.time() - started:.0f}s]",
                flush=True,
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, dict(
        best_loss=best_loss,
        best_epoch=int(best_epoch),
        seconds=float(time.time() - started),
        history=history,
    )


def log_intensity(model, X, device, chunk=200_000):
    import torch

    device = torch.device(device)
    out = np.empty((len(X), model.fc3_lambda.out_features), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(X), chunk):
            block = torch.as_tensor(
                X[start : start + chunk], dtype=torch.float32, device=device
            )
            out[start : start + chunk] = model(block).cpu().numpy()
    return out


def log_normaliser(lam):
    """log_z_s = log mean_cells exp(lambda_s), via logsumexp so a hot species
    cannot overflow. Fixes the free constant the maxent loss leaves open."""
    from scipy.special import logsumexp

    return logsumexp(lam, axis=0) - np.log(lam.shape[0])


def fold_auc(lam_val, presence_val):
    """Per-species AUC on a held-out spatial block: cells holding the species
    against the target-group cells that do not. AUC ignores prevalence, so it
    reads on the same scale as the XGBoost trainer's balanced-sample AUC."""
    n_species = presence_val.shape[1]
    scores = np.full(n_species, np.nan, dtype=np.float64)
    for s in range(n_species):
        truth = presence_val[:, s]
        n_pos = int(truth.sum())
        if n_pos == 0 or n_pos == len(truth):
            continue
        scores[s] = roc_auc_score(truth, lam_val[:, s])
    return scores


def cross_validate(table, args):
    """Spatial-block CV over the cell table, mirroring the XGBoost trainer's
    KMeans blocks and fold count."""
    blocks = spatial_block_ids(
        table.coords, n_blocks=N_SPATIAL_BLOCKS, random_state=SEED
    )
    n_blocks = len(np.unique(blocks))
    n_splits = int(min(args.folds, n_blocks))
    if n_splits < 2:
        print("Not enough spatial blocks for CV; every species will be ungated.")
        return np.zeros((0, len(table.species))), 0

    splits = list(GroupKFold(n_splits=n_splits).split(table.X, groups=blocks))
    print(
        f"\nSpatial-block CV: {n_splits} folds over {n_blocks} KMeans blocks "
        f"({table.n_cells:,} cells)"
    )
    per_fold = np.full((len(splits), len(table.species)), np.nan, dtype=np.float64)

    for i, (train_idx, val_idx) in enumerate(splits, start=1):
        scaler = StandardScaler().fit(table.X[train_idx])
        X_train = scaler.transform(table.X[train_idx]).astype(np.float32)
        X_val = scaler.transform(table.X[val_idx]).astype(np.float32)
        print(
            f"  fold {i}/{len(splits)}  train {len(train_idx):,} cells  "
            f"val {len(val_idx):,} cells",
            flush=True,
        )
        model, info = train_network(
            X_train,
            table.Y[train_idx],
            device=args.device,
            hidden_size=args.hidden_size,
            hidden_nbr=args.hidden_nbr,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            epochs=args.epochs,
            loss_option=args.loss,
            seed=SEED + i,
            label=f"fold {i}",
            log_every=args.log_every,
        )
        lam_val = log_intensity(model, X_val, args.device)
        per_fold[i - 1] = fold_auc(lam_val, table.presence[val_idx])
        usable = int(np.isfinite(per_fold[i - 1]).sum())
        mean_auc = float(np.nanmean(per_fold[i - 1])) if usable else float("nan")
        print(
            f"  fold {i}/{len(splits)}  mean AUC = {mean_auc:.3f} over "
            f"{usable} species  [{info['seconds']:.0f}s]",
            flush=True,
        )
        del model

    return per_fold, len(splits)


def unique_cells_per_species(occ):
    """Thinning already collapses a species to one row per 1 km cell, but a
    hand-made --occurrences file may not have."""
    if "grid_row" in occ.columns and "grid_col" in occ.columns:
        occ = occ.drop_duplicates(["species", "grid_row", "grid_col"])
    return occ.groupby("species").size()


def build_metrics(
    table,
    counts,
    cells_per_species,
    species_order,
    per_fold,
    n_folds,
    checkpoint_path,
    n_epochs,
):
    trained = {name: i for i, name in enumerate(table.species)}
    stored_path = relpath(checkpoint_path)

    rows = []
    tally = {
        "saved": 0,
        "skip_no_records": 0,
        "skip_few_cells": 0,
        "skip_no_folds": 0,
        "skip_few_folds": 0,
    }
    for species in species_order:
        row = dict(
            species=species,
            n_records=int(counts.get(species, 0)),
            n_unique_cells=0,
            n_background_cells=0,
            auc=np.nan,
            n_epochs=0,
            n_folds_usable=0,
            n_folds=n_folds,
            model_path="",
            status="skip_no_records",
        )
        if species in trained:
            column = trained[species]
            n_cells = int(table.n_presence[column])
            row["n_unique_cells"] = n_cells
            row["n_background_cells"] = int(table.n_cells - n_cells)
            row["n_epochs"] = int(n_epochs)
            if per_fold.size:
                scores = per_fold[:, column]
                usable = int(np.isfinite(scores).sum())
                row["n_folds_usable"] = usable
                if usable:
                    row["auc"] = float(np.nanmean(scores))
                if usable == 0:
                    row["status"] = "skip_no_folds"
                elif usable < MIN_USABLE_FOLDS:
                    row["status"] = "skip_few_folds"
                else:
                    row["status"] = "saved"
                    row["model_path"] = stored_path
            else:
                row["status"] = "saved"
                row["model_path"] = stored_path
        elif int(cells_per_species.get(species, 0)) > 0:
            row["status"] = "skip_few_cells"
            row["n_unique_cells"] = int(cells_per_species.get(species, 0))
        tally[row["status"]] = tally.get(row["status"], 0) + 1
        rows.append(row)
    return pd.DataFrame(rows), tally


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Train the USA 1 km DeepMaxent Deep SDM on the same data as "
            "xgboost_training_usa_30s.py (leaves the boosters alone)."
        )
    )
    parser.add_argument("--occurrences", default=THINNED_CSV)
    parser.add_argument("--species-list", default=SEED_LIST)
    parser.add_argument(
        "--full-list",
        action="store_true",
        help=f"use {FULL_LIST} instead of the seed list",
    )
    parser.add_argument(
        "--all-species-in-file",
        action="store_true",
        help="train every species present in the occurrence table",
    )
    parser.add_argument("--model-dir", default=MODEL_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE)
    parser.add_argument("--hidden-nbr", type=int, default=DEFAULT_HIDDEN_NBR)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--loss", default=DEFAULT_LOSS, choices=LOSS_CHOICES)
    parser.add_argument("--folds", type=int, default=N_SPATIAL_FOLDS)
    parser.add_argument(
        "--no-cv",
        action="store_true",
        help="skip spatial CV; every trained species is written ungated",
    )
    parser.add_argument(
        "--use-record-counts",
        action="store_true",
        help=(
            "weight each cell by n_records instead of presence/absence; "
            "reintroduces the survey effort the 1 km thinning removed"
        ),
    )
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="seed list, 3 folds, 5 epochs — checks the wiring, not the science",
    )
    args = parser.parse_args(argv)
    if args.smoke:
        args.epochs = min(args.epochs, 5)
        # MIN_USABLE_FOLDS is 3, so fewer folds could only ever write skips.
        args.folds = max(MIN_USABLE_FOLDS, 3)
        args.log_every = 1
    return args


def main(argv=None):
    args = parse_args(argv)
    species_list_path = FULL_LIST if args.full_list else args.species_list
    os.makedirs(args.model_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.model_dir, os.path.basename(CHECKPOINT_PATH))
    metrics_path = os.path.join(args.model_dir, os.path.basename(METRICS_PATH))
    features_path = os.path.join(args.model_dir, os.path.basename(FEATURE_NAMES_PATH))

    print("=" * 64)
    print("DeepMaxent Deep SDM — USA 1 km (30 arcsec)")
    print("=" * 64)
    print(f"Upstream model: {UPSTREAM['repo']} @ {UPSTREAM['commit'][:12]}")

    grid = read_grid()
    occ = load_occurrence_table(args.occurrences, grid)
    print(f"Occurrences:    {args.occurrences}")
    print(
        f"                {len(occ):,} thinned records, "
        f"{occ['species'].nunique()} species"
    )

    if args.all_species_in_file:
        requested = list(occ["species"].drop_duplicates())
        print(f"Species:        every name in the occurrence table ({len(requested)})")
    else:
        requested = load_species_list(species_list_path)
        print(f"Species list:   {species_list_path}  ({len(requested)} names)")

    predictors = Usa30sPredictors(collect_predictor_paths())
    print(f"Predictors:     {len(predictors.names)} layers (same order as XGBoost)")

    cells_per_species = unique_cells_per_species(occ)
    record_counts = (
        occ.groupby("species")["n_records"].sum()
        if "n_records" in occ.columns
        else cells_per_species
    )
    trainable = [
        name
        for name in requested
        if int(cells_per_species.get(name, 0)) >= MIN_UNIQUE_CELLS
    ]
    dropped = len(requested) - len(trainable)
    if not trainable:
        raise SystemExit(
            f"No requested species reaches {MIN_UNIQUE_CELLS} thinned 1 km cells."
        )
    print(
        f"Outputs:        {len(trainable)} species with >= {MIN_UNIQUE_CELLS} "
        f"cells ({dropped} below the gate)"
    )

    table = CellTable(
        occ, predictors, trainable, use_record_counts=args.use_record_counts
    )
    del predictors
    print(
        f"Cell table:     {table.n_cells:,} target-group cells x "
        f"{len(table.species)} species"
        + (
            f"  ({table.n_dropped_cells:,} dropped for a missing layer)"
            if table.n_dropped_cells
            else ""
        )
    )
    print(
        "Targets:        "
        + (
            "n_records per cell"
            if args.use_record_counts
            else "presence/absence per cell (matches the 1 km thinning)"
        )
    )
    print(
        f"Hyperparams:    hidden {args.hidden_size} x {args.hidden_nbr}  "
        f"lr {args.learning_rate}  wd {args.weight_decay}  "
        f"batch {args.batch_size}  epochs {args.epochs}  loss {args.loss}"
    )
    print(f"Device:         {args.device}")

    if args.no_cv:
        per_fold, n_folds = np.zeros((0, len(table.species))), 0
    else:
        per_fold, n_folds = cross_validate(table, args)

    print("\nFinal fit on every cell …")
    scaler = StandardScaler().fit(table.X)
    X_all = scaler.transform(table.X).astype(np.float32)
    model, info = train_network(
        X_all,
        table.Y,
        device=args.device,
        hidden_size=args.hidden_size,
        hidden_nbr=args.hidden_nbr,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        loss_option=args.loss,
        seed=SEED,
        label="final",
        log_every=args.log_every,
    )
    lam_all = log_intensity(model, X_all, args.device)
    log_z = log_normaliser(lam_all)

    save_checkpoint(
        checkpoint_path,
        state_dict=model.state_dict(),
        species=table.species,
        feature_names=table.names,
        hidden_size=args.hidden_size,
        hidden_nbr=args.hidden_nbr,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        log_z=log_z,
        training=dict(
            loss=args.loss,
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            learning_rate=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
            hidden_size=int(args.hidden_size),
            hidden_nbr=int(args.hidden_nbr),
            n_cells=int(table.n_cells),
            n_species=int(len(table.species)),
            best_epoch=int(info["best_epoch"]),
            best_loss=float(info["best_loss"]),
            seconds=float(info["seconds"]),
            occurrences=relpath(args.occurrences),
            species_list=(
                "occurrence-table"
                if args.all_species_in_file
                else relpath(species_list_path)
            ),
            target="n_records" if args.use_record_counts else "presence",
            seed=SEED,
        ),
    )

    with open(features_path, "w") as handle:
        handle.write("\n".join(table.names) + "\n")

    metrics, tally = build_metrics(
        table,
        record_counts,
        cells_per_species,
        requested,
        per_fold,
        n_folds,
        checkpoint_path,
        n_epochs=info["best_epoch"] + 1,
    )
    metrics.to_csv(metrics_path, index=False)

    saved = metrics[metrics["status"] == "saved"]
    mean_auc = float(saved["auc"].mean()) if len(saved) else float("nan")
    print("\n" + "=" * 64)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Metrics:    {metrics_path}")
    print(
        f"Saved {len(saved)} of {len(metrics)} species  mean AUC = {mean_auc:.3f}  "
        f"(final fit {info['seconds']:.0f}s, best epoch {info['best_epoch'] + 1})"
    )
    for status, n in sorted(tally.items()):
        if n and status != "saved":
            print(f"  {status:18s} {n}")
    print(
        "\nXGBoost boosters in data/models_usa_30s/ were not touched.\n"
        "Score a pin with:\n"
        "  python recommend_usa_30s.py --model deepmaxent --lat 30.2672 "
        "--lon -97.7431"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
