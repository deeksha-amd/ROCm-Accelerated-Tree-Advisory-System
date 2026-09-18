#!/usr/bin/env python3
"""Replay one SDM-sized XGBoost train (no rasters) for rocprofv3.

Matches xgboost_training_usa_30s.py: QuantileDMatrix once, fold matrices with
ref= cuts, xgb.train, then a final fit. Default matrix is a typical oak
(~5k x 61).
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import xgboost as xgb

sys.path.insert(0, "/home/AMD/dgoplani/hackathon_2026")
from xgboost_training_usa_30s import (  # noqa: E402
    EARLY_STOPPING_ROUNDS,
    MAX_BOOST_ROUNDS,
    _fold_train_eval,
    _train_booster,
    _trees_used,
    make_binned_dmatrix,
)


def make_xy(n_rows, n_cols, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_rows, n_cols)).astype(np.float32)
    # Mildly separable so early-stop behaves like a real SDM, not 500 rounds.
    z = X[:, :8].sum(axis=1)
    p = 1.0 / (1.0 + np.exp(-0.35 * z))
    y = (rng.random(n_rows) < p).astype(np.float32)
    return X, y


def kfold_indices(n, n_folds, seed):
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    return np.array_split(order, n_folds)


def run_sdm_pattern(X, y, device, n_folds):
    binned = make_binned_dmatrix(X, y, device)
    folds = kfold_indices(len(y), n_folds, seed=0)
    idx = np.arange(len(y))
    best_ntrees = []
    t0 = time.perf_counter()
    for i, test_idx in enumerate(folds):
        train_idx = np.setdiff1d(idx, test_idx, assume_unique=False)
        dtrain, dtest = _fold_train_eval(
            binned, train_idx, test_idx, X, y, device,
        )
        bst = _train_booster(
            dtrain,
            n_estimators=MAX_BOOST_ROUNDS,
            n_samples=len(y),
            device=device,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            dval=dtest,
        )
        trees = _trees_used(bst, MAX_BOOST_ROUNDS)
        best_ntrees.append(trees)
        print(f"  fold {i + 1}/{n_folds}  trees={trees}", flush=True)
    n_trees = max(int(np.median(best_ntrees)), 10)
    _train_booster(
        binned,
        n_estimators=n_trees,
        n_samples=len(y),
        device=device,
        early_stopping_rounds=None,
    )
    elapsed = time.perf_counter() - t0
    print(
        f"  final trees={n_trees}  fit={elapsed:.2f}s  "
        f"device={device}  rows={len(y)} cols={X.shape[1]}",
        flush=True,
    )
    return elapsed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    p.add_argument("--n-rows", type=int, default=5110)
    p.add_argument("--n-cols", type=int, default=61)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-warmup", action="store_true")
    args = p.parse_args()

    print(
        f"xgboost {xgb.__version__}  device={args.device}  "
        f"{args.n_rows}x{args.n_cols}  folds={args.n_folds}",
        flush=True,
    )
    X, y = make_xy(args.n_rows, args.n_cols, args.seed)
    if not args.skip_warmup:
        t0 = time.perf_counter()
        warm = make_binned_dmatrix(X[:800], y[:800], args.device)
        _train_booster(
            warm,
            n_estimators=8,
            n_samples=800,
            device=args.device,
            early_stopping_rounds=None,
        )
        print(f"warmup {time.perf_counter() - t0:.2f}s", flush=True)
    print("PROFILE_START", flush=True)
    run_sdm_pattern(X, y, args.device, args.n_folds)
    print("PROFILE_END", flush=True)


if __name__ == "__main__":
    main()
