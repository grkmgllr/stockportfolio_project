"""
Normalization views over the canonical panel.

Adds three parallel column sets on top of the canonical panel so every
downstream adapter sees identical inputs, differing only in scaling:

    raw feature `f`         →  LightGBM (trees ignore scale)
    `f_ps`  (per-series z)  →  TimesNet, TimeMixer, LightGBM (optional)
    `f_cs`  (cross-sect z)  →  MASTER (needs same-day relative ranking)

Splits
------
Fixed date-based split, no ratios:

    train : 2010-01-04 → 2021-12-31   (~12 yrs, all regimes incl. COVID)
    val   : 2022-01-01 → 2023-06-30   (18 mo, bear + early AI rally)
    test  : 2023-07-01 → 2026-07-10   (~3 yrs, post-model-selection out-of-sample)

The two z-score views
---------------------
Per-series z-score:
    For each ticker, fit (mean, std) on that ticker's TRAIN rows only.
    Transform all splits with the same stats. Prevents leakage of future
    scale into val/test.

Cross-sectional z-score:
    For each (date, feature), compute (mean, std) across all tickers
    that were `available` on that date. Purely local to each date — no
    train fit needed. NaN'd for `available=False` cells.

Outputs
-------
    data/processed/views_panel.parquet
        Canonical columns + `split` + `{feature}_ps` + `{feature}_cs`.
    data/processed/views_stats.json
        Per-series (mean, std) per ticker per feature, fitted on train.
        Kept so the same transform can be re-applied to fresh data.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

CANONICAL_PATH = "data/processed/canonical_panel.parquet"
OUT_PATH = "data/processed/views_panel.parquet"
STATS_PATH = "data/processed/views_stats.json"

TRAIN_END = "2021-12-31"
VAL_END = "2023-06-30"

FEATURES = [
    "log_return",
    "daily_range",
    "intraday_move",
    "upper_wick",
    "lower_wick",
    "overnight_gap",
    "log_volume",
]


def add_split_column(panel: pd.DataFrame) -> pd.DataFrame:
    """Label each row as train / val / test by date."""
    date = pd.to_datetime(panel["date"])
    split = pd.Series("test", index=panel.index)
    split[date <= pd.Timestamp(TRAIN_END)] = "train"
    split[(date > pd.Timestamp(TRAIN_END)) & (date <= pd.Timestamp(VAL_END))] = "val"
    panel = panel.copy()
    panel["split"] = split
    return panel


def fit_per_series_stats(panel: pd.DataFrame, features: list) -> dict:
    """Compute (mean, std) per (ticker, feature) from train + available rows."""
    train_mask = (panel["split"] == "train") & panel["available"]
    train = panel.loc[train_mask, ["ticker", *features]]
    stats = {}
    for tkr, g in train.groupby("ticker", sort=False):
        stats[tkr] = {}
        for f in features:
            m = float(g[f].mean(skipna=True))
            s = float(g[f].std(skipna=True))
            if not np.isfinite(s) or s < 1e-12:
                s = 1.0  # degenerate series → passthrough
            stats[tkr][f] = {"mean": m, "std": s}
    return stats


def apply_per_series(panel: pd.DataFrame, features: list, stats: dict) -> pd.DataFrame:
    """Add `f_ps` columns using train-fitted per-ticker stats.

    Tickers absent from `stats` (IPO'd after train window) get NaN in
    `f_ps` — MASTER's cross-sectional view still covers them.
    """
    panel = panel.copy()
    for f in features:
        mean_lookup = {t: stats[t][f]["mean"] for t in stats}
        std_lookup = {t: stats[t][f]["std"] for t in stats}
        m = panel["ticker"].map(mean_lookup)
        s = panel["ticker"].map(std_lookup)
        panel[f"{f}_ps"] = (panel[f] - m) / s
    return panel


def apply_cross_sectional(panel: pd.DataFrame, features: list) -> pd.DataFrame:
    """
    Add `f_cs` columns: for each date, z-score across available tickers.

    Unavailable rows stay NaN. Days with <3 available tickers get NaN
    (degenerate — no meaningful cross-section).
    """
    panel = panel.copy()
    grp = panel.groupby("date", sort=False)
    for f in features:
        # Mask feature to NaN where unavailable so stats ignore them
        masked = panel[f].where(panel["available"])
        mean = grp[f].transform(lambda s, m=masked: m.loc[s.index].mean())
        std = grp[f].transform(lambda s, m=masked: m.loc[s.index].std())
        # Guard degenerate std → NaN (rare early dates with 1-2 tickers)
        z = (panel[f] - mean) / std.where(std > 1e-12)
        panel[f"{f}_cs"] = z.where(panel["available"])
    return panel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--canonical", default=CANONICAL_PATH)
    p.add_argument("--out", default=OUT_PATH)
    p.add_argument("--stats", default=STATS_PATH)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print(f"Loading canonical panel: {args.canonical}")
    panel = pd.read_parquet(args.canonical)
    print(f"  {len(panel):,} rows, {panel['ticker'].nunique()} tickers, "
          f"{panel['date'].nunique()} dates")

    panel = add_split_column(panel)
    split_counts = panel["split"].value_counts().to_dict()
    print(f"\nSplits: {split_counts}")

    print("Fitting per-series stats on train ...")
    stats = fit_per_series_stats(panel, FEATURES)
    print(f"  {len(stats)} tickers × {len(FEATURES)} features")

    print("Applying per-series z-score ...")
    panel = apply_per_series(panel, FEATURES, stats)

    print("Applying cross-sectional z-score ...")
    panel = apply_cross_sectional(panel, FEATURES)

    panel.to_parquet(args.out, index=False)
    with open(args.stats, "w") as f:
        json.dump(stats, f, indent=2)

    # Sanity
    print("\n" + "=" * 60)
    print(f"Saved views panel: {panel.shape}")
    print(f"  {args.out}")
    print(f"  {args.stats}")

    train_ps = panel.loc[
        (panel["split"] == "train") & panel["available"],
        [f"{f}_ps" for f in FEATURES],
    ]
    print("\nPer-series view — train stats (should be ~0 mean, ~1 std):")
    print(train_ps.describe().loc[["mean", "std"]].round(4))

    cs = panel.loc[panel["available"], [f"{f}_cs" for f in FEATURES]]
    print("\nCross-sectional view — all-time stats (mean ~0, std ~1 per day):")
    print(cs.describe().loc[["mean", "std"]].round(4))
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
