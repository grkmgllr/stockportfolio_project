"""Raw-CSV loading shared by LightGBM and PyTorch trainers/evaluators."""
from __future__ import annotations

import os
from typing import List

import pandas as pd

from dataset import StockDataset


def load_raw_df(ticker: str, data_root: str, ma_targets: List[str],
                train_ratio: float = 0.7, val_ratio: float = 0.15,
                start_date: str | None = None,
                end_date: str | None = None,
                train_end_date: str | None = None,
                val_end_date: str | None = None):
    """Load CSV, compute MA columns, trim warm-up, and split.

    Returns (df, train_end, val_end, target_features) — where target_features
    is ["High", "Close", *ma_targets_not_already_in_High_Close].

    ``start_date`` / ``end_date`` bound the date range (applied before the MA
    columns are computed), mirroring StockDataset so LightGBM sees the same
    window as the neural models. Pass None to use the whole CSV.

    When both ``train_end_date`` and ``val_end_date`` are given they replace
    train_ratio/val_ratio with explicit date-based boundaries — this is what
    walk-forward evaluation uses.
    """
    csv_path = os.path.join(data_root, f"{ticker}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Data file not found: {csv_path}\n"
            f"Run: python src/scripts/fetch_data.py first."
        )

    df = pd.read_csv(csv_path)
    if start_date and "Date" in df.columns:
        df = df[df["Date"] >= start_date].reset_index(drop=True)
    if end_date and "Date" in df.columns:
        df = df[df["Date"] <= end_date].reset_index(drop=True)
    df = df.ffill().bfill()

    for ma_name in ma_targets:
        cfg = StockDataset.MA_CONFIGS[ma_name]
        df[ma_name] = StockDataset._compute_ma(
            df["Close"], cfg["method"], cfg["period"],
        )

    if ma_targets:
        max_period = max(
            StockDataset.MA_CONFIGS[n]["period"] for n in ma_targets
        )
        df = df.iloc[max_period - 1:].reset_index(drop=True)

    total = len(df)
    if train_end_date and val_end_date and "Date" in df.columns:
        train_end = int((df["Date"] <= train_end_date).sum())
        val_end = int((df["Date"] <= val_end_date).sum())
    else:
        train_end = int(total * train_ratio)
        val_end = int(total * (train_ratio + val_ratio))

    target_features = ["High", "Close"] + [
        n for n in ma_targets if n not in ("High", "Close")
    ]

    return df, train_end, val_end, target_features
