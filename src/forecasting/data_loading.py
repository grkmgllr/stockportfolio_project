"""Raw-CSV loading shared by LightGBM and PyTorch trainers/evaluators."""

import os
from typing import List

import pandas as pd

from dataset import ParquetDataset


def load_raw_df(ticker: str, data_root: str, ma_targets: List[str],
                train_ratio: float = 0.7, val_ratio: float = 0.15):
    """Load CSV, compute MA columns, trim warm-up, and split.

    Returns (df, train_end, val_end, target_features) — where target_features
    is ["High", "Close", *ma_targets_not_already_in_High_Close].
    """
    csv_path = os.path.join(data_root, f"{ticker}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Data file not found: {csv_path}\n"
            f"Run resample_parquet.py or fetch_data.py first."
        )

    df = pd.read_csv(csv_path).ffill().bfill()

    for ma_name in ma_targets:
        cfg = ParquetDataset.MA_CONFIGS[ma_name]
        df[ma_name] = ParquetDataset._compute_ma(
            df["Close"], cfg["method"], cfg["period"],
        )

    if ma_targets:
        max_period = max(
            ParquetDataset.MA_CONFIGS[n]["period"] for n in ma_targets
        )
        df = df.iloc[max_period - 1:].reset_index(drop=True)

    total = len(df)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))

    target_features = ["High", "Close"] + [
        n for n in ma_targets if n not in ("High", "Close")
    ]

    return df, train_end, val_end, target_features
