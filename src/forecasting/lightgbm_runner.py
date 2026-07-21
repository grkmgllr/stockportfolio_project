"""LightGBM forecaster: train and evaluate one ticker's price predictions."""
from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np

from models.LightGBMForecaster import LightGBMForecaster
from paths import CHECKPOINTS_ROOT, forecaster_checkpoint
from utils import calculate_return_metrics

from forecasting.data_loading import load_raw_df


def train_one(ticker: str, data_root: str, ma_targets: List[str],
              seq_len: int, pred_len: int,
              patience: int = 30) -> LightGBMForecaster:
    """Fit a LightGBM forecaster for a single ticker and save the checkpoint."""
    df, train_end, val_end, target_features = load_raw_df(
        ticker, data_root, ma_targets,
    )
    df_train = df.iloc[:train_end].copy()
    df_val = df.iloc[train_end:val_end].copy()

    forecaster = LightGBMForecaster(seq_len=seq_len, pred_len=pred_len)
    forecaster.fit(
        df_train, df_val,
        target_features=target_features,
        early_stopping_rounds=patience * 5,
    )

    os.makedirs(CHECKPOINTS_ROOT, exist_ok=True)
    checkpoint_path = forecaster_checkpoint(ticker, "LightGBM")
    forecaster.save(checkpoint_path)

    print("\nTop-10 Feature Importance (gain):")
    for i, (name, score) in enumerate(forecaster.feature_importance().items()):
        if i >= 10:
            break
        print(f"  {name:25s}: {score:.1f}")

    return forecaster


def train_pooled(tickers: List[str], data_root: str, ma_targets: List[str],
                 seq_len: int, pred_len: int,
                 patience: int = 30) -> LightGBMForecaster:
    """Train one LightGBM model on many tickers without cross-ticker leakage.

    Each ticker's (X, y) arrays are built inside its own frame — feature
    engineering (rolling / ewm / lag / shift) never crosses a ticker
    boundary — and only the resulting matrices are pooled.
    """
    print(f"\nPooled LightGBM training: {', '.join(tickers)}")

    train_dfs, val_dfs = [], []
    target_features = None
    for t in tickers:
        df, train_end, val_end, tf = load_raw_df(t, data_root, ma_targets)
        train_dfs.append(df.iloc[:train_end].copy())
        val_dfs.append(df.iloc[train_end:val_end].copy())
        target_features = target_features or tf
        print(f"  {t}: train={train_end}, val={val_end - train_end}")

    forecaster = LightGBMForecaster(seq_len=seq_len, pred_len=pred_len)
    forecaster.fit_pooled(
        train_dfs=train_dfs, val_dfs=val_dfs,
        target_features=target_features,
        early_stopping_rounds=patience * 5,
        ticker_labels=tickers,
    )

    os.makedirs(CHECKPOINTS_ROOT, exist_ok=True)
    checkpoint_path = os.path.join(CHECKPOINTS_ROOT, "pooled_LightGBM_best.joblib")
    forecaster.save(checkpoint_path)
    print(f"\nCheckpoint: {checkpoint_path}")

    return forecaster


def evaluate(ticker: str, data_root: str, ma_targets: List[str],
             seq_len: int, pred_len: int,
             checkpoint_override: str | None = None
             ) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[str, dict]]:
    """Load a saved forecaster and predict on the test window.

    Slicing matches what generate_meta_labels.py expects:
        preds_full[j] corresponds to df row (seq_len + j).
        Test sample i has entry bar (val_end + i + seq_len - 1),
        so j = val_end + i - 1; test predictions start at j = val_end - 1.

    Returns ``(preds, trues, target_features, eval_results)`` — mirroring
    ``pytorch_runner.evaluate``. ``eval_results`` holds the return-based
    IC / RIC / DA metrics (overall and per target), reconstructed from the
    per-sample anchor Close so they match the return-space the model fits on.
    """
    df, _train_end, val_end, _target_features = load_raw_df(
        ticker, data_root, ma_targets,
    )

    checkpoint_path = checkpoint_override or forecaster_checkpoint(ticker, "LightGBM")
    forecaster = LightGBMForecaster.load(checkpoint_path)

    preds_full = forecaster.predict(df)
    trues_full = forecaster.get_ground_truth(df)

    n_test = (len(df) - val_end) - forecaster.seq_len - forecaster.pred_len + 1
    test_start = val_end - 1
    preds = preds_full[test_start:test_start + n_test]
    trues = trues_full[test_start:test_start + n_test]

    # Reconstruct returns relative to each sample's anchor Close — the exact
    # alignment predict() uses: anchor = Close[seq_len : len(df) - pred_len].
    anchor_full = df["Close"].values[
        forecaster.seq_len: len(df) - forecaster.pred_len
    ]
    anchor = anchor_full[test_start:test_start + n_test][:, None, None]
    pred_returns = (preds - anchor) / anchor
    true_returns = (trues - anchor) / anchor

    target_features = forecaster.target_features
    eval_results = {"overall_returns": calculate_return_metrics(pred_returns, true_returns)}
    for i, name in enumerate(target_features):
        eval_results[f"{name}_returns"] = calculate_return_metrics(
            pred_returns[:, :, i], true_returns[:, :, i],
        )

    return preds, trues, target_features, eval_results
