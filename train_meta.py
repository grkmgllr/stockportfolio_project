"""
Training script for the LightGBM meta-classifier.

Loads the meta-label CSV produced by ``generate_meta_labels.py``, trains
a LightGBM binary classifier using Purged K-Fold cross-validation, and
saves the trained model for later inference.

The meta-classifier learns to predict the *probability* that a signal
from the primary model (TimeMixer) will result in a profitable trade.
It is trained on market-context features (ATR, RSI, MACD, volatility)
and prediction-derived features (pred_return, pred_close_return).

Usage
-----
    python train_meta.py --ticker AAPL
    python train_meta.py --ticker AAPL --n_splits 5 --n_embargo 5
    python train_meta.py --ticker AAPL --threshold 0.6
"""

import pandas as pd
import numpy as np
import os
import argparse

from models.meta_classifier.lightgbm_model import MetaClassifier


# Features the meta-classifier is trained on.
# These are the market-context and prediction-derived columns
# produced by generate_meta_labels.py.
FEATURE_COLUMNS = [
    "atr",
    "rolling_vol",
    "rsi",
    "macd_line",
    "macd_signal",
    "macd_hist",
    "pred_return",
    "pred_close_return",
]

# Additional features included when available in the CSV
# (present when the data was resampled from minute-bar parquet).
OPTIONAL_FEATURE_COLUMNS = [
    "Vwap",
    "Transactions",
    "daily_vol",
]

TARGET_COLUMN = "meta_label"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train LightGBM meta-classifier on meta-labeled data."
    )
    parser.add_argument("--ticker", type=str, default="AAPL")
    parser.add_argument("--meta_dir", type=str, default="data/meta",
                        help="Directory containing meta_labels_{ticker}.csv")
    parser.add_argument("--output_dir", type=str, default="checkpoints/meta",
                        help="Directory to save the trained model")
    parser.add_argument("--n_splits", type=int, default=5,
                        help="Number of PurgedKFold splits")
    parser.add_argument("--n_embargo", type=int, default=5,
                        help="Embargo bars after each test fold")
    parser.add_argument("--early_stopping", type=int, default=50,
                        help="Early stopping rounds per fold")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Classification threshold for summary stats")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── 1. Load meta-label CSV ──
    csv_path = os.path.join(args.meta_dir, f"meta_labels_{args.ticker}.csv")
    if not os.path.exists(csv_path):
        print(f"Error: Meta-label file not found: {csv_path}")
        print(f"Run: python scripts/generate_meta_labels.py --ticker {args.ticker} first.")
        return

    df = pd.read_csv(csv_path)
    print(f"\nLoaded {len(df)} rows from {csv_path}")

    # ── 2. Resolve feature columns ──
    feature_cols = FEATURE_COLUMNS.copy()
    for col in OPTIONAL_FEATURE_COLUMNS:
        if col in df.columns:
            feature_cols.append(col)

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"Warning: Missing feature columns (skipping): {missing}")
        feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].values.astype(np.float64)
    y = df[TARGET_COLUMN].values.astype(np.int32)
    t_start = df["t_start"].values.astype(np.int64)
    t_end = df["t_end"].values.astype(np.int64)

    print(f"Features ({len(feature_cols)}): {feature_cols}")
    print(f"Target distribution: {int(y.sum())} positive / {int(len(y) - y.sum())} negative "
          f"({100 * y.mean():.1f}% positive rate)")

    # ── 3. Train ──
    clf = MetaClassifier()
    clf.fit(
        X, y,
        t_start=t_start,
        t_end=t_end,
        n_splits=args.n_splits,
        n_embargo=args.n_embargo,
        feature_names=feature_cols,
        early_stopping_rounds=args.early_stopping,
    )

    # ── 4. Feature importance ──
    print("Feature Importance (gain):")
    print("-" * 40)
    for name, score in clf.feature_importance("gain").items():
        print(f"  {name:25s}: {score:.1f}")

    # ── 5. Full-dataset predictions (for evaluation) ──
    proba = clf.predict_proba(X)
    preds = (proba >= args.threshold).astype(int)

    n_passed = preds.sum()
    precision_raw = y.mean()
    precision_filtered = y[preds == 1].mean() if n_passed > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"Quick Evaluation (threshold={args.threshold})")
    print(f"{'='*60}")
    print(f"  Total signals:        {len(y)}")
    print(f"  Signals after filter: {n_passed} ({100 * n_passed / len(y):.1f}%)")
    print(f"  Precision (baseline): {precision_raw:.4f}")
    print(f"  Precision (filtered): {precision_filtered:.4f}")
    print(f"{'='*60}")

    # ── 6. Save ──
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, f"{args.ticker}_meta_clf.joblib")
    clf.save(model_path)

    # Also save the predictions for evaluation in Phase 6
    df["meta_proba"] = proba
    df["meta_pred"] = preds
    eval_path = os.path.join(args.meta_dir, f"meta_predictions_{args.ticker}.csv")
    df.to_csv(eval_path, index=False)
    print(f"Predictions saved to {eval_path}")


if __name__ == "__main__":
    main()
