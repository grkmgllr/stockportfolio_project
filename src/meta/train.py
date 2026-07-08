"""
Train the LightGBM meta-classifier with Purged K-Fold CV.

Reads ``meta_labels_{ticker}_{model}.csv`` produced by meta.generate,
trains, and writes both the classifier joblib and a per-row prediction CSV.
"""

import argparse
import os

import numpy as np
import pandas as pd

from models.meta_classifier.lightgbm_model import MetaClassifier
from paths import (
    META_CHECKPOINTS_ROOT,
    META_ROOT,
    meta_classifier_checkpoint,
    meta_labels_path,
    meta_predictions_path,
)


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

# Present only when the CSV came from minute-bar resample or with MA targets.
OPTIONAL_FEATURE_COLUMNS = [
    "Vwap",
    "Transactions",
    "daily_vol",
    "pred_ema20_vs_close",
    "pred_sma50_vs_close",
    "pred_ma_crossover",
]

TARGET_COLUMN = "meta_label"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train LightGBM meta-classifier on meta-labeled data.",
    )
    parser.add_argument("--ticker", type=str, default="AAPL")
    parser.add_argument("--model", type=str, default="LightGBM",
                        choices=["LightGBM", "TimesNet", "TimeMixer"],
                        help="Primary forecaster whose meta-labels to train on")
    parser.add_argument("--meta_dir", type=str, default=META_ROOT,
                        help="Directory containing meta_labels_{ticker}_{model}.csv")
    parser.add_argument("--output_dir", type=str, default=META_CHECKPOINTS_ROOT)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--n_embargo", type=int, default=5)
    parser.add_argument("--early_stopping", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Classification threshold for summary stats")
    return parser.parse_args()


def main():
    args = parse_args()

    # honor --meta_dir but keep default filename convention centralized
    if args.meta_dir == META_ROOT:
        csv_path = meta_labels_path(args.ticker, args.model)
    else:
        csv_path = os.path.join(
            args.meta_dir, f"meta_labels_{args.ticker}_{args.model}.csv",
        )
    if not os.path.exists(csv_path):
        print(f"Error: Meta-label file not found: {csv_path}")
        print(f"Run: python main.py meta-label --ticker {args.ticker} "
              f"--model {args.model} first.")
        return

    df = pd.read_csv(csv_path)
    print(f"\nLoaded {len(df)} rows from {csv_path}")

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
    print(f"Target distribution: {int(y.sum())} positive / "
          f"{int(len(y) - y.sum())} negative "
          f"({100 * y.mean():.1f}% positive rate)")

    clf = MetaClassifier()
    clf.fit(
        X, y,
        t_start=t_start, t_end=t_end,
        n_splits=args.n_splits, n_embargo=args.n_embargo,
        feature_names=feature_cols,
        early_stopping_rounds=args.early_stopping,
    )

    print("Feature Importance (gain):")
    print("-" * 40)
    for name, score in clf.feature_importance("gain").items():
        print(f"  {name:25s}: {score:.1f}")

    proba = clf.predict_proba(X)
    preds = (proba >= args.threshold).astype(int)
    n_passed = int(preds.sum())
    precision_raw = float(y.mean())
    precision_filtered = float(y[preds == 1].mean()) if n_passed > 0 else 0.0

    print(f"\n{'=' * 60}")
    print(f"Quick Evaluation (threshold={args.threshold})")
    print(f"{'=' * 60}")
    print(f"  Total signals:        {len(y)}")
    print(f"  Signals after filter: {n_passed} ({100 * n_passed / len(y):.1f}%)")
    print(f"  Precision (baseline): {precision_raw:.4f}")
    print(f"  Precision (filtered): {precision_filtered:.4f}")
    print(f"{'=' * 60}")

    os.makedirs(args.output_dir, exist_ok=True)
    if args.output_dir == META_CHECKPOINTS_ROOT:
        model_path = meta_classifier_checkpoint(args.ticker, args.model)
    else:
        model_path = os.path.join(
            args.output_dir, f"{args.ticker}_{args.model}_meta_clf.joblib",
        )
    clf.save(model_path)

    df["meta_proba"] = proba
    df["meta_pred"] = preds
    if args.meta_dir == META_ROOT:
        eval_path = meta_predictions_path(args.ticker, args.model)
    else:
        eval_path = os.path.join(
            args.meta_dir, f"meta_predictions_{args.ticker}_{args.model}.csv",
        )
    df.to_csv(eval_path, index=False)
    print(f"Predictions saved to {eval_path}")


if __name__ == "__main__":
    main()
