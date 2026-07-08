"""
Feature engineering bridge between the primary model and the meta-classifier.

Steps:
1. Load ``.npy`` predictions produced by ``main.py test``.
2. Load the raw OHLCV CSV, align it to the test window.
3. Apply the Triple Barrier Method → binary ``meta_label``.
4. Attach market-context features (ATR, RSI, MACD, rolling vol).
5. Attach prediction-derived features (pred_return, pred_close_return,
   pred_ema20_vs_close, pred_sma50_vs_close, pred_ma_crossover).
6. Drop warm-up NaNs and save to ``data/meta/meta_labels_{ticker}_{model}.csv``.
"""

import argparse
import os

import numpy as np
import pandas as pd

from dataset import ParquetDataset
from paths import DATA_ROOT, META_ROOT, RESULTS_ROOT, meta_labels_path
from trading_logic.triple_barrier import apply_triple_barrier

from meta.features import add_market_context_features, add_prediction_features


def build_meta_dataset(
    ticker: str,
    model: str,
    data_root: str = DATA_ROOT,
    results_root: str = RESULTS_ROOT,
    seq_len: int = 14,
    pred_len: int = 5,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    vol_lookback: int = 20,
    sl_multiplier: float = 2.0,
    vertical_barrier_periods: int = 5,
    target_names=None,
) -> pd.DataFrame:
    """Build the training-ready DataFrame for one (ticker, model) pair."""
    if target_names is None:
        target_names = ["High", "Close"]
    target_idx = {name: i for i, name in enumerate(target_names)}
    if "High" not in target_idx or "Close" not in target_idx:
        raise ValueError(
            "target_names must contain at least 'High' and 'Close'. "
            f"Got: {target_names}"
        )
    has_ema20 = "EMA_20" in target_idx
    has_sma50 = "SMA_50" in target_idx

    # ── 1. Load raw OHLCV ──
    csv_path = os.path.join(data_root, f"{ticker}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Raw data not found: {csv_path}\n"
            f"Run: python main.py resample --ticker {ticker} first."
        )
    df_raw = pd.read_csv(csv_path).ffill().bfill()

    # When trained with MA targets, ParquetDataset trims warm-up rows before
    # splitting.  Replicate that trim so val_end aligns.
    ma_names_in_targets = [n for n in target_names if n in ("EMA_20", "SMA_50")]
    if ma_names_in_targets:
        ma_periods = [ParquetDataset.MA_CONFIGS[n]["period"]
                      for n in ma_names_in_targets]
        trim = max(ma_periods) - 1
        df_raw = df_raw.iloc[trim:].reset_index(drop=True)
        print(f"Trimmed {trim} MA warm-up rows to match dataset splits.")
    total_len = len(df_raw)

    # ── 2. Load predictions ──
    results_dir = os.path.join(results_root, ticker, model)
    pred_path = os.path.join(results_dir, "predictions.npy")
    true_path = os.path.join(results_dir, "ground_truth.npy")
    if not os.path.exists(pred_path) or not os.path.exists(true_path):
        raise FileNotFoundError(
            f"Prediction files not found in {results_dir}/.\n"
            f"Run: python main.py test --ticker {ticker} --model {model} first."
        )
    preds = np.load(pred_path)
    trues = np.load(true_path)  # noqa: F841 — kept for shape symmetry
    n_test = preds.shape[0]

    # ── 3. Align prediction indices with the raw CSV ──
    val_end = int(total_len * (train_ratio + val_ratio))
    entry_indices = np.arange(n_test) + val_end + seq_len - 1

    hi = target_idx["High"]
    ci = target_idx["Close"]
    pred_high = preds[:, :, hi].max(axis=1)
    pred_close = preds[:, -1, ci]

    df_entry = df_raw.iloc[entry_indices].reset_index(drop=True)
    df_entry["pred_high"] = pred_high
    df_entry["pred_close"] = pred_close
    if has_ema20:
        df_entry["pred_ema20"] = preds[:, -1, target_idx["EMA_20"]]
    if has_sma50:
        df_entry["pred_sma50"] = preds[:, -1, target_idx["SMA_50"]]

    # ── 4. Apply Triple Barrier ──
    df_labeled = apply_triple_barrier(
        df_entry,
        pred_high_col="pred_high",
        close_col="Close",
        high_col="High",
        low_col="Low",
        vol_lookback=vol_lookback,
        sl_multiplier=sl_multiplier,
        vertical_barrier_periods=vertical_barrier_periods,
    )

    # ── 5. Feature engineering ──
    df_labeled = add_market_context_features(df_labeled)
    df_labeled = add_prediction_features(
        df_labeled, has_ema20=has_ema20, has_sma50=has_sma50,
    )

    # ── 6. Drop warm-up NaNs ──
    n_before = len(df_labeled)
    df_labeled = df_labeled.dropna()          # keep OLD positional index
    survivors_old = df_labeled.index.values   # OLD positions that survived
    df_labeled = df_labeled.reset_index(drop=True)
    print(f"Dropped {n_before - len(df_labeled)} warm-up rows with NaN values.")

    # ── 7. Re-align event window indices to the reset-index space ──
    # Before this step, `t_start` / `t_end` still refer to positions in the
    # pre-dropna DataFrame.  PurgedKFold compares them against fold
    # boundaries that live in the post-dropna coordinate system, so mixing
    # the two silently defeats the purge and leaks training data.
    # See src/trading_logic/purged_cv.py:split.
    #
    # After `apply_triple_barrier`, `t_start[i]` always equals row i's own
    # position, so in the new frame it collapses to `arange(n_new)`.
    # `t_end` is remapped via searchsorted; if the exact end row was
    # dropped (rare — only prefix warm-up gets dropped in practice), we
    # snap to the next surviving row, or clamp to the last one.
    n_new = len(df_labeled)
    old_t_end = df_labeled["t_end"].values
    new_t_end = np.searchsorted(survivors_old, old_t_end, side="left")
    new_t_end = np.minimum(new_t_end, n_new - 1)

    df_labeled["t_start"] = np.arange(n_new, dtype=np.int64)
    df_labeled["t_end"] = new_t_end.astype(np.int64)

    return df_labeled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate meta-labels + features from primary predictions.",
    )
    parser.add_argument("--ticker", type=str, default="AAPL")
    parser.add_argument("--model", type=str, default="LightGBM",
                        choices=["LightGBM", "TimesNet", "TimeMixer"],
                        help="Primary forecaster whose predictions to read")
    parser.add_argument("--data_root", type=str, default=DATA_ROOT)
    parser.add_argument("--results_root", type=str, default=RESULTS_ROOT)
    parser.add_argument("--seq_len", type=int, default=30)
    parser.add_argument("--pred_len", type=int, default=5)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--vol_lookback", type=int, default=20)
    parser.add_argument("--sl_multiplier", type=float, default=2.0)
    parser.add_argument("--vertical_barrier", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default=META_ROOT)
    parser.add_argument("--target_names", nargs="*", default=None,
                        help="Ordered target names matching the .npy shape "
                             "(e.g. High Close EMA_20 SMA_50)")
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("Meta-Label Generation Pipeline")
    print("=" * 60)
    target_names = args.target_names or ["High", "Close"]

    print(f"Ticker:          {args.ticker}")
    print(f"Model:           {args.model}")
    print(f"Seq len:         {args.seq_len}")
    print(f"Pred len:        {args.pred_len}")
    print(f"Target names:    {target_names}")
    print(f"SL multiplier:   {args.sl_multiplier}x daily vol")
    print(f"Vertical barrier:{args.vertical_barrier} bars")
    print("=" * 60 + "\n")

    df = build_meta_dataset(
        ticker=args.ticker,
        model=args.model,
        data_root=args.data_root,
        results_root=args.results_root,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        vol_lookback=args.vol_lookback,
        sl_multiplier=args.sl_multiplier,
        vertical_barrier_periods=args.vertical_barrier,
        target_names=target_names,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    # honor --output_dir but keep default filename convention centralized
    if args.output_dir == META_ROOT:
        out_path = meta_labels_path(args.ticker, args.model)
    else:
        out_path = os.path.join(
            args.output_dir,
            f"meta_labels_{args.ticker}_{args.model}.csv",
        )
    df.to_csv(out_path, index=False)

    n_pos = int((df["meta_label"] == 1).sum())
    n_neg = int((df["meta_label"] == 0).sum())
    print(f"\nSaved {len(df)} rows to {out_path}")
    print(f"Positive labels: {n_pos} ({100 * n_pos / len(df):.1f}%)")
    print(f"Negative labels: {n_neg} ({100 * n_neg / len(df):.1f}%)")


if __name__ == "__main__":
    main()
