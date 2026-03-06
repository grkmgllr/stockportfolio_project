"""
Feature Engineering bridge between the primary model and the meta-classifier.

This script transforms the raw output of TimeMixer (predicted High / Close
/ optional MAs saved as ``.npy`` by ``test.py --save_predictions``) into a
training-ready CSV for the LightGBM meta-classifier.

Pipeline
--------
1. Load the ``.npy`` prediction and ground-truth arrays produced by ``test.py``.
2. Load the original OHLCV CSV and slice to the test period so that every
   prediction row can be paired with the actual market data it corresponds to.
3. Build a ``pred_high`` column from the model's forecast.  For multi-step
   horizons (``pred_len > 1``), the *maximum* predicted High across the
   horizon is used as the take-profit target.
4. Apply the Triple Barrier Method (Phase 2) to generate the binary
   ``meta_label`` column (1 = profitable, 0 = unprofitable).
5. Engineer market-context features that give the secondary model information
   about the *regime* at the time of the signal:
       - Volatility:  ATR, rolling standard deviation of returns.
       - Momentum:    RSI, MACD line, MACD signal, MACD histogram.
       - Trend (when MA targets present): predicted EMA_20 / SMA_50 vs
         current close, predicted MA crossover direction.
6. Drop rows with NaN (warm-up period of the rolling indicators) and save the
   combined DataFrame as ``data/meta/meta_labels_{ticker}.csv``.

Usage
-----
    python scripts/generate_meta_labels.py --ticker AAPL
    python scripts/generate_meta_labels.py --ticker AAPL --pred_len 5 --seq_len 30
    python scripts/generate_meta_labels.py --ticker AAPL --target_names High Close EMA_20 SMA_50
"""

import pandas as pd
import numpy as np
import os
import argparse
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from trading_logic.triple_barrier import apply_triple_barrier


# ─────────────────────────────────────────────────────────
# Technical-indicator helpers
# ─────────────────────────────────────────────────────────

def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14,) -> pd.Series:
    """
    Compute the Average True Range (ATR).

    ATR measures market volatility by decomposing the entire range of a bar.
    It is the rolling mean of the True Range, which accounts for overnight
    gaps by comparing the current high/low against the previous close.

    Args:
        high (pd.Series):
            Intraday high prices.
        low (pd.Series):
            Intraday low prices.
        close (pd.Series):
            Closing prices.
        period (int):
            Smoothing window length (default 14 bars).

    Returns:
        pd.Series:
            ATR values.  The first ``period`` entries will be NaN.
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute the Relative Strength Index (RSI).

    RSI is a bounded [0, 100] momentum oscillator.  Values above 70 are
    conventionally considered overbought; below 30, oversold.  For the
    meta-classifier this feature captures whether the primary model's
    signal arrived in an overextended market.

    Args:
        close (pd.Series):
            Closing prices.
        period (int):
            Look-back window (default 14 bars).

    Returns:
        pd.Series:
            RSI values in the range [0, 100].
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def compute_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """
    Compute the MACD indicator (line, signal, histogram).

    MACD captures trend momentum by measuring the gap between a fast and
    a slow exponential moving average of the closing price.  The signal
    line smooths the MACD itself, and the histogram shows the divergence
    between the two, which often leads price reversals.

    Args:
        close (pd.Series):
            Closing prices.
        fast (int):
            Fast EMA period (default 12).
        slow (int):
            Slow EMA period (default 26).
        signal (int):
            Signal-line EMA period (default 9).

    Returns:
        pd.DataFrame:
            Three columns: ``macd_line``, ``macd_signal``, ``macd_hist``.
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - macd_signal

    return pd.DataFrame(
        {"macd_line": macd_line, "macd_signal": macd_signal, "macd_hist": macd_hist},
        index=close.index,
    )


def compute_rolling_vol(close: pd.Series, period: int = 20) -> pd.Series:
    """
    Compute rolling standard deviation of simple returns.

    This provides a second volatility measure (complementary to ATR) that
    is scale-normalised and directly comparable across different price
    levels.

    Args:
        close (pd.Series):
            Closing prices.
        period (int):
            Rolling window length (default 20 bars).

    Returns:
        pd.Series:
            Rolling return volatility.
    """
    returns = close.pct_change()
    return returns.rolling(window=period, min_periods=period).std()


# ─────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────

def build_meta_dataset(
    ticker: str,
    data_root: str = "data/raw",
    results_dir: str = "results",
    seq_len: int = 30,
    pred_len: int = 1,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    vol_lookback: int = 20,
    sl_multiplier: float = 2.0,
    vertical_barrier_periods: int = 5,
    target_names=None,
) -> pd.DataFrame:
    """
    Build the complete meta-labeling dataset for a single ticker.

    This function orchestrates the full pipeline: loading predictions,
    aligning them with actual prices, applying the triple barrier, and
    engineering features.

    Data alignment
    --------------
    ``test.py`` evaluates on the test split of ``ParquetDataset``.  The test
    split starts at row ``val_end = int(N * (train_ratio + val_ratio))``
    of the raw CSV.  Each test sample *i* uses:

        Input window : rows [val_end + i, val_end + i + seq_len)
        Target window: rows [val_end + i + seq_len, val_end + i + seq_len + pred_len)

    The *entry bar* for a trade is the last bar of the input window
    (index ``val_end + i + seq_len - 1``), and the prediction covers
    the next ``pred_len`` bars.

    Args:
        ticker (str):
            Stock ticker symbol (e.g. ``'AAPL'``).
        data_root (str):
            Directory containing ``{ticker}.csv`` with OHLCV data.
        results_dir (str):
            Directory where ``test.py`` saved the ``.npy`` files.
        seq_len (int):
            Look-back window used during training / testing.
        pred_len (int):
            Forecast horizon used during training / testing.
        train_ratio (float):
            Training split ratio (must match ``ParquetDataset``).
        val_ratio (float):
            Validation split ratio (must match ``ParquetDataset``).
        vol_lookback (int):
            Rolling window for the daily-volatility estimate inside the
            triple barrier.
        sl_multiplier (float):
            Stop-loss width in daily-volatility units.
        vertical_barrier_periods (int):
            Maximum holding period for the triple barrier.
        target_names (list[str] | None):
            Ordered names of the prediction targets that ``test.py``
            produced (e.g. ``['High', 'Close', 'EMA_20', 'SMA_50']``).
            Defaults to ``['High', 'Close']`` for backward compatibility.

    Returns:
        pd.DataFrame:
            Cleaned DataFrame ready for LightGBM training with columns:
            ``pred_high``, ``pred_close``, actual OHLCV, ``meta_label``,
            and all engineered features.

    Raises:
        FileNotFoundError:
            If the raw CSV or the ``.npy`` prediction files are missing.
    """
    if target_names is None:
        target_names = ["High", "Close"]

    # Build index lookup for the prediction tensor
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
            f"Run: python scripts/fetch_data.py or "
            f"python scripts/resample_parquet.py --ticker {ticker} first."
        )

    df_raw = pd.read_csv(csv_path).ffill().bfill()
    total_len = len(df_raw)

    # ── 2. Load predictions ──
    pred_path = os.path.join(results_dir, f"{ticker}_predictions.npy")
    true_path = os.path.join(results_dir, f"{ticker}_ground_truth.npy")

    if not os.path.exists(pred_path) or not os.path.exists(true_path):
        raise FileNotFoundError(
            f"Prediction files not found in {results_dir}/.\n"
            f"Run: python test.py --ticker {ticker} --save_predictions first."
        )

    preds = np.load(pred_path)   # [N_test, pred_len, n_targets]
    trues = np.load(true_path)   # [N_test, pred_len, n_targets]

    n_test = preds.shape[0]

    # ── 3. Align prediction indices with the raw CSV ──
    val_end = int(total_len * (train_ratio + val_ratio))

    entry_indices = np.arange(n_test) + val_end + seq_len - 1

    hi = target_idx["High"]
    ci = target_idx["Close"]

    pred_high = preds[:, :, hi].max(axis=1)   # [N_test]
    pred_close = preds[:, -1, ci]              # last-step close prediction

    # Build a DataFrame aligned to entry bars
    df_entry = df_raw.iloc[entry_indices].reset_index(drop=True)
    df_entry["pred_high"] = pred_high
    df_entry["pred_close"] = pred_close

    # Extract last-step MA predictions when available
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

    # ── 5. Engineer features ──
    close = df_labeled["Close"]
    high = df_labeled["High"]
    low = df_labeled["Low"]

    df_labeled["atr"] = compute_atr(high, low, close, period=14)
    df_labeled["rolling_vol"] = compute_rolling_vol(close, period=20)
    df_labeled["rsi"] = compute_rsi(close, period=14)

    macd_df = compute_macd(close, fast=12, slow=26, signal=9)
    df_labeled = pd.concat([df_labeled, macd_df], axis=1)

    # Prediction-derived features: how ambitious is the TP target?
    df_labeled["pred_return"] = (df_labeled["pred_high"] / close) - 1.0
    df_labeled["pred_close_return"] = (df_labeled["pred_close"] / close) - 1.0

    # MA-derived trend features (only when MA targets were predicted)
    if has_ema20:
        df_labeled["pred_ema20_vs_close"] = (
            df_labeled["pred_ema20"] / close
        ) - 1.0
    if has_sma50:
        df_labeled["pred_sma50_vs_close"] = (
            df_labeled["pred_sma50"] / close
        ) - 1.0
    if has_ema20 and has_sma50:
        df_labeled["pred_ma_crossover"] = (
            df_labeled["pred_ema20"] - df_labeled["pred_sma50"]
        )

    # ── 6. Drop warm-up NaNs ──
    n_before = len(df_labeled)
    df_labeled = df_labeled.dropna().reset_index(drop=True)
    n_after = len(df_labeled)
    print(f"Dropped {n_before - n_after} warm-up rows with NaN values.")

    return df_labeled


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate meta-labels and features from TimeMixer predictions."
    )
    parser.add_argument("--ticker", type=str, default="AAPL")
    parser.add_argument("--data_root", type=str, default="data/raw")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--seq_len", type=int, default=30)
    parser.add_argument("--pred_len", type=int, default=5)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--vol_lookback", type=int, default=20)
    parser.add_argument("--sl_multiplier", type=float, default=2.0)
    parser.add_argument("--vertical_barrier", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="data/meta")
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
    print(f"Seq len:         {args.seq_len}")
    print(f"Pred len:        {args.pred_len}")
    print(f"Target names:    {target_names}")
    print(f"SL multiplier:   {args.sl_multiplier}x daily vol")
    print(f"Vertical barrier:{args.vertical_barrier} bars")
    print("=" * 60 + "\n")

    df = build_meta_dataset(
        ticker=args.ticker,
        data_root=args.data_root,
        results_dir=args.results_dir,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        vol_lookback=args.vol_lookback,
        sl_multiplier=args.sl_multiplier,
        vertical_barrier_periods=args.vertical_barrier,
        target_names=target_names,
    )

    # ── Save ──
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"meta_labels_{args.ticker}.csv")
    df.to_csv(out_path, index=False)

    # ── Summary ──
    n_pos = (df["meta_label"] == 1).sum()
    n_neg = (df["meta_label"] == 0).sum()
    print(f"\nSaved {len(df)} rows to {out_path}")
    print(f"  Label 1 (take-profit): {n_pos}  ({100 * n_pos / len(df):.1f}%)")
    print(f"  Label 0 (SL/timeout):  {n_neg}  ({100 * n_neg / len(df):.1f}%)")

    exit_counts = df["exit_type"].value_counts()
    print(f"\nExit breakdown:")
    for exit_type, count in exit_counts.items():
        print(f"  {exit_type:15s}: {count:5d}  ({100 * count / len(df):.1f}%)")

    feature_cols = ["atr", "rolling_vol", "rsi", "macd_line", "macd_signal",
                    "macd_hist", "pred_return", "pred_close_return"]
    for extra in ["Vwap", "Transactions",
                  "pred_ema20_vs_close", "pred_sma50_vs_close",
                  "pred_ma_crossover"]:
        if extra in df.columns:
            feature_cols.append(extra)
    print(f"\nFeature summary:")
    print(df[feature_cols].describe().round(4).to_string())
    print()


if __name__ == "__main__":
    main()
