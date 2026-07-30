"""Canonical causal feature engineering from daily OHLCV.

This module is the **single source of truth** for the model input features.
Every forecaster — LightGBM and the neural nets (TimeMixer/TimesNet) — consumes
exactly this set, so the model comparison is fair (same information, different
learner). The columns are written to each ticker CSV by ``scripts/fetch_data``
and read back by both pipelines.

Every feature is causal (uses only past/current bars, no look-ahead) and
scale-invariant (returns, ratios, bounded oscillators) so the pooled model
generalises across price levels and stocks.

The set was chosen by ablation (``experiments/ablation_features.py``): this lean
12 recovers — and slightly beats — the old 28-feature set (upside IC 0.133 vs
0.128) at less than half the width. Redundant technical variants added noise,
not signal, because they are all deterministic functions of the same OHLCV.

Feature groups (target = forward vol-normalised high/low band):
  * momentum       — ret_1d, ret_5d, ret_20d
  * volatility     — volatility_20, atr, bb_width
  * range structure— high_close_ratio, low_close_ratio, high_low_range, price_pos_20d
  * oscillator/vol — rsi_14, vol_ma_ratio

``volatility_20`` is also the σ[t] used to vol-normalise the range target, so it
must stay in the set.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


# Canonical order of the engineered feature columns (single source of truth).
FEATURE_COLUMNS: List[str] = [
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "volatility_20",
    "atr",
    "bb_width",
    "high_close_ratio",
    "low_close_ratio",
    "high_low_range",
    "price_pos_20d",
    "rsi_14",
    "vol_ma_ratio",
]

# Longest look-back used below; rows before this are dropped as warm-up.
_MAX_WARMUP = 20


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI (causal, EWM smoothing with alpha=1/period)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # When avg_loss == 0 (pure gains) RSI is 100 by definition.
    rsi = rsi.where(avg_loss != 0.0, 100.0)
    return rsi


def add_features(df: pd.DataFrame, drop_warmup: bool = True) -> pd.DataFrame:
    """Append :data:`FEATURE_COLUMNS` to a daily OHLCV DataFrame.

    Args:
        df: DataFrame with ``Open, High, Low, Close, Volume`` columns, sorted
            ascending by date.
        drop_warmup: If True, drop leading rows whose rolling windows have not
            warmed up yet (NaN feature values).

    Returns:
        A new DataFrame with the feature columns appended. Row order and the
        original OHLCV columns are preserved.
    """
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"add_features: missing OHLCV columns: {missing}")

    out = df.copy()
    close = out["Close"]
    high = out["High"]
    low = out["Low"]
    volume = out["Volume"]
    prev_close = close.shift(1)

    # -- Momentum (multi-horizon returns) --
    out["ret_1d"] = close.pct_change(1)
    out["ret_5d"] = close.pct_change(5)
    out["ret_20d"] = close.pct_change(20)

    # -- Volatility --
    log_return = np.log(close / prev_close)
    out["volatility_20"] = log_return.rolling(window=20, min_periods=20).std()
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    out["atr"] = tr.rolling(window=14, min_periods=14).mean() / close
    sma20 = close.rolling(window=20, min_periods=20).mean()
    std20 = close.rolling(window=20, min_periods=20).std()
    out["bb_width"] = (2.0 * std20) / sma20.replace(0.0, np.nan)

    # -- Range structure (directly relevant to a forward high/low band) --
    out["high_close_ratio"] = high / close
    out["low_close_ratio"] = low / close
    out["high_low_range"] = (high - low) / close
    roll_high = high.rolling(window=20, min_periods=20).max()
    roll_low = low.rolling(window=20, min_periods=20).min()
    denom = (roll_high - roll_low).replace(0.0, np.nan)
    out["price_pos_20d"] = (close - roll_low) / denom

    # -- Oscillator + volume --
    out["rsi_14"] = _rsi(close, period=14)
    vol_ma20 = volume.rolling(window=20, min_periods=20).mean()
    out["vol_ma_ratio"] = volume / vol_ma20.replace(0.0, np.nan)

    if drop_warmup:
        out = out.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)

    return out
