"""
Causal feature engineering from daily OHLCV.

Every feature here is computed using only past/current bars (no look-ahead),
so it is safe to feed as model input. Rows whose rolling windows have not
warmed up yet are dropped by :func:`add_features`, so no NaN leaks into the
downstream StandardScaler (which would otherwise be filled by bfill and leak
future information).

These replace the Polygon-era ``Vwap`` / ``Transactions`` columns, which are
not available from daily Yahoo Finance data.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


# Canonical order of the engineered feature columns.
FEATURE_COLUMNS: List[str] = [
    "log_return",
    "overnight_gap",
    "intraday_move",
    "daily_range",
    "rel_volume",
    "volatility_20",
    "rsi_14",
    "dist_sma_20",
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
    """Append the engineered feature columns to a daily OHLCV DataFrame.

    Args:
        df: DataFrame with ``Open, High, Low, Close, Volume`` columns, sorted
            ascending by date.
        drop_warmup: If True, drop leading rows that contain NaN feature
            values (the rolling-window warm-up period).

    Returns:
        A new DataFrame with :data:`FEATURE_COLUMNS` appended. Row order and
        the original OHLCV columns are preserved.
    """
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"add_features: missing OHLCV columns: {missing}")

    out = df.copy()
    close = out["Close"]
    prev_close = close.shift(1)

    out["log_return"] = np.log(close / prev_close)
    out["overnight_gap"] = (out["Open"] - prev_close) / prev_close
    out["intraday_move"] = (out["Close"] - out["Open"]) / out["Open"]
    out["daily_range"] = (out["High"] - out["Low"]) / close

    vol_ma20 = out["Volume"].rolling(window=20, min_periods=20).mean()
    out["rel_volume"] = out["Volume"] / vol_ma20.replace(0.0, np.nan)

    out["volatility_20"] = out["log_return"].rolling(window=20, min_periods=20).std()

    out["rsi_14"] = _rsi(close, period=14)

    sma20 = close.rolling(window=20, min_periods=20).mean()
    out["dist_sma_20"] = (close - sma20) / sma20

    if drop_warmup:
        out = out.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)

    return out
