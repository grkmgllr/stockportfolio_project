"""Technical-indicator features consumed by the meta-classifier."""

import numpy as np
import pandas as pd


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series,
                period: int = 14) -> pd.Series:
    """Average True Range — rolling mean of the true range (gap-aware)."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index — Wilder's smoothing, bounded in [0, 100]."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26,
                 signal: int = 9) -> pd.DataFrame:
    """MACD line + signal + histogram."""
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
    """Rolling standard deviation of simple returns (scale-free volatility)."""
    returns = close.pct_change()
    return returns.rolling(window=period, min_periods=period).std()


def add_market_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ATR / rolling-vol / RSI / MACD columns to a bar-level DataFrame."""
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    df["atr"] = compute_atr(high, low, close, period=14)
    df["rolling_vol"] = compute_rolling_vol(close, period=20)
    df["rsi"] = compute_rsi(close, period=14)
    macd_df = compute_macd(close, fast=12, slow=26, signal=9)
    return pd.concat([df, macd_df], axis=1)


def add_prediction_features(df: pd.DataFrame, *,
                            has_ema20: bool, has_sma50: bool) -> pd.DataFrame:
    """Add ratio features derived from the primary model's predictions."""
    close = df["Close"]

    df["pred_return"] = (df["pred_high"] / close) - 1.0
    df["pred_close_return"] = (df["pred_close"] / close) - 1.0

    if has_ema20:
        df["pred_ema20_vs_close"] = (df["pred_ema20"] / close) - 1.0
    if has_sma50:
        df["pred_sma50_vs_close"] = (df["pred_sma50"] / close) - 1.0
    if has_ema20 and has_sma50:
        df["pred_ma_crossover"] = df["pred_ema20"] - df["pred_sma50"]

    return df
