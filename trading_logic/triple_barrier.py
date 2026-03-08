"""
Triple Barrier Method for Meta-Label generation.

Implements the labeling framework from Marcos Lopez de Prado's
*Advances in Financial Machine Learning* (Chapter 3).

The primary model (TimesNet, TimeMixer, or LightGBM) provides the direction
signal and predicts the take-profit level (predicted High).  This module
evaluates whether that prediction would have been *profitable* by simulating
a trade with three exit conditions (barriers):

    Upper barrier  -  Take-Profit: predicted High price from primary model
    Lower barrier  -  Stop-Loss:   dynamic, set at a multiple of daily volatility
    Vertical barrier - Timeout:    max holding period in bars

Design goals
------------
- Separate the *direction* decision (handled by the primary DL model) from the
  *sizing / filtering* decision (handled by the secondary ML model).
- Produce binary labels that answer: "Given the primary model said BUY here,
  would that trade have been profitable?"
- Track event windows (t_start, t_end) so that PurgedKFold (Phase 4) can
  correctly purge overlapping observations between train and test folds.

Output contract
---------------
    1  if the take-profit (upper barrier) is hit first.
    0  if the stop-loss (lower barrier) or the timeout (vertical barrier)
       is hit first.
"""

import pandas as pd
import numpy as np
from typing import Tuple


def get_daily_volatility(close: pd.Series, lookback: int = 20,) -> pd.Series:
    """
    Compute daily volatility as the rolling standard deviation of log returns.

    This is the fundamental risk measure used to set the dynamic stop-loss
    barrier. Using log returns (rather than simple returns) ensures the
    estimate is symmetric and additive across time, which is a standard
    practice in quantitative finance.

    Args:
        close (pd.Series):
            Series of closing prices.  Must contain strictly positive values.
        lookback (int):
            Rolling window length in bars.  Default is 20, which corresponds
            to approximately one trading month for daily data.

    Returns:
        pd.Series:
            Series of daily volatility values, aligned to the same index as
            ``close``.  The first ``lookback`` entries will be NaN because
            there are not enough observations to fill the rolling window.

    Notes
    -----
    - The output is *not* annualised.  It represents the per-bar standard
      deviation of log returns, which is what we need for setting intraday
      or daily stop-loss levels.
    - A ``min_periods=lookback`` constraint is applied so that partial windows
      at the start of the series do not produce unreliable estimates.
    """
    log_returns = np.log(close / close.shift(1))
    return log_returns.rolling(window=lookback, min_periods=lookback).std()


def apply_triple_barrier(df: pd.DataFrame, pred_high_col: str = "pred_high", close_col: str = "Close", high_col: str = "High", low_col: str = "Low",
    vol_lookback: int = 20, sl_multiplier: float = 2.0, vertical_barrier_periods: int = 5, min_sl_pct: float = 0.005,) -> pd.DataFrame:
    """
    Apply the Triple Barrier Method with a dynamic volatility-based stop-loss.

    For every bar where the primary model generates a long signal, three
    barriers are constructed around the entry price.  The function then
    simulates the trade forward in time to determine which barrier is
    touched first, and assigns a binary meta-label accordingly.

    Barrier construction
    --------------------
    Upper barrier (Take-Profit):
        Set to the predicted High price from the primary model.  This
        directly evaluates the model's price-level forecast: "Did the
        market actually reach the price the model predicted?"
        Clipped to be at least marginally above the entry price to avoid
        degenerate labels where TP == entry.

    Lower barrier (Stop-Loss):
        Set dynamically at ``entry * (1 - max(daily_vol * sl_multiplier,
        min_sl_pct))``.  In high-volatility regimes the stop widens to
        accommodate larger swings; in calm markets it tightens to protect
        capital.  A ``min_sl_pct`` floor prevents near-zero stops during
        extremely low-volatility periods.

    Vertical barrier (Timeout):
        The maximum number of bars the trade is allowed to remain open.
        If neither the upper nor lower barrier is hit within this window,
        the trade is closed at a loss (label = 0).

    Args:
        df (pd.DataFrame):
            DataFrame containing actual OHLC price columns **and** the
            primary model's predicted high price.  One row per bar.
        pred_high_col (str):
            Column name holding the primary model's predicted High price.
            This value is used *directly* as the upper-barrier
            (take-profit) level.
        close_col (str):
            Column name for the execution / entry price (close of the
            signal bar).
        high_col (str):
            Column name for the actual intraday high (used to detect
            take-profit hits during the forward window).
        low_col (str):
            Column name for the actual intraday low (used to detect
            stop-loss hits during the forward window).
        vol_lookback (int):
            Rolling window length for the daily volatility estimate.
            Default is 20 (~ 1 trading month).
        sl_multiplier (float):
            Number of daily-volatility units below entry to place the
            stop-loss.  Higher values produce a wider stop (fewer
            stop-outs but larger losses when they occur).
        vertical_barrier_periods (int):
            Maximum bars to hold before forcing an exit (timeout).
        min_sl_pct (float):
            Floor on the stop-loss distance as a fraction of entry price.
            Prevents a near-zero stop during extremely low-volatility
            regimes.

    Returns:
        pd.DataFrame:
            A copy of ``df`` with the following columns appended:

            ===============  ================================================
            ``daily_vol``    Per-bar rolling volatility estimate.
            ``upper_barrier``Absolute take-profit price level.
            ``lower_barrier``Absolute stop-loss price level.
            ``meta_label``   1 if TP hit first, 0 if SL or timeout.
            ``exit_type``    ``'take_profit'`` | ``'stop_loss'`` | ``'timeout'``
            ``t_start``      Event window start index (the signal bar).
            ``t_end``        Event window end index (where the trade exits).
            ===============  ================================================

    Notes
    -----
    - Rows where the volatility estimate is NaN (the first ``vol_lookback``
      rows) receive a default label of 0 / ``'timeout'``.  These should be
      dropped before training the meta-classifier.
    - The ``t_start`` / ``t_end`` columns are integer positional indices into
      the DataFrame.  They are consumed by ``PurgedKFold`` (Phase 4) to purge
      training observations whose event windows overlap with the test set.
    - Barrier levels are computed in a vectorised pass.  Only the forward scan
      that checks which barrier is hit first uses a Python loop, where each
      per-window check is performed with ``np.where`` on numpy arrays for
      performance.
    """
    out = df.copy()

    # --- dynamic volatility ---
    out["daily_vol"] = get_daily_volatility(out[close_col], lookback=vol_lookback)

    # --- barrier levels (vectorised) ---
    entry = out[close_col]
    pred_high = out[pred_high_col]

    out["upper_barrier"] = pred_high.clip(lower=entry * 1.0001)

    sl_pct = (out["daily_vol"] * sl_multiplier).clip(lower=min_sl_pct)
    out["lower_barrier"] = entry * (1.0 - sl_pct)

    # --- initialise output columns ---
    n = len(out)
    out["meta_label"] = 0
    out["exit_type"] = "timeout"
    out["t_start"] = np.arange(n)
    out["t_end"] = np.minimum(
        np.arange(n) + vertical_barrier_periods, n - 1
    )

    # --- scan each event window ---
    highs = out[high_col].values
    lows = out[low_col].values
    upper = out["upper_barrier"].values
    lower = out["lower_barrier"].values

    labels = np.zeros(n, dtype=np.int8)
    exits = np.full(n, "timeout", dtype=object)
    t_ends = out["t_end"].values.copy()

    for i in range(n):
        if np.isnan(upper[i]) or np.isnan(lower[i]):
            continue

        window_end = min(i + 1 + vertical_barrier_periods, n)
        if i + 1 >= window_end:
            continue

        w_highs = highs[i + 1 : window_end]
        w_lows = lows[i + 1 : window_end]

        tp_hits = np.where(w_highs >= upper[i])[0]
        sl_hits = np.where(w_lows <= lower[i])[0]

        tp_bar = tp_hits[0] if len(tp_hits) > 0 else None
        sl_bar = sl_hits[0] if len(sl_hits) > 0 else None

        if tp_bar is not None and sl_bar is None:
            labels[i] = 1
            exits[i] = "take_profit"
            t_ends[i] = i + 1 + tp_bar

        elif sl_bar is not None and tp_bar is None:
            labels[i] = 0
            exits[i] = "stop_loss"
            t_ends[i] = i + 1 + sl_bar

        elif tp_bar is not None and sl_bar is not None:
            if tp_bar <= sl_bar:
                labels[i] = 1
                exits[i] = "take_profit"
                t_ends[i] = i + 1 + tp_bar
            else:
                labels[i] = 0
                exits[i] = "stop_loss"
                t_ends[i] = i + 1 + sl_bar

    out["meta_label"] = labels
    out["exit_type"] = exits
    out["t_end"] = t_ends

    return out


def get_event_spans(labeled_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract (t_start, t_end) pairs from a labeled DataFrame.

    These event spans define the time interval during which each trade is
    "alive" (from signal bar to the bar where a barrier is hit).  They are
    consumed by ``PurgedKFold`` to identify and purge training observations
    whose event windows overlap with the test set, preventing data leakage
    in time-series cross-validation.

    Args:
        labeled_df (pd.DataFrame):
            Output of ``apply_triple_barrier``.  Must contain the columns
            ``t_start`` and ``t_end``.

    Returns:
        pd.DataFrame:
            Two-column DataFrame with ``['t_start', 't_end']``, one row
            per event.  The index is preserved from the input.
    """
    return labeled_df[["t_start", "t_end"]].copy()
