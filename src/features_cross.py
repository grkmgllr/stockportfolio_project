"""Cross-sectional / market-context features (panel-level, causal).

The per-ticker features in :mod:`features` and
``LightGBMForecaster.engineer_features`` describe each stock **in isolation**.
This module adds the missing dimension: where a stock sits **relative to the
rest of the universe** on each date, and how the whole market is moving.

All features are causal — every value at date *t* uses only information known
at or before the close of *t*:

* ``mkt_ret_{1,5,20}d`` — equal-weight universe log-return (same-date / trailing).
* ``rel_strength_20d``  — the stock's own 20-day log-return minus the market's
  (idiosyncratic momentum: did it beat the tape?).
* ``beta_60d``          — trailing 60-day cov(stock, market) / var(market).
* ``xs_rank_mom20``     — cross-sectional percentile rank of 20-day momentum
  across the universe on date *t* (bounded [0, 1], scale-free).
* ``xs_rank_vol20``     — percentile rank of ``volatility_20``.
* ``xs_rank_rsi``       — percentile rank of ``rsi_14``.

The equal-weight universe return is used as the market proxy (no external
index needed): the panel *is* the market, which keeps the design self-contained
and free of index-membership / survivorship arguments beyond the ones the
universe itself already carries.

Ranks use only the same-date cross-section; the market return uses only same
or past dates; beta uses a trailing window — so computing these on the full
panel (including future rows) and slicing to a train window is identical to
computing them on the train window alone. No look-ahead is introduced.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
import pandas as pd


# Canonical order of the cross-sectional columns (fixed so the LightGBM
# feature-name order is identical between training and prediction).
CROSS_FEATURE_COLUMNS: List[str] = [
    "mkt_ret_1d",
    "mkt_ret_5d",
    "mkt_ret_20d",
    "rel_strength_20d",
    "beta_60d",
    "xs_rank_mom20",
    "xs_rank_vol20",
    "xs_rank_rsi",
]

_REQUIRED = {"Date", "Close", "ret_1d", "volatility_20", "rsi_14"}


def add_cross_sectional_features(
    dfs: Sequence[pd.DataFrame], tickers: Sequence[str]
) -> List[pd.DataFrame]:
    """Return copies of ``dfs`` with :data:`CROSS_FEATURE_COLUMNS` appended.

    Args:
        dfs: One DataFrame per ticker, each with at least ``Date, Close,
            log_return, volatility_20, rsi_14`` and sorted ascending by date.
        tickers: Ticker symbols, aligned positionally with ``dfs``.

    Cross-sectional values are computed by aligning every ticker on ``Date``,
    so tickers with differing coverage are handled per-date on whatever names
    are present that day.
    """
    if len(dfs) != len(tickers):
        raise ValueError("dfs and tickers must have the same length")
    for t, df in zip(tickers, dfs):
        missing = _REQUIRED - set(df.columns)
        if missing:
            raise ValueError(f"{t}: add_cross_sectional_features missing {missing}")

    # -- Wide Date-indexed panels: rows = dates, cols = tickers --
    lr = pd.DataFrame({t: df.set_index("Date")["ret_1d"] for t, df in zip(tickers, dfs)})
    mom = pd.DataFrame({t: df.set_index("Date")["Close"].pct_change(20) for t, df in zip(tickers, dfs)})
    vol = pd.DataFrame({t: df.set_index("Date")["volatility_20"] for t, df in zip(tickers, dfs)})
    rsi = pd.DataFrame({t: df.set_index("Date")["rsi_14"] for t, df in zip(tickers, dfs)})

    # Equal-weight market proxy (mean daily log-return across the universe).
    mkt = lr.mean(axis=1)
    mkt_5 = mkt.rolling(5, min_periods=5).sum()
    mkt_20 = mkt.rolling(20, min_periods=20).sum()

    # Cross-sectional percentile ranks (same-date, bounded [0, 1]).
    r_mom = mom.rank(axis=1, pct=True)
    r_vol = vol.rank(axis=1, pct=True)
    r_rsi = rsi.rank(axis=1, pct=True)

    out: List[pd.DataFrame] = []
    for t, df in zip(tickers, dfs):
        d = df.copy()
        idx = d["Date"]

        d["mkt_ret_1d"] = mkt.reindex(idx).to_numpy()
        d["mkt_ret_5d"] = mkt_5.reindex(idx).to_numpy()
        d["mkt_ret_20d"] = mkt_20.reindex(idx).to_numpy()

        own_20 = np.log(d["Close"] / d["Close"].shift(20))
        d["rel_strength_20d"] = own_20.to_numpy() - mkt_20.reindex(idx).to_numpy()

        # Trailing 60-day beta of the stock to the equal-weight market.
        own_lr = d["ret_1d"].reset_index(drop=True)
        m_lr = pd.Series(mkt.reindex(idx).to_numpy())
        cov = own_lr.rolling(60, min_periods=60).cov(m_lr)
        var = m_lr.rolling(60, min_periods=60).var()
        d["beta_60d"] = (cov / var.replace(0.0, np.nan)).to_numpy()

        d["xs_rank_mom20"] = r_mom[t].reindex(idx).to_numpy()
        d["xs_rank_vol20"] = r_vol[t].reindex(idx).to_numpy()
        d["xs_rank_rsi"] = r_rsi[t].reindex(idx).to_numpy()

        out.append(d)

    return out
