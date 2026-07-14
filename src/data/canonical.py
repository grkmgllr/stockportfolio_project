"""
Canonical multi-ticker panel builder.

Turns the raw ``data/raw/universe.parquet`` (long-format OHLCV + sector)
into a *calendar-aligned* panel that every downstream adapter can share.

What "canonical" means here
---------------------------
1. **Trading calendar**: a single sorted list of dates, taken as the
   union of all tickers' available dates. Every ticker is reindexed
   onto this calendar so the panel is a full (n_dates x n_tickers) grid.
2. **`available` mask**: True where the ticker actually traded that day,
   False for pre-IPO / delisted / gap rows. Cross-sectional models
   (MASTER) use this to zero out attention for absent tickers.
3. **Returns**: computed *before* reindexing (per ticker on its own
   traded days) so that gaps don't leak zero returns.
   - ``log_return``: log(close_t / close_{t-1}) using adj_close.
   - ``fwd_return_1d``, ``fwd_return_5d``: forward log returns for
     Stage-1 targets. NaN at the tail where horizon runs past the data.
4. **OHLC shape features**: scale-invariant intraday features derived
   from Open/High/Low/Close/PrevClose. Raw OHLC prices are kept in the
   panel for reference but downstream models should prefer these:
   - ``daily_range = (H - L) / C``
   - ``intraday_move = (C - O) / O``
   - ``upper_wick = (H - max(O, C)) / C``
   - ``lower_wick = (min(O, C) - L) / C``
   - ``overnight_gap = (O - PrevClose) / PrevClose``
   - ``log_volume = log1p(volume)`` (raw scale, normalize later)

Deliberately NOT here
---------------------
- Normalization (per-series vs cross-sectional) — that is `views.py`.
- Model-specific reshape (tabular / per-series / cross-section) — adapters.
- Alpha158-style feature engineering — separate module later.

Outputs
-------
    data/processed/canonical_panel.parquet
        Columns: date, ticker, sector, industry, available,
                 open, high, low, close, adj_close, volume,
                 log_return, fwd_return_1d, fwd_return_5d,
                 daily_range, intraday_move, upper_wick, lower_wick,
                 overnight_gap, log_volume
    data/processed/canonical_meta.json
        {n_dates, n_tickers, date_min, date_max, coverage_pct}

Usage
-----
    python src/data/canonical.py
    python src/data/canonical.py --raw data/raw/universe.parquet \\
                                  --out data/processed/canonical_panel.parquet
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

RAW_PATH = "data/raw/universe.parquet"
OUT_DIR = "data/processed"
OUT_PATH = os.path.join(OUT_DIR, "canonical_panel.parquet")
META_PATH = os.path.join(OUT_DIR, "canonical_meta.json")


def _compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add returns + OHLC shape features per ticker (on traded days only)."""
    out = []
    for tkr, g in df.groupby("ticker", sort=False):
        g = g.sort_values("date").copy()
        o = g["open"].astype("float64")
        h = g["high"].astype("float64")
        low = g["low"].astype("float64")
        c = g["close"].astype("float64")
        vol = g["volume"].astype("float64")
        log_adj = np.log(g["adj_close"].astype("float64"))

        # Return targets (adj_close-based, dividend/split aware)
        g["log_return"] = log_adj.diff()
        g["fwd_return_1d"] = log_adj.shift(-1) - log_adj
        g["fwd_return_5d"] = log_adj.shift(-5) - log_adj

        # Intraday shape (scale-invariant)
        g["daily_range"] = (h - low) / c
        g["intraday_move"] = (c - o) / o
        body_top = np.maximum(o, c)
        body_bot = np.minimum(o, c)
        g["upper_wick"] = (h - body_top) / c
        g["lower_wick"] = (body_bot - low) / c
        g["overnight_gap"] = (o - c.shift(1)) / c.shift(1)

        # Volume: log1p keeps 0-volume days finite; z-score happens in views
        g["log_volume"] = np.log1p(vol)

        out.append(g)
    return pd.concat(out, ignore_index=True)


def build_canonical(raw: pd.DataFrame) -> pd.DataFrame:
    """Calendar-align, mask, and return-augment the raw panel."""
    raw = raw.copy()
    raw["date"] = pd.to_datetime(raw["date"])

    # Compute returns on traded days first, before reindexing
    with_returns = _compute_returns(raw)

    # Trading calendar = union of all tickers' dates
    calendar = pd.Index(sorted(with_returns["date"].unique()), name="date")
    tickers = sorted(with_returns["ticker"].unique())

    # Full grid: n_dates x n_tickers
    grid = pd.MultiIndex.from_product([calendar, tickers], names=["date", "ticker"])
    grid_df = pd.DataFrame(index=grid).reset_index()

    # Merge; missing rows -> NaN in numeric cols
    panel = grid_df.merge(with_returns, on=["date", "ticker"], how="left")

    # Fill sector/industry per ticker (constant), forward-fill from any row
    static_cols = ["sector", "industry"]
    static_map = (
        with_returns.dropna(subset=static_cols)
        .drop_duplicates("ticker")[["ticker"] + static_cols]
    )
    panel = panel.drop(columns=static_cols).merge(static_map, on="ticker", how="left")

    # available = did this ticker actually trade this date?
    panel["available"] = panel["close"].notna()

    # Column order: identity, mask, prices, returns
    ordered = [
        "date", "ticker", "sector", "industry", "available",
        "open", "high", "low", "close", "adj_close", "volume",
        "log_return", "fwd_return_1d", "fwd_return_5d",
        "daily_range", "intraday_move", "upper_wick", "lower_wick",
        "overnight_gap", "log_volume",
    ]
    panel = panel[ordered].sort_values(["ticker", "date"]).reset_index(drop=True)
    return panel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--raw", default=RAW_PATH)
    p.add_argument("--out", default=OUT_PATH)
    p.add_argument("--meta", default=META_PATH)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print(f"Loading raw panel: {args.raw}")
    raw = pd.read_parquet(args.raw)
    print(f"  {len(raw):,} rows, {raw['ticker'].nunique()} tickers")

    panel = build_canonical(raw)

    n_dates = panel["date"].nunique()
    n_tickers = panel["ticker"].nunique()
    n_expected = n_dates * n_tickers
    coverage = panel["available"].mean() * 100

    print(f"\nCanonical panel: {len(panel):,} rows "
          f"(expected {n_expected:,} = {n_dates} dates x {n_tickers} tickers)")
    print(f"Coverage: {coverage:.2f}% cells available")
    print(f"Date range: {panel['date'].min().date()} -> {panel['date'].max().date()}")

    panel.to_parquet(args.out, index=False)
    meta = {
        "n_dates": int(n_dates),
        "n_tickers": int(n_tickers),
        "date_min": str(panel["date"].min().date()),
        "date_max": str(panel["date"].max().date()),
        "coverage_pct": round(float(coverage), 4),
    }
    with open(args.meta, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  panel -> {args.out}")
    print(f"  meta  -> {args.meta}")

    # Sanity: pick one recent-IPO ticker and show the transition
    recent = panel.groupby("ticker")["available"].sum().sort_values().index[0]
    sample = panel[panel["ticker"] == recent]
    first_avail = sample[sample["available"]]["date"].min()
    print(f"\nSanity check — {recent}: first available date = {first_avail.date()}, "
          f"{(~sample['available']).sum()} pre-IPO masked rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
