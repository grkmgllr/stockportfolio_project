"""
Daily OHLCV downloader from Yahoo Finance + causal feature engineering.

Downloads daily bars for a universe of tickers, appends the engineered
feature columns from :mod:`features`, and writes one CSV per ticker into
``data/raw/{ticker}.csv``. These CSVs are the single input contract for
``StockDataset`` and ``CrossStockDataset``.

This is the whole data pipeline now — Polygon.io minute-bar parquet and
``resample_parquet.py`` are no longer used.

Usage
-----
    python src/scripts/fetch_data.py                       # default universe
    python src/scripts/fetch_data.py --tickers AAPL MSFT   # ad-hoc list
    python src/scripts/fetch_data.py --start 2015-01-01 --end 2025-11-30
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List

import pandas as pd
import yfinance as yf

# Allow `from features import ...` when run as a script from repo root.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from features import FEATURE_COLUMNS, add_features  # noqa: E402

RAW_DATA_DIR = "data/raw"

# Small starter universe — validate the pipeline here, expand later.
STARTER_UNIVERSE: List[str] = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META",
    "NVDA", "TSLA", "AMD", "AVGO", "NFLX",
    "JPM", "BAC", "V", "MA", "COST",
]

# Nasdaq-100 constituents (approximate, point-in-time). Index membership
# changes over time, so a few of these may be stale; fetch_universe simply
# skips any ticker Yahoo has no data for, so an imperfect list is harmless.
# Edit freely to match the exact universe you want to study.
NDX100: List[str] = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "GOOG", "META", "TSLA", "AVGO",
    "COST", "NFLX", "ASML", "AMD", "PEP", "ADBE", "LIN", "CSCO", "TMUS",
    "INTU", "QCOM", "TXN", "AMGN", "ISRG", "AMAT", "BKNG", "HON", "VRTX",
    "ADP", "PANW", "GILD", "MU", "ADI", "REGN", "SBUX", "MDLZ", "LRCX",
    "KLAC", "SNPS", "CDNS", "PYPL", "MELI", "CRWD", "MAR", "CTAS", "ORLY",
    "ABNB", "CEG", "WDAY", "NXPI", "PCAR", "ROP", "MRVL", "MNST", "ADSK",
    "FTNT", "DASH", "TTD", "CPRT", "PAYX", "KDP", "ROST", "ODFL", "FANG",
    "EA", "VRSK", "XEL", "KHC", "EXC", "GEHC", "CTSH", "FAST", "CCEP",
    "DDOG", "LULU", "BKR", "CSGP", "IDXX", "TEAM", "ON", "ZS", "DXCM",
    "CDW", "TTWO", "WBD", "GFS", "MRNA", "MDB", "ARM", "SMCI", "BIIB",
    "PDD", "INTC", "AEP", "CMCSA",
]

# Named universes selectable via --universe (or the DEFAULT_UNIVERSE alias
# kept for backward compatibility with older callers).
UNIVERSES = {"starter": STARTER_UNIVERSE, "ndx100": NDX100}
DEFAULT_UNIVERSE = STARTER_UNIVERSE

OHLCV = ["Open", "High", "Low", "Close", "Volume"]
OUTPUT_COLUMNS = ["Date"] + OHLCV + FEATURE_COLUMNS


def _download_one(ticker: str, start: str, end: str, retries: int = 3) -> pd.DataFrame:
    """Download a single ticker's daily OHLCV with simple retry/backoff."""
    for attempt in range(1, retries + 1):
        try:
            df = yf.download(
                ticker, start=start, end=end,
                auto_adjust=True, progress=False,
            )
            if df is not None and not df.empty:
                return df
            print(f"  [{ticker}] empty response (attempt {attempt}/{retries})")
        except Exception as e:  # network hiccups, rate limits, etc.
            print(f"  [{ticker}] error (attempt {attempt}/{retries}): {e}")
        time.sleep(2 * attempt)
    return pd.DataFrame()


def _flatten(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Normalize a yfinance frame to Date + OHLCV columns."""
    # yfinance returns a MultiIndex (field, ticker) column layout for a
    # single ticker too depending on version — collapse it.
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel(1, axis=1)
    df = df.reset_index().rename(columns={"index": "Date"})
    if "Date" not in df.columns and "Datetime" in df.columns:
        df = df.rename(columns={"Datetime": "Date"})
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    keep = ["Date"] + [c for c in OHLCV if c in df.columns]
    return df[keep]


def fetch_universe(tickers: List[str], start: str, end: str) -> None:
    """Download, feature-engineer, and save one CSV per ticker."""
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    print(f"Fetching {len(tickers)} tickers from {start} to {end}\n")

    ok, skipped = [], []
    for ticker in tickers:
        print(f"[{ticker}] downloading...")
        raw = _download_one(ticker, start, end)
        if raw.empty:
            print(f"  SKIP: no data for {ticker}")
            skipped.append(ticker)
            continue

        df = _flatten(raw, ticker)
        if not set(OHLCV).issubset(df.columns):
            print(f"  SKIP: missing OHLCV columns for {ticker} ({list(df.columns)})")
            skipped.append(ticker)
            continue

        df = df.sort_values("Date").reset_index(drop=True)
        df = add_features(df, drop_warmup=True)

        if df.empty:
            print(f"  SKIP: no rows left after feature warm-up for {ticker}")
            skipped.append(ticker)
            continue

        out_path = os.path.join(RAW_DATA_DIR, f"{ticker}.csv")
        df[OUTPUT_COLUMNS].to_csv(out_path, index=False)
        print(f"  -> {out_path}  ({len(df)} rows, {df['Date'].iloc[0]} → {df['Date'].iloc[-1]})")
        ok.append(ticker)

    print(f"\nDone. {len(ok)} saved, {len(skipped)} skipped.")
    if skipped:
        print(f"  Skipped: {skipped}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch daily OHLCV + features from Yahoo Finance.")
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Explicit ticker list (overrides --universe).")
    parser.add_argument("--universe", choices=list(UNIVERSES), default="starter",
                        help="Named universe to fetch (default: starter, 15 tickers).")
    parser.add_argument("--start", type=str, default="2015-01-01")
    parser.add_argument("--end", type=str, default="2025-11-30")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    universe = args.tickers if args.tickers else UNIVERSES[args.universe]
    fetch_universe(universe, args.start, args.end)
