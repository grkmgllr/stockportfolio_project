"""
Download a stock universe (NASDAQ 100 by default) from Yahoo Finance.

Fetches daily OHLCV + Adj Close for every ticker in the universe,
enriches with sector/industry metadata, and saves a single long-format
parquet plus a small metadata CSV.

Outputs
-------
    data/raw/universe.parquet
        Long-format panel:
        date, ticker, sector, industry, open, high, low, close, adj_close, volume
    data/raw/universe_metadata.csv
        ticker -> sector, industry, name, download status, row count

Usage
-----
    # Default: NDX 100, 2010 -> today
    python src/scripts/download_universe.py

    # Custom universe (e.g. S&P 500) and date range
    python src/scripts/download_universe.py --universe sp500 --start 2005-01-01

    # Explicit ticker list (skips Wikipedia scrape)
    python src/scripts/download_universe.py --tickers AAPL MSFT NVDA

Note
----
Uses *current* index constituents (survivor bias). Fine for a first
MASTER integration pass; switch to historical constituents before any
final benchmarking.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date
from typing import List

import io

import pandas as pd
import requests
import yfinance as yf

_HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

RAW_DIR = "data/raw"

SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

# NASDAQ-100 constituents snapshot (~2026-Q3).
# Wikipedia no longer renders the constituents list as a static table
# (moved to a JS-loaded template), so we ship a static list here.
# Update annually or override with --tickers.
NDX100_SNAPSHOT = [
    "AAPL", "ABNB", "ADBE", "ADI", "ADP", "ADSK", "AEP", "AMAT", "AMD", "AMGN",
    "AMZN", "ANSS", "APP", "ARM", "ASML", "AVGO", "AXON", "AZN", "BIIB", "BKNG",
    "BKR", "CCEP", "CDNS", "CDW", "CEG", "CHTR", "CMCSA", "COST", "CPRT", "CRWD",
    "CSCO", "CSGP", "CSX", "CTAS", "CTSH", "DASH", "DDOG", "DXCM", "EA", "EXC",
    "FANG", "FAST", "FTNT", "GEHC", "GFS", "GILD", "GOOG", "GOOGL", "HON", "IDXX",
    "INTC", "INTU", "ISRG", "KDP", "KHC", "KLAC", "LIN", "LRCX", "LULU", "MAR",
    "MCHP", "MDB", "MDLZ", "MELI", "META", "MNST", "MRVL", "MSFT", "MSTR", "MU",
    "NFLX", "NVDA", "NXPI", "ODFL", "ON", "ORLY", "PANW", "PAYX", "PCAR", "PDD",
    "PEP", "PLTR", "PYPL", "QCOM", "REGN", "ROP", "ROST", "SBUX", "SNPS", "TEAM",
    "TMUS", "TSLA", "TTD", "TTWO", "TXN", "VRSK", "VRTX", "WBD", "WDAY", "XEL",
    "ZS",
]


def _read_wiki_tables(url: str) -> list:
    """Fetch a Wikipedia page with a real User-Agent and parse tables."""
    resp = requests.get(url, headers=_HTTP_HEADERS, timeout=30)
    resp.raise_for_status()
    return pd.read_html(io.StringIO(resp.text))


def fetch_universe_tickers(universe: str) -> List[str]:
    """Return current index constituents."""
    if universe == "ndx100":
        return sorted(set(NDX100_SNAPSHOT))

    if universe == "sp500":
        tables = _read_wiki_tables(SP500_WIKI_URL)
        tickers = tables[0]["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
        return sorted(set(tickers))

    raise ValueError(f"Unknown universe: {universe}")


def download_ohlcv(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """Batch-download daily bars for all tickers, return long-format DataFrame."""
    print(f"Downloading {len(tickers)} tickers, {start} -> {end} ...")
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        group_by="ticker",
        auto_adjust=False,
        progress=True,
        threads=True,
    )
    if raw.empty:
        raise RuntimeError("yfinance returned empty data. Check network / tickers.")

    frames = []
    for tkr in tickers:
        if tkr not in raw.columns.get_level_values(0):
            print(f"  [warn] no data returned for {tkr}, skipping")
            continue
        df = raw[tkr].copy()
        df = df.dropna(how="all")
        if df.empty:
            print(f"  [warn] {tkr} all-NaN, skipping")
            continue
        df = df.reset_index().rename(columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        })
        df["ticker"] = tkr
        frames.append(df)

    if not frames:
        raise RuntimeError("No tickers produced usable data.")

    panel = pd.concat(frames, ignore_index=True)
    return panel


def fetch_metadata(tickers: List[str]) -> pd.DataFrame:
    """Fetch sector/industry/name for each ticker (one API call per ticker)."""
    print(f"Fetching metadata for {len(tickers)} tickers ...")
    rows = []
    for i, tkr in enumerate(tickers, 1):
        try:
            info = yf.Ticker(tkr).info or {}
            rows.append({
                "ticker": tkr,
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
                "name": info.get("shortName", tkr),
            })
        except Exception as e:  # network hiccups, rate limits, etc.
            print(f"  [warn] metadata failed for {tkr}: {e}")
            rows.append({"ticker": tkr, "sector": "Unknown",
                         "industry": "Unknown", "name": tkr})
        if i % 20 == 0:
            print(f"  ... {i}/{len(tickers)}")
            time.sleep(1)  # be nice to yfinance
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--universe", choices=["ndx100", "sp500"], default="ndx100")
    p.add_argument("--tickers", nargs="+", default=None,
                   help="Explicit ticker list (overrides --universe).")
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--end", default=date.today().isoformat())
    p.add_argument("--out_dir", default=RAW_DIR)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    tickers = args.tickers or fetch_universe_tickers(args.universe)
    print(f"Universe: {args.universe if not args.tickers else 'custom'} "
          f"({len(tickers)} tickers)")

    meta = fetch_metadata(tickers)
    panel = download_ohlcv(tickers, args.start, args.end)

    panel = panel.merge(meta[["ticker", "sector", "industry"]], on="ticker", how="left")
    panel = panel[["date", "ticker", "sector", "industry",
                   "open", "high", "low", "close", "adj_close", "volume"]]
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Download-summary stats per ticker
    counts = panel.groupby("ticker").size().rename("n_rows").reset_index()
    meta = meta.merge(counts, on="ticker", how="left").fillna({"n_rows": 0})
    meta["n_rows"] = meta["n_rows"].astype(int)
    meta["status"] = meta["n_rows"].apply(lambda n: "ok" if n > 0 else "no_data")

    panel_path = os.path.join(args.out_dir, "universe.parquet")
    meta_path = os.path.join(args.out_dir, "universe_metadata.csv")
    panel.to_parquet(panel_path, index=False)
    meta.to_csv(meta_path, index=False)

    print("\n" + "=" * 60)
    print(f"Saved {len(panel):,} rows across {panel['ticker'].nunique()} tickers")
    print(f"Date range: {panel['date'].min()} -> {panel['date'].max()}")
    print(f"Sectors: {meta['sector'].value_counts().to_dict()}")
    print(f"  panel -> {panel_path}")
    print(f"  meta  -> {meta_path}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
