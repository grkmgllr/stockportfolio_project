"""
Resample minute-bar parquet data to daily OHLCV + VWAP + Transactions.

Reads a high-resolution parquet file (e.g. minute bars from Polygon.io),
filters to a configurable start date, resamples to daily frequency using
standard market aggregation rules, and saves the result as a CSV that is
directly compatible with the existing ``ParquetDataset`` pipeline.

Aggregation rules
-----------------
    Open         - first bar's open of the trading day
    High         - max of all intraday highs
    Low          - min of all intraday lows
    Close        - last bar's close of the trading day
    Volume       - sum of all intraday volumes
    Vwap         - volume-weighted average:  sum(vwap * volume) / sum(volume)
    Transactions - sum of all intraday transactions

Usage
-----
    python scripts/resample_parquet.py --ticker AAPL
    python scripts/resample_parquet.py --ticker AAPL --start_date 2021-01-01
    python scripts/resample_parquet.py --ticker AAPL --input_path data/raw/custom.parquet
"""

import pandas as pd
import numpy as np
import os
import argparse
import glob


def resample_minute_to_daily(
    df: pd.DataFrame,
    start_date: str = "2021-01-01",
) -> pd.DataFrame:
    """
    Resample minute-bar data to daily bars.

    The input DataFrame is expected to have a ``datetime`` column and
    lowercase OHLCV columns (``open``, ``high``, ``low``, ``close``,
    ``volume``), plus optional ``vwap`` and ``transactions`` columns.

    Args:
        df (pd.DataFrame):
            Minute-bar DataFrame with a ``datetime`` column.
        start_date (str):
            ISO date string.  Rows before this date are discarded.

    Returns:
        pd.DataFrame:
            Daily-bar DataFrame with capitalized column names matching
            the ``ParquetDataset`` convention: ``Date``, ``Open``, ``High``,
            ``Low``, ``Close``, ``Volume``, ``Vwap``, ``Transactions``.
    """
    df = df.copy()

    if "datetime" not in df.columns:
        raise ValueError("Expected a 'datetime' column in the DataFrame.")

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df[df["datetime"] >= pd.Timestamp(start_date)]

    if df.empty:
        raise ValueError(f"No data remaining after filtering to >= {start_date}")

    df = df.set_index("datetime").sort_index()

    agg_rules = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }

    if "transactions" in df.columns:
        agg_rules["transactions"] = "sum"

    daily = df.resample("D").agg(agg_rules)

    # VWAP requires a weighted average: sum(vwap * volume) / sum(volume)
    if "vwap" in df.columns:
        dollar_volume = (df["vwap"] * df["volume"]).resample("D").sum()
        total_volume = df["volume"].resample("D").sum()
        daily["vwap"] = dollar_volume / total_volume.replace(0, np.nan)

    # Drop non-trading days (weekends / holidays with no bars)
    daily = daily.dropna(subset=["open"])
    daily = daily[daily["volume"] > 0]

    # Capitalize column names to match ParquetDataset convention
    daily = daily.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
        "vwap": "Vwap",
        "transactions": "Transactions",
    })

    daily.index.name = "Date"
    daily = daily.reset_index()

    return daily


def find_parquet_file(data_root: str, ticker: str) -> str:
    """
    Locate the parquet file for a given ticker in the data directory.

    Searches for files matching ``{ticker}_*.parquet`` (e.g.
    ``AAPL_minute_bars_2015-09-30_2025-09-30.parquet``).

    Args:
        data_root (str):
            Directory to search in.
        ticker (str):
            Stock ticker symbol.

    Returns:
        str:
            Full path to the parquet file.

    Raises:
        FileNotFoundError:
            If no matching parquet file is found.
    """
    pattern = os.path.join(data_root, f"{ticker}_*.parquet")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(
            f"No parquet file matching '{pattern}' found.\n"
            f"Place your minute-bar parquet file in {data_root}/ with "
            f"the naming convention {{TICKER}}_*.parquet"
        )
    matches.sort(key=os.path.getmtime, reverse=True)
    return matches[0]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Resample minute-bar parquet data to daily CSV."
    )
    parser.add_argument("--ticker", type=str, default="AAPL")
    parser.add_argument("--start_date", type=str, default="2021-01-01")
    parser.add_argument("--data_root", type=str, default="data/raw")
    parser.add_argument(
        "--input_path", type=str, default=None,
        help="Explicit path to the parquet file (overrides auto-detection).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("Minute-Bar to Daily Resampling")
    print("=" * 60)
    print(f"Ticker:     {args.ticker}")
    print(f"Start date: {args.start_date}")
    print("=" * 60 + "\n")

    # Locate parquet
    if args.input_path:
        parquet_path = args.input_path
    else:
        parquet_path = find_parquet_file(args.data_root, args.ticker)

    print(f"Reading: {parquet_path}")
    df_raw = pd.read_parquet(parquet_path)
    print(f"  Loaded {len(df_raw):,} minute bars")

    # Resample
    df_daily = resample_minute_to_daily(df_raw, start_date=args.start_date)
    print(f"  Resampled to {len(df_daily)} daily bars")
    print(f"  Date range: {df_daily['Date'].iloc[0]} to {df_daily['Date'].iloc[-1]}")
    print(f"  Columns: {list(df_daily.columns)}")

    # Save
    out_path = os.path.join(args.data_root, f"{args.ticker}.csv")
    df_daily.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

    # Quick summary
    print(f"\nSample (first 5 rows):")
    print(df_daily.head().to_string(index=False))
    print(f"\nSample (last 5 rows):")
    print(df_daily.tail().to_string(index=False))
    print()


if __name__ == "__main__":
    main()
