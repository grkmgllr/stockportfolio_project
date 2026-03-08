"""
Resample minute-bar parquet data to daily OHLCV + VWAP + Transactions.

Reads a high-resolution parquet file (e.g. minute bars from Polygon.io),
filters to regular US market hours (09:30-16:00 ET), resamples to daily
frequency, and removes weekends and holidays to produce clean daily bars
compatible with the ``ParquetDataset`` pipeline.

Data-cleaning pipeline
----------------------
    1.  Filter to ``start_date`` onward.
    2.  Keep only Regular Trading Hours (09:30-16:00 ET).
    3.  Resample to daily bars using standard aggregation rules.
    4.  Drop weekends (Saturday / Sunday).
    5.  Drop holidays and partial sessions via a volume-based filter:
        days with volume < ``min_volume_pct`` of the rolling median
        are discarded.

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


MARKET_OPEN = "09:30"
MARKET_CLOSE = "16:00"

MIN_VOLUME_PCT = 0.05


def _filter_market_hours(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only bars within US Regular Trading Hours (09:30-16:00 ET).

    Assumes the ``datetime`` index is already in US/Eastern or a
    UTC-naive representation of Eastern time (as is typical of
    Polygon.io minute-bar exports).

    Args:
        df (pd.DataFrame):
            Minute-bar DataFrame indexed by ``datetime``.

    Returns:
        pd.DataFrame:
            Filtered DataFrame containing only RTH bars.
    """
    time_idx = df.index.time
    rth_mask = (
        (time_idx >= pd.Timestamp(MARKET_OPEN).time()) &
        (time_idx < pd.Timestamp(MARKET_CLOSE).time())
    )
    filtered = df[rth_mask]
    n_dropped = len(df) - len(filtered)
    print(f"  Filtered to RTH ({MARKET_OPEN}-{MARKET_CLOSE}): "
          f"dropped {n_dropped:,} pre/post-market bars "
          f"({n_dropped / len(df) * 100:.1f}%)")
    return filtered


def _drop_weekends(daily: pd.DataFrame) -> pd.DataFrame:
    """Remove Saturday (5) and Sunday (6) rows."""
    mask = daily.index.dayofweek < 5
    n_dropped = (~mask).sum()
    if n_dropped:
        print(f"  Dropped {n_dropped} weekend rows")
    return daily[mask]


def _drop_low_volume_days(
    daily: pd.DataFrame,
    min_pct: float = MIN_VOLUME_PCT,
) -> pd.DataFrame:
    """
    Remove days whose volume is abnormally low (holidays, half-days).

    Uses a 21-day rolling median as the baseline.  Any day with volume
    below ``min_pct`` (default 5%) of that median is dropped.

    Args:
        daily (pd.DataFrame):
            Daily-bar DataFrame with a ``volume`` column.
        min_pct (float):
            Minimum fraction of rolling-median volume to keep.

    Returns:
        pd.DataFrame:
            Cleaned DataFrame with low-volume days removed.
    """
    rolling_median = daily["volume"].rolling(
        window=21, min_periods=5, center=True,
    ).median()
    threshold = rolling_median * min_pct
    low_vol_mask = daily["volume"] < threshold
    n_dropped = low_vol_mask.sum()
    if n_dropped:
        dropped_dates = daily.index[low_vol_mask].strftime("%Y-%m-%d").tolist()
        print(f"  Dropped {n_dropped} low-volume days (holidays/partial): "
              f"{dropped_dates[:10]}{'...' if n_dropped > 10 else ''}")
    return daily[~low_vol_mask]


def resample_minute_to_daily(
    df: pd.DataFrame,
    start_date: str = "2021-01-01",
) -> pd.DataFrame:
    """
    Resample minute-bar data to clean daily bars.

    Pipeline: date filter -> RTH filter -> daily resample -> drop
    weekends -> drop low-volume holidays.

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

    # Step 1: Keep only Regular Trading Hours
    df = _filter_market_hours(df)

    if df.empty:
        raise ValueError("No bars remain after RTH filtering.")

    # Step 2: Resample to daily bars
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

    if "vwap" in df.columns:
        dollar_volume = (df["vwap"] * df["volume"]).resample("D").sum()
        total_volume = df["volume"].resample("D").sum()
        daily["vwap"] = dollar_volume / total_volume.replace(0, np.nan)

    # Drop rows with no bars at all (NaN open means no trades)
    daily = daily.dropna(subset=["open"])
    daily = daily[daily["volume"] > 0]

    # Step 3: Drop weekends
    daily = _drop_weekends(daily)

    # Step 4: Drop holidays / abnormally low-volume days
    daily = _drop_low_volume_days(daily)

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
    print("Minute-Bar to Daily Resampling (RTH-only)")
    print("=" * 60)
    print(f"Ticker:       {args.ticker}")
    print(f"Start date:   {args.start_date}")
    print(f"Market hours: {MARKET_OPEN} - {MARKET_CLOSE}")
    print("=" * 60 + "\n")

    # Locate parquet
    if args.input_path:
        parquet_path = args.input_path
    else:
        parquet_path = find_parquet_file(args.data_root, args.ticker)

    print(f"Reading: {parquet_path}")
    df_raw = pd.read_parquet(parquet_path)
    print(f"  Loaded {len(df_raw):,} minute bars")
    print(f"  Raw date range: {df_raw['datetime'].min()} to {df_raw['datetime'].max()}")

    # Resample
    df_daily = resample_minute_to_daily(df_raw, start_date=args.start_date)

    # Final summary
    n_days = len(df_daily)
    print(f"\n{'=' * 60}")
    print(f"RESULT: {n_days} clean trading days")
    print(f"  Date range: {df_daily['Date'].iloc[0]} to {df_daily['Date'].iloc[-1]}")
    print(f"  Columns:    {list(df_daily.columns)}")
    print(f"  Avg volume: {df_daily['Volume'].mean():,.0f}")

    # Sanity checks
    dates = pd.to_datetime(df_daily["Date"])
    weekend_count = (dates.dt.dayofweek >= 5).sum()
    print(f"\n  Sanity checks:")
    print(f"    Weekend rows: {weekend_count} (should be 0)")
    print(f"    NaN values:   {df_daily.isnull().sum().sum()} (should be 0)")
    print(f"    Min volume:   {df_daily['Volume'].min():,.0f}")
    print(f"{'=' * 60}")

    # Save
    out_path = os.path.join(args.data_root, f"{args.ticker}.csv")
    df_daily.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

    # Quick sample
    print(f"\nFirst 5 rows:")
    print(df_daily.head().to_string(index=False))
    print(f"\nLast 5 rows:")
    print(df_daily.tail().to_string(index=False))
    print()


if __name__ == "__main__":
    main()
