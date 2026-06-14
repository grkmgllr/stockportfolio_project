"""
Daily OHLCV downloader from Yahoo Finance.

Downloads daily bars for a list of tickers and saves one CSV per ticker
into ``data/raw/``. For minute-bar data from Polygon.io, use
``resample_parquet.py`` instead.
"""
import yfinance as yf
import pandas as pd
import os
import argparse

RAW_DATA_DIR = "data/raw"
os.makedirs(RAW_DATA_DIR, exist_ok=True)


def fetch_data(tickers, start_date, end_date):
    """Fetch daily OHLCV from Yahoo Finance and save one CSV per ticker."""
    print(f"Fetching data for {tickers} from {start_date} to {end_date}")

    # batch-download all tickers in one call
    data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')

    if data.empty:
        print("No data fetched. Check the ticker symbols or internet connection.")
        return

    for ticker in tickers:
        print(f"Processing {ticker} data...")
        try:
            df = data[ticker].copy()
            # drop rows where all values are NaN (market holidays)
            df.dropna(how='all', inplace=True)
            # make Date a regular column instead of index
            df.reset_index(inplace=True)

            save_path = f"{RAW_DATA_DIR}/{ticker}.csv"
            df.to_csv(save_path, index=False)
            print(f"-> saved to {save_path}")

        except Exception as e:
            print(f"Error processing {ticker} data: {e}")


if __name__ == "__main__":
    TICKER_LIST = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "JPM", "BAC", "WFC"]
    START_DATE = "2023-01-01"
    END_DATE = "2025-11-30"

    fetch_data(TICKER_LIST, START_DATE, END_DATE)
