import yfinance as yf
import pandas as pd
import os
import argparse

RAW_DATA_DIR = "data/raw"
os.makedirs(RAW_DATA_DIR, exist_ok=True)

def fetch_data(tickers, start_date, end_date):
    """
    This function fetches daily OHLCV data from Yahoo Finance and saves it to a csv file.
    """
    print(f"Fetching data for {tickers} from {start_date} to {end_date}")

    # download data
    # group the data by 'ticker'
    data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')

    if data.empty:
        print("No data fetched. Check the ticker symbols or internet connection.")
        return

    # save individual csv files for each ticker
    for ticker in tickers:
        print("Processing {ticker} data...")
        try:
            # extract the data for the current ticker
            df = data[ticker].copy()

            # drop the rows with missing values that identified as NaN -> market close day
            df.dropna(how='all', inplace=True)

            # reset indext to make date a column
            df.reset_index(inplace=True)
            
            # save data to csv file
            save_path = f"{RAW_DATA_DIR}/{ticker}.csv"
            df.to_csv(save_path, index=False)
            print(f"-> saved to {save_path}")

        except Exception as e:
            print(f"Error processing {ticker} data: {e}")

if __name__ == "__main__":
    """
    we can identify the portfolio companies here in the ticker list 
    and select the start and end dates
    """
    TICKER_LIST = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "JPM", "BAC", "WFC"]
    START_DATE = "2023-01-01"
    END_DATE = "2025-11-30" 

    fetch_data(TICKER_LIST, START_DATE, END_DATE)          