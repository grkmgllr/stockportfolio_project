"""
Raw-CSV preprocessor: missing-value handling, feature selection, normalization.

Note: the main training pipeline (ParquetDataset) does its own scaling
on the train split only. This script is for standalone exploratory use.
"""
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)


def process_data(ticker):
    """Load raw CSV, fill missing values, normalize OHLCV, save."""
    file_path = f"{RAW_DATA_DIR}/{ticker}.csv"
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    df = pd.read_csv(file_path)

    # forward-fill then back-fill to handle holiday gaps
    df = df.ffill().bfill()

    features = ['Open', 'High', 'Low', 'Close', 'Volume']

    if not all(col in df.columns for col in features):
        print(f"Missing columns in {ticker}. Skipping.")
        return

    # standard-scale features (fits on full CSV — only for exploration)
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    save_path = f"{PROCESSED_DATA_DIR}/{ticker}_processed.csv"
    df.to_csv(save_path, index=False)
    print(f"Processed and Normalized: {save_path}")


if __name__ == "__main__":
    raw_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.csv')]

    for file in raw_files:
        ticker_name = file.replace(".csv", "")
        process_data(ticker_name)
