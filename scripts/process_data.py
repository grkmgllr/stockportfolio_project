import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

def process_data(ticker):
    """
    load the raw data, handles missing values and normalizes features.
    """
    file_path = f"{RAW_DATA_DIR}/{ticker}.csv"
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    df = pd.read_csv(file_path)
    """
    1. Handling Missing Values:
    we are using the forward fill method to handle missing values that 
    if data is missing (e.g., holiday), assume that the price hasn't changed.
    """
    df = df.ffill().bfill()
    
    """
    2. Feature Selection:
    we need OHLCV as per proposal 
    """
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # ensure columns exist
    if not all(col in df.columns for col in features):
        print(f"Missing columns in {ticker}. Skipping.")
        return

    """
    3. Normalization:
    we use StandardScaler to center data around 0 with std dev 1
    """
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    
    # save the processed data
    # we keep the Date column for reference, but the model will only see features
    save_path = f"{PROCESSED_DATA_DIR}/{ticker}_processed.csv"
    df.to_csv(save_path, index=False)
    print(f"Processed and Normalized: {save_path}")

if __name__ == "__main__":
    # process all CSVs found in the raw folder
    raw_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.csv')]
    
    for file in raw_files:
        ticker_name = file.replace(".csv", "")
        process_data(ticker_name)