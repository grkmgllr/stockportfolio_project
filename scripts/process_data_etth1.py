import os
import pandas as pd
import numpy as np

RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
SRC_PATH = os.path.join(RAW_DATA_DIR, "ETTh1.csv")
OUT_PATH = os.path.join(PROCESSED_DATA_DIR, "ETTh1_processed.csv")

def process_etth1():
    if not os.path.exists(SRC_PATH):
        print(f"File not found: {SRC_PATH}")
        return

    df = pd.read_csv(SRC_PATH)

    # ensuring we have a proper datetime column named exactly 'date' 
    date_cols = [c for c in df.columns if c.lower() in ("date", "time", "timestamp")]
    date_col = date_cols[0] if date_cols else None

    if date_col is not None:
        # normalize column name to 'date' for consistency
        if date_col != "date":
            df = df.rename(columns={date_col: "date"})
        try:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        except Exception:
            pass
        # drop rows with invalid date to avoid breaking time features later
        if "date" in df.columns:
            df = df.dropna(subset=["date"])
            df = df.sort_values("date").reset_index(drop=True)

    # keeping only numeric feature columns
    # NOTE: NOT standardized here.
    # The Dataset_ETT_hour / data_loader will standardize using TRAIN split only.
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        print("No numeric columns found to process.")
        return

    # handling missing values: forward-fill then back-fill
    df[numeric_cols] = df[numeric_cols].ffill().bfill()

    # ensuring numeric columns are float (avoids dtype issues in torch)
    df[numeric_cols] = df[numeric_cols].astype(float)

    # drop duplicate timestamps if any
    if "date" in df.columns:
        df = df.drop_duplicates(subset=["date"], keep="last")

    # save cleaned (but UNSCALED) data
    df.to_csv(OUT_PATH, index=False)
    print(f"Processed saved (cleaned, unscaled): {OUT_PATH}")

if __name__ == "__main__":
    process_etth1()