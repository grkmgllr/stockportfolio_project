import os
import urllib.request

RAW_DATA_DIR = "data/raw"
os.makedirs(RAW_DATA_DIR, exist_ok=True)

ETTH1_URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/refs/heads/main/ETT-small/ETTh1.csv"
SAVE_PATH = os.path.join(RAW_DATA_DIR, "ETTh1.csv")

def fetch_etth1(overwrite=False):
    if os.path.exists(SAVE_PATH) and not overwrite:
        print(f"ETTh1 already exists: {SAVE_PATH}")
        return
    try:
        print("Downloading ETTh1...")
        urllib.request.urlretrieve(ETTH1_URL, SAVE_PATH)
        print(f"Saved: {SAVE_PATH}")
    except Exception as e:
        print(f"Download failed: {e}")

if __name__ == "__main__":
    fetch_etth1()