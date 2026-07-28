"""Diagnostic: is the SMA5-anchor IC gain real forecasting, or a trivial
known-at-t component leaking into the target?

Decompose  target = (mean(Close[t+1..t+5]) - SMA5[t]) / SMA5[t]
                   = pure_future + known_term
  pure_future = (mean(future) - Close[t]) / Close[t]   <- tradeable, unknown at t
  known_term  = (Close[t]      - SMA5[t]) / SMA5[t]     <- KNOWN at t (a feature)

Report:
  IC(pred, target)       headline (~0.57 expected)
  IC(pred, pure_future)  the ONLY tradeable signal
  IC(known, target)      how much a trivial 'predict the known part' scores
  IC(pred, known)        is the model mostly echoing the known part?
"""
import os, sys
import numpy as np, pandas as pd
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
import lightgbm as lgb
from models.LightGBMForecaster import LightGBMForecaster

TICKERS = open(os.path.join(REPO, "experiments/tickers.txt")).read().split()
DATA_ROOT = os.path.join(REPO, "data/raw")
SEQ, PRED = 30, 5
START, TREND, VEND = "2015-01-01", "2023-11-24", "2024-05-28"
P = dict(objective="regression", n_estimators=2000, learning_rate=0.01, num_leaves=15,
         max_depth=5, min_child_samples=20, subsample=0.7, colsample_bytree=0.7,
         reg_alpha=1.0, reg_lambda=1.0, random_state=42, verbose=-1, n_jobs=-1)
_fe = LightGBMForecaster(seq_len=SEQ, pred_len=PRED).engineer_features

def ic(a, b):
    a, b = a.ravel(), b.ravel(); m = np.isfinite(a) & np.isfinite(b)
    return np.corrcoef(a[m], b[m])[0, 1]

rows = {"tr": [], "te": []}
for t in TICKERS:
    df = pd.read_csv(f"{DATA_ROOT}/{t}.csv"); df = df[df["Date"] >= START].reset_index(drop=True).ffill()
    c = df["Close"].values.astype(float); d = df["Date"].values
    X = np.nan_to_num(_fe(df).values.astype(float))
    sma5 = pd.Series(c).rolling(5, min_periods=5).mean().values
    for i in range(SEQ, len(df) - PRED):
        if not np.isfinite(sma5[i]) or sma5[i] <= 0: continue
        favg = c[i+1:i+1+PRED].mean()
        target = (favg - sma5[i]) / sma5[i]
        pure  = (favg - c[i]) / c[i]
        known = (c[i] - sma5[i]) / sma5[i]
        rec = (X[i], target, pure, known)
        rows["tr" if d[i] <= TREND else ("te" if d[i] > VEND else "sk")].append(rec) if d[i] <= TREND or d[i] > VEND else None

Xtr = np.stack([r[0] for r in rows["tr"]]); ytr = np.array([r[1] for r in rows["tr"]])
Xte = np.stack([r[0] for r in rows["te"]])
tgt = np.array([r[1] for r in rows["te"]]); pure = np.array([r[2] for r in rows["te"]]); known = np.array([r[3] for r in rows["te"]])
m = lgb.LGBMRegressor(**P); m.fit(Xtr, ytr)
pred = m.predict(Xte)

print(f"\n{'='*56}\n  SMA5-ANCHOR IC DECOMPOSITION (test n={len(tgt)})\n{'='*56}")
print(f"  IC(pred, target)      = {ic(pred, tgt):.3f}   <- manset")
print(f"  IC(pred, pure_future) = {ic(pred, pure):.3f}   <- TEK tradeable sinyal")
print(f"  IC(known, target)     = {ic(known, tgt):.3f}   <- trivial 'bilineni tahmin' skoru")
print(f"  IC(pred, known)       = {ic(pred, known):.3f}   <- model bilineni mi tekrarliyor?")
print(f"\n  var(target)={np.var(tgt):.2e}  var(known)={np.var(known):.2e}  var(pure)={np.var(pure):.2e}")
print(f"  known payi = var(known)/var(target) = {np.var(known)/np.var(tgt)*100:.0f}%")
