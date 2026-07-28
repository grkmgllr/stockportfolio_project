"""
================================================================
 STAGE-1 FORECASTER — locked system design (LightGBM iteration)
================================================================
 DATA    : 78 NDX-100 tickers, daily OHLCV 2015+, ffill only (NO bfill leak)
 INPUT   : current LightGBM feature set (28, scale-invariant), same for all
 OUTPUT  : 2 channels, forward-averaged, Close[t] anchor, vol-normalized
             up = (mean(High[t+1..t+5]) - Close[t]) / Close[t] / sigma[t]
             dn = (mean(Low [t+1..t+5]) - Close[t]) / Close[t] / sigma[t]
           sigma[t] = volatility_20 (20d log-ret std, known at t)
           -> feeds triple-barrier (upper=High, lower=Low) in Stage-2
 POOLING : build (X,y) INSIDE each ticker, then concatenate (no cross leak)
 SPLIT   : single train/val/test by date (val used for early stopping)
 MODEL   : LightGBM (one regressor per channel)
 EVAL    : IC + non-overlap IC + DA/up% + naive climatology baseline
 usage   : python run_v3.py on|off        (vol-normalization)
================================================================
"""
from __future__ import annotations
import os, sys, time
import numpy as np, pandas as pd
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
import lightgbm as lgb
from models.LightGBMForecaster import LightGBMForecaster

# ---- locked config --------------------------------------------------------
TICKERS   = open(os.path.join(REPO, "experiments/tickers.txt")).read().split()
DATA_ROOT = os.path.join(REPO, "data/raw")
SEQ, PRED = 30, 5
START, TRAIN_END, VAL_END = "2015-01-01", "2023-11-24", "2024-05-28"
NAIVE_W   = 40          # trailing window (completed outcomes) for climatology baseline
VOLNORM   = (sys.argv[1] if len(sys.argv) > 1 else "on") == "on"
LGB = dict(objective="regression", metric="mae", n_estimators=2000, learning_rate=0.01,
           num_leaves=15, max_depth=5, min_child_samples=20, subsample=0.7,
           colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=1.0, random_state=42,
           verbose=-1, n_jobs=-1)
_fe = LightGBMForecaster(seq_len=SEQ, pred_len=PRED).engineer_features


def ic(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    m = np.isfinite(a) & np.isfinite(b)
    return np.corrcoef(a[m], b[m])[0, 1] if m.sum() > 3 else np.nan


def build(t):
    """Per-ticker (date, X, up, dn, naive_up, naive_dn) rows. All causal."""
    df = pd.read_csv(f"{DATA_ROOT}/{t}.csv")
    df = df[df["Date"] >= START].reset_index(drop=True).ffill()      # ffill only
    c = df["Close"].values.astype(float); h = df["High"].values.astype(float)
    lo = df["Low"].values.astype(float); sig = df["volatility_20"].values.astype(float)
    d = df["Date"].values
    X = np.nan_to_num(_fe(df).values.astype(float))
    T = len(df)

    # full target series (NaN where invalid) so we can build a causal climatology
    up = np.full(T, np.nan); dn = np.full(T, np.nan)
    for i in range(SEQ, T - PRED):
        s = sig[i]
        if not np.isfinite(s) or s <= 0:
            continue
        u = (h[i+1:i+1+PRED].mean() - c[i]) / c[i]
        v = (lo[i+1:i+1+PRED].mean() - c[i]) / c[i]
        if VOLNORM:
            u, v = u / s, v / s
        up[i], dn[i] = u, v

    rows = []
    for i in range(SEQ, T - PRED):
        if not np.isfinite(up[i]):
            continue
        # naive: mean of COMPLETED past outcomes (decision day <= i-PRED)
        lo_j, hi_j = i - PRED - NAIVE_W, i - PRED
        past_u = up[max(SEQ, lo_j):hi_j]; past_d = dn[max(SEQ, lo_j):hi_j]
        pu = np.nanmean(past_u) if np.isfinite(past_u).any() else np.nan
        pv = np.nanmean(past_d) if np.isfinite(past_d).any() else np.nan
        rows.append((d[i], X[i], up[i], dn[i], pu, pv))
    return rows


def main():
    t0 = time.time()
    print(f"\n{'='*60}\n  V3  vol-norm = {'ON' if VOLNORM else 'OFF'}\n{'='*60}")
    TR, VA, TE = [], [], []
    for t in TICKERS:
        for r in build(t):
            d = r[0]
            (TR if d <= TRAIN_END else VA if d <= VAL_END else TE).append(r)

    def cols(rows):
        return (np.stack([r[1] for r in rows]),
                np.array([r[2] for r in rows]), np.array([r[3] for r in rows]),
                np.array([r[4] for r in rows]), np.array([r[5] for r in rows]))
    Xtr, uptr, dntr, _, _ = cols(TR)
    Xva, upva, dnva, _, _ = cols(VA)
    Xte, upte, dnte, nu_te, nd_te = cols(TE)
    print(f"  train={len(TR)} val={len(VA)} test={len(TE)}  | features={Xtr.shape[1]}")

    def fit_pred(ytr, yva):
        m = lgb.LGBMRegressor(**LGB)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)],
              callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
        return m.predict(Xte)
    pu, pd_ = fit_pred(uptr, upva), fit_pred(dntr, dnva)

    idx = np.arange(0, len(TE), PRED)   # non-overlap sanity
    def report(name, pred, true, naive=None):
        da = 100 * np.mean(np.sign(pred) == np.sign(true))
        upr = 100 * np.mean(true > 0)
        mae = np.mean(np.abs(pred - true))
        base = f"  naive_IC={ic(naive,true):6.3f} naive_MAE={np.mean(np.abs(naive-true)):.3f}" if naive is not None else ""
        print(f"  {name:<12} IC={ic(pred,true):6.3f}  IC_no={ic(pred[idx],true[idx]):6.3f}"
              f"  DA={da:4.1f}  up%={upr:4.1f}  MAE={mae:.3f}{base}")

    print(f"\n  {'channel':<12}{'model vs naive-climatology baseline'}")
    report("upside",   pu,       upte,        nu_te)
    report("downside", pd_,      dnte,        nd_te)
    report("net(up+dn)", pu+pd_, upte+dnte)   # directional proxy
    print(f"\n  IC_no=non-overlap. model IC 'naive_IC'i gecmeli (yoksa sadece climatology).")
    print(f"  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
