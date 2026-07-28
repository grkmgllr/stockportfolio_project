"""
Range-mode meta-labeling (Stage-2) — both triple-barrier levels come from the
primary model's own upside/downside predictions.

Flow per ticker:
  1. Load the pooled range LightGBM checkpoint, predict [upside, downside]
     (vol-normalised) on the same date-split test window as Stage-1.
  2. Reconstruct barrier prices off the entry Close:
        upper (take-profit) = Close[t] * (1 + upside   * sigma[t])
        lower (stop-loss)   = Close[t] * (1 + downside * sigma[t])
  3. Triple barrier -> binary meta_label ("would this long be profitable?").
  4. Market-context + range-prediction features.
Pool across tickers (offset event windows so PurgedKFold stays valid), then
train the meta-classifier and report precision before vs after filtering.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

from models.LightGBMForecaster import LightGBMForecaster
from trading_logic.triple_barrier import apply_triple_barrier
from meta.features import add_market_context_features

SEQ, PRED = 30, 5
START, TRAIN_END, VAL_END = "2015-01-01", "2023-11-24", "2024-05-28"

# Market context + range-prediction features fed to the meta-classifier.
FEATURES = ["atr", "rolling_vol", "rsi", "macd_line", "macd_signal", "macd_hist",
            "pred_upside", "pred_downside", "pred_band", "pred_skew"]


def build_ticker(ticker, data_root, fc):
    """Return a labelled, feature-rich frame for one ticker (local t_start/t_end)."""
    df = pd.read_csv(os.path.join(data_root, f"{ticker}.csv"))
    df = df[df["Date"] >= START].reset_index(drop=True).ffill()   # ffill only (no leak)
    val_end = int((df["Date"] <= VAL_END).sum())

    preds, anchor, sigma = fc.predict_range(df)   # positions seq_len..len-PRED
    n_test = (len(df) - val_end) - SEQ - PRED + 1
    if n_test <= 0:
        return None
    s = val_end - 1
    up = preds[s:s + n_test, 0]; dn = preds[s:s + n_test, 1]
    anc = anchor[s:s + n_test]; sig = sigma[s:s + n_test]

    entry_idx = np.arange(n_test) + val_end + SEQ - 1
    d = df.iloc[entry_idx].reset_index(drop=True).copy()
    # de-vol-normalise the returns, then to absolute barrier prices
    up_ret, dn_ret = up * sig, dn * sig
    d["pred_upper"] = anc * (1.0 + up_ret)
    d["pred_lower"] = anc * (1.0 + dn_ret)

    d = apply_triple_barrier(
        d, pred_high_col="pred_upper", pred_low_col="pred_lower",
        close_col="Close", high_col="High", low_col="Low",
        vertical_barrier_periods=PRED,
    )
    d = add_market_context_features(d)
    d["pred_upside"] = up_ret
    d["pred_downside"] = dn_ret
    d["pred_band"] = up_ret - dn_ret          # expected range width
    d["pred_skew"] = up_ret + dn_ret          # up/down asymmetry (direction proxy)

    # drop warm-up NaN, re-align event windows to the reset index
    survivors = d.dropna(subset=FEATURES + ["meta_label"]).index.values
    d = d.loc[survivors].reset_index(drop=True)
    n_new = len(d)
    new_t_end = np.minimum(np.searchsorted(survivors, d["t_end"].values), n_new - 1)
    d["t_start"] = np.arange(n_new, dtype=np.int64)
    d["t_end"] = new_t_end.astype(np.int64)
    return d


def build_pooled(tickers, data_root, ckpt):
    fc = LightGBMForecaster.load(ckpt)
    frames, offset = [], 0
    for t in tickers:
        d = build_ticker(t, data_root, fc)
        if d is None or len(d) == 0:
            continue
        d = d.copy()
        d["t_start"] += offset
        d["t_end"] += offset
        offset += len(d)
        frames.append(d)
    return pd.concat(frames, ignore_index=True)
