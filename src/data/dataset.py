"""
Dataset adapters over the canonical + views panel.

Three modes, one file:

    CrossStockPanelDataset   → [B, N, L, F] + mask, target [B, N, H]
                                for StockMixer, MASTER
    PerSeriesPanelDataset    → [B, L, F], target [B, H]
                                for TimesNet, TimeMixer
    tabular_view(...)        → flat pandas DataFrame
                                for LightGBM

Design notes
------------
- The panel is pre-normalised (see `views.py`). Each mode picks a view:
    * cross-sectional uses `f_cs` columns (MASTER-style relative ranking)
    * per-series uses `f_ps` columns (train-fitted per-ticker z-score)
    * tabular can pick either or raw
- The `available` mask travels with cross-sectional batches so downstream
  attention layers (MASTER) can zero out pre-IPO / delisted cells.
- Target is next-`pred_len` daily log returns per ticker, taken from the
  panel's already-computed `log_return` column. Simple, deterministic,
  identical across modes.
"""
from __future__ import annotations

import os
from typing import List, Literal, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

DEFAULT_PANEL = "data/processed/views_panel.parquet"

DEFAULT_FEATURES = [
    "log_return",
    "daily_range",
    "intraday_move",
    "upper_wick",
    "lower_wick",
    "overnight_gap",
    "log_volume",
]


def _feature_columns(features: List[str], view: str) -> List[str]:
    """Suffix feature names with the chosen view (`_ps`, `_cs`, or raw)."""
    if view == "raw":
        return list(features)
    if view in ("ps", "cs"):
        return [f"{f}_{view}" for f in features]
    raise ValueError(f"Unknown view: {view!r} (expected 'ps', 'cs', or 'raw')")


def _load_panel(panel_path: str, split: Optional[str]) -> pd.DataFrame:
    if not os.path.exists(panel_path):
        raise FileNotFoundError(
            f"Views panel not found: {panel_path}\n"
            f"Run `python src/data/views.py` first."
        )
    df = pd.read_parquet(panel_path)
    df["date"] = pd.to_datetime(df["date"])
    if split is not None:
        df = df[df["split"] == split].reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────
# Cross-sectional (StockMixer, MASTER)
# ─────────────────────────────────────────────────────────────────────
class CrossStockPanelDataset(Dataset):
    """
    Yields joint (all-tickers) windows over the canonical panel.

    Each sample:
        x        : FloatTensor [N, L, F]        — features, view-normalised
        y        : FloatTensor [N, H]           — per-ticker daily log returns
        mask_x   : BoolTensor  [N, L]           — True where ticker traded
        mask_y   : BoolTensor  [N, H]           — True where target is finite

    Missing feature values are filled with 0 *after* the mask is captured,
    so downstream models can safely multiply the input by the mask instead
    of dealing with NaNs.
    """

    def __init__(
        self,
        panel_path: str = DEFAULT_PANEL,
        split: Literal["train", "val", "test"] = "train",
        seq_len: int = 30,
        pred_len: int = 5,
        features: Optional[List[str]] = None,
        view: Literal["cs", "ps", "raw"] = "cs",
        target: str = "log_return",
        min_available: int = 50,
    ):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.features = list(features or DEFAULT_FEATURES)
        self.view = view
        self.target = target

        # Load the panel WITHOUT split filter — we need contiguous dates
        # around the split boundary to form windows that end inside the
        # split. Filter samples by end-date at __getitem__ construction.
        full = _load_panel(panel_path, split=None)
        feat_cols = _feature_columns(self.features, view)

        # Pivot to (T, N) grids per feature — memory-cheap for 4k×100.
        dates = pd.Index(sorted(full["date"].unique()), name="date")
        tickers = sorted(full["ticker"].unique())
        self.dates = dates
        self.tickers = tickers
        T, N, F = len(dates), len(tickers), len(feat_cols)

        # Reindex to guaranteed (date, ticker) order
        idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
        wide = full.set_index(["date", "ticker"]).reindex(idx)

        # Features tensor: (T, N, F)
        feat_arr = wide[feat_cols].to_numpy(dtype=np.float32).reshape(T, N, F)
        avail = wide["available"].to_numpy(dtype=bool).reshape(T, N)
        target_arr = wide[target].to_numpy(dtype=np.float32).reshape(T, N)
        split_series = wide["split"].to_numpy().reshape(T, N)[:, 0]  # split is per-date

        # NaNs in features (pre-IPO cells, tickers without _ps stats) → 0
        # Mask carries the "this was fake" information.
        finite_feat = np.isfinite(feat_arr)
        feat_arr = np.where(finite_feat, feat_arr, 0.0).astype(np.float32)
        # A cell counts as available only if the ticker traded AND all
        # requested features are finite for that (t, ticker).
        avail = avail & finite_feat.all(axis=-1)

        # Target finiteness (last row per ticker etc.)
        finite_tgt = np.isfinite(target_arr)
        target_arr = np.where(finite_tgt, target_arr, 0.0).astype(np.float32)

        self._feat = feat_arr
        self._avail = avail
        self._target = target_arr
        self._target_finite = finite_tgt

        # Determine which sample indices land inside the requested split.
        # Sample i uses input [i, i+L) and target [i+L, i+L+H). Anchor the
        # sample to its LAST INPUT DATE (i+L-1) and require that to be in
        # the split. Also require enough dates on both sides + at least
        # `min_available` tickers throughout the input window.
        idxs = []
        for i in range(T - seq_len - pred_len + 1):
            last_input = i + seq_len - 1
            if split_series[last_input] != split:
                continue
            window_mask = self._avail[i:i + seq_len]  # (L, N)
            if window_mask.sum(axis=1).min() < min_available:
                continue
            idxs.append(i)
        self._sample_starts = np.asarray(idxs, dtype=np.int64)

    def __len__(self) -> int:
        return len(self._sample_starts)

    def __getitem__(self, index: int):
        i = int(self._sample_starts[index])
        L, H = self.seq_len, self.pred_len

        x = self._feat[i:i + L]                   # (L, N, F)
        x = np.transpose(x, (1, 0, 2))            # (N, L, F)
        mask_x = self._avail[i:i + L].T           # (N, L)

        y = self._target[i + L:i + L + H].T       # (N, H)
        mask_y = (self._avail[i + L:i + L + H].T
                  & self._target_finite[i + L:i + L + H].T)

        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(mask_x),
            torch.from_numpy(mask_y),
        )


# ─────────────────────────────────────────────────────────────────────
# Per-series (TimesNet, TimeMixer)
# ─────────────────────────────────────────────────────────────────────
class PerSeriesPanelDataset(Dataset):
    """
    Sliding windows per ticker, concatenated across tickers.

    Each sample:
        x : FloatTensor [L, F]
        y : FloatTensor [H]         (target series, e.g. log_return)
        ticker_id : LongTensor []   (index into self.tickers)

    Only contiguous `available=True` runs within the requested split are
    used, so pre-IPO padding never leaks into a window.
    """

    def __init__(
        self,
        panel_path: str = DEFAULT_PANEL,
        split: Literal["train", "val", "test"] = "train",
        seq_len: int = 30,
        pred_len: int = 5,
        features: Optional[List[str]] = None,
        view: Literal["ps", "cs", "raw"] = "ps",
        target: str = "log_return",
        tickers: Optional[List[str]] = None,
    ):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.features = list(features or DEFAULT_FEATURES)
        self.view = view
        self.target = target

        df = _load_panel(panel_path, split=split)
        if tickers is not None:
            df = df[df["ticker"].isin(tickers)].reset_index(drop=True)
        feat_cols = _feature_columns(self.features, view)

        self.tickers = sorted(df["ticker"].unique())
        self._ticker_to_id = {t: i for i, t in enumerate(self.tickers)}

        samples = []  # list of (feat[L,F], target[H], ticker_id)
        for tkr, g in df.groupby("ticker", sort=False):
            g = g.sort_values("date")
            avail = g["available"].to_numpy()
            feat = g[feat_cols].to_numpy(dtype=np.float32)
            tgt = g[target].to_numpy(dtype=np.float32)
            # Also require finite features + target
            finite = np.isfinite(feat).all(axis=1) & np.isfinite(tgt)
            usable = avail & finite

            L, H = seq_len, pred_len
            n = len(g)
            for i in range(n - L - H + 1):
                if not usable[i:i + L + H].all():
                    continue
                samples.append((
                    feat[i:i + L],
                    tgt[i + L:i + L + H],
                    self._ticker_to_id[tkr],
                ))
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int):
        x, y, tid = self._samples[index]
        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.tensor(tid, dtype=torch.long),
        )


# ─────────────────────────────────────────────────────────────────────
# Tabular (LightGBM)
# ─────────────────────────────────────────────────────────────────────
def tabular_view(
    panel_path: str = DEFAULT_PANEL,
    split: Optional[str] = None,
    features: Optional[List[str]] = None,
    view: Literal["raw", "ps", "cs"] = "raw",
    target: str = "fwd_return_5d",
    drop_unavailable: bool = True,
) -> pd.DataFrame:
    """
    Return a flat DataFrame ready for LightGBM: identity + features + target.

    Unlike the two Dataset classes, this does NOT build sliding windows —
    each row is one (date, ticker) sample. That matches how the legacy
    LightGBM pipeline is trained (row = observation).
    """
    df = _load_panel(panel_path, split=split)
    feat_cols = _feature_columns(list(features or DEFAULT_FEATURES), view)
    cols = ["date", "ticker", "sector", "industry", "split", "available"] + feat_cols + [target]
    out = df[cols].copy()
    if drop_unavailable:
        out = out[out["available"]].reset_index(drop=True)
    out = out.dropna(subset=feat_cols + [target]).reset_index(drop=True)
    return out
