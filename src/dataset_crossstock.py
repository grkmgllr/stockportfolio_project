"""
Cross-stock dataset for StockMixer-style joint forecasting.

Unlike :class:`dataset.ParquetDataset`, each sample here contains ALL tickers
at the same time-window:

    seq_x: [num_stocks, seq_len, n_input_features]
    seq_y: [num_stocks, pred_len, n_target_features]

Tickers are aligned by inner-join on the Date column so every sample has the
same fixed ``num_stocks`` dimension the StockMixer stock-mixing block expects.

When ``return_targets=True`` (default), targets are per-ticker percentage
returns relative to that ticker's last Close in the input window.
"""
from __future__ import annotations

import os
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from dataset import ParquetDataset


class CrossStockDataset(Dataset):
    """Sliding-window dataset yielding all tickers jointly.

    See module docstring for output shapes. Split boundaries are computed on
    the shared (inner-joined) date index so train/val/test are aligned across
    tickers.
    """

    OHLCV_COLUMNS = ParquetDataset.OHLCV_COLUMNS
    EXTENDED_COLUMNS = ParquetDataset.EXTENDED_COLUMNS
    DEFAULT_TARGETS = ParquetDataset.DEFAULT_TARGETS
    MA_CONFIGS = ParquetDataset.MA_CONFIGS

    def __init__(
        self,
        tickers: List[str],
        root_path: str = 'data/raw',
        flag: Literal['train', 'val', 'test'] = 'train',
        seq_len: int = 30,
        pred_len: int = 5,
        input_features: Optional[List[str]] = None,
        target_features: Optional[List[str]] = None,
        scale: bool = True,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        ma_targets: Optional[List[str]] = None,
        return_targets: bool = True,
        start_date: Optional[str] = "2022-01-01",
    ):
        if not tickers or len(tickers) < 2:
            raise ValueError("CrossStockDataset requires at least 2 tickers")

        self.tickers = list(tickers)
        self.root_path = root_path
        self.flag = flag
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.scale = scale
        self.return_targets = return_targets
        self.start_date = start_date
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

        self._input_features_override = input_features
        self.target_features = list(target_features) if target_features else list(self.DEFAULT_TARGETS)

        self.ma_targets = ma_targets or []
        for name in self.ma_targets:
            if name not in self.MA_CONFIGS:
                raise ValueError(
                    f"Unknown MA target '{name}'. Available: {list(self.MA_CONFIGS.keys())}"
                )

        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scalers_x: Dict[str, StandardScaler] = {t: StandardScaler() for t in self.tickers}
        self.scalers_y: Dict[str, StandardScaler] = {t: StandardScaler() for t in self.tickers}

        self._load_data()

    # ------------------------------------------------------------------
    # per-ticker preprocessing
    # ------------------------------------------------------------------
    def _load_one(self, ticker: str) -> pd.DataFrame:
        """Load a single ticker's CSV, apply start_date filter, MA columns,
        and MA warm-up trim. Returns a DataFrame indexed by Date."""
        path = os.path.join(self.root_path, f"{ticker}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Data file not found: {path}\n"
                f"Run resample_parquet.py or fetch_data.py first."
            )

        df = pd.read_csv(path)
        if 'Date' not in df.columns:
            raise ValueError(f"[{ticker}] CSV missing 'Date' column")

        if self.start_date:
            df = df[df['Date'] >= self.start_date].reset_index(drop=True)

        df = df.ffill().bfill()

        for ma_name in self.ma_targets:
            cfg = self.MA_CONFIGS[ma_name]
            df[ma_name] = ParquetDataset._compute_ma(
                df['Close'], cfg['method'], cfg['period'],
            )
        if self.ma_targets:
            max_period = max(self.MA_CONFIGS[n]['period'] for n in self.ma_targets)
            df = df.iloc[max_period - 1:].reset_index(drop=True)

        df = df.set_index('Date')
        return df

    def _resolve_input_features(self, dfs: Dict[str, pd.DataFrame]) -> List[str]:
        """Pick a single input-feature list that works for every ticker.

        If the caller supplied an override, use it as-is. Otherwise use
        EXTENDED columns when ALL tickers have them, else fall back to OHLCV.
        """
        if self._input_features_override is not None:
            for t, df in dfs.items():
                missing = set(self._input_features_override) - set(df.columns)
                if missing:
                    raise ValueError(f"[{t}] missing input columns: {missing}")
            return list(self._input_features_override)

        all_have_extended = all(
            all(c in df.columns for c in self.EXTENDED_COLUMNS) for df in dfs.values()
        )
        return list(self.EXTENDED_COLUMNS) if all_have_extended else list(self.OHLCV_COLUMNS)

    # ------------------------------------------------------------------
    # main assembly
    # ------------------------------------------------------------------
    def _load_data(self):
        raw = {t: self._load_one(t) for t in self.tickers}

        self.input_features = self._resolve_input_features(raw)
        all_targets = list(self.target_features) + [
            n for n in self.ma_targets if n not in self.target_features
        ]
        self.target_features = all_targets
        for tf in self.target_features:
            if tf not in self.input_features and tf not in self.ma_targets:
                raise ValueError(f"Target '{tf}' not in input_features {self.input_features}")

        # inner-join dates
        common = None
        for df in raw.values():
            idx = set(df.index)
            common = idx if common is None else (common & idx)
        common_dates = sorted(common)
        if len(common_dates) < self.seq_len + self.pred_len + 1:
            raise ValueError(
                f"Only {len(common_dates)} common dates across tickers — not enough "
                f"for seq_len={self.seq_len}, pred_len={self.pred_len}."
            )

        # warn on heavy misalignment
        max_len = max(len(df) for df in raw.values())
        drop_ratio = 1.0 - len(common_dates) / max_len
        if drop_ratio > 0.10 and self.flag == 'train':
            print(
                f"[CrossStock] WARN: inner-join dropped {drop_ratio:.1%} of rows "
                f"({max_len} → {len(common_dates)}). Tickers are heavily misaligned."
            )

        # reindex each ticker on the shared calendar and pack tensors
        T = len(common_dates)
        N = len(self.tickers)
        F_in = len(self.input_features)
        C_out = len(self.target_features)

        total_len = T
        train_end = int(total_len * self.train_ratio)
        val_end = int(total_len * (self.train_ratio + self.val_ratio))
        border1s = [0, train_end, val_end]
        border2s = [train_end, val_end, total_len]
        border1, border2 = border1s[self.set_type], border2s[self.set_type]

        data_x = np.zeros((N, T, F_in), dtype=np.float64)
        data_y = np.zeros((N, T, C_out), dtype=np.float64)
        raw_close = np.zeros((N, T), dtype=np.float64)

        close_idx = self.input_features.index('Close')

        for i, t in enumerate(self.tickers):
            df = raw[t].reindex(common_dates).ffill().bfill()
            x_full = df[self.input_features].values.astype(np.float64)
            y_full = df[self.target_features].values.astype(np.float64)

            raw_close[i] = x_full[:, close_idx]

            if self.scale:
                self.scalers_x[t].fit(x_full[border1s[0]:border2s[0]])
                data_x[i] = self.scalers_x[t].transform(x_full)
                if not self.return_targets:
                    self.scalers_y[t].fit(y_full[border1s[0]:border2s[0]])
                    data_y[i] = self.scalers_y[t].transform(y_full)
                else:
                    data_y[i] = y_full
            else:
                data_x[i] = x_full
                data_y[i] = y_full

        # slice to split
        self.data_x = data_x[:, border1:border2, :]
        self.data_y = data_y[:, border1:border2, :]
        self.raw_close = raw_close[:, border1:border2]
        self.dates = common_dates[border1:border2]

        self.n_input_features = F_in
        self.n_target_features = C_out
        self.num_stocks = N

        if self.flag == 'train':
            print(
                f"[CrossStock] {self.flag}: {len(self)} windows | "
                f"N_stocks={N} | Input={self.input_features} | "
                f"Target={self.target_features}"
            )

    # ------------------------------------------------------------------
    # dataset protocol
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self.data_x.shape[1] - self.seq_len - self.pred_len + 1

    def __getitem__(self, index: int):
        s_end = index + self.seq_len
        r_end = s_end + self.pred_len

        seq_x = self.data_x[:, index:s_end, :]        # [N, L, F_in]
        seq_y = self.data_y[:, s_end:r_end, :]        # [N, H, C_out]

        if self.return_targets:
            anchor = self.raw_close[:, s_end - 1]     # [N]
            anchor = anchor[:, None, None]            # broadcast over H, C_out
            seq_y = (seq_y - anchor) / anchor

        return (
            torch.tensor(seq_x, dtype=torch.float32),
            torch.tensor(seq_y, dtype=torch.float32),
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def get_anchors(self) -> np.ndarray:
        """Anchor Close per (sample, ticker). Shape: [n_samples, num_stocks]."""
        n = len(self)
        # raw_close is [N, T_split]; anchor for sample i is column i+seq_len-1
        cols = np.arange(n) + self.seq_len - 1
        return self.raw_close[:, cols].T  # [n, N]

    def inverse_transform_y(
        self, data: np.ndarray, anchors: Optional[np.ndarray] = None,
        ticker: Optional[str] = None,
    ) -> np.ndarray:
        """Inverse-transform predictions for a single ticker back to prices.

        When ``return_targets=True``, ``anchors`` (shape matching ``data``'s
        leading dim) is required. Otherwise the per-ticker scaler_y is used.
        """
        if self.return_targets:
            if anchors is None:
                raise ValueError("anchors required when return_targets=True")
            # broadcast anchors over trailing dims
            while anchors.ndim < data.ndim:
                anchors = anchors[..., None]
            return anchors * (1.0 + data)
        if ticker is None:
            raise ValueError("ticker required for scaler_y inverse transform")
        return self.scalers_y[ticker].inverse_transform(data)

    @property
    def enc_in(self) -> int:
        return self.n_input_features

    @property
    def c_out(self) -> int:
        return self.n_target_features

    @property
    def denorm_indices(self) -> tuple:
        """Same semantics as ParquetDataset.denorm_indices — one tuple shared
        across all tickers because feature layout is identical."""
        close_idx = self.input_features.index('Close')
        indices = []
        for tf in self.target_features:
            if tf in self.input_features:
                indices.append(self.input_features.index(tf))
            else:
                indices.append(close_idx)
        return tuple(indices)
