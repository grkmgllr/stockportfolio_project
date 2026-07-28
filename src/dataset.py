"""
Dataset for stock price forecasting.

Input: OHLCV + optional engineered features (see features.py; 5 or 13 cols)
Output: High, Close + optional moving average predictions

Reads per-ticker CSVs produced by scripts/fetch_data.py (Yahoo Finance).
"""
from __future__ import annotations

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
from typing import Dict, List, Optional, Literal

from features import FEATURE_COLUMNS


class StockDataset(Dataset):
    """
    Sliding-window dataset for stock price forecasting.

    Reads a CSV per ticker, splits train/val/test by time (no shuffle),
    fits scalers on train only, and yields (seq_x, seq_y) windows.

    Two output modes:
      - return_targets=False: targets are standard-scaled absolute prices.
      - return_targets=True:  targets are % returns relative to the last
                              Close in the input window ("anchor").

    Shapes:
        seq_x: [seq_len, n_input_features]
        seq_y: [pred_len, n_target_features]
    """
    
    OHLCV_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume']
    # Extended input = raw OHLCV + causal engineered features (see features.py).
    # Replaces the Polygon-era Vwap/Transactions, which daily Yahoo data lacks.
    EXTENDED_COLUMNS = OHLCV_COLUMNS + list(FEATURE_COLUMNS)
    DEFAULT_TARGETS = ['High', 'Close']
    
    MA_CONFIGS: Dict[str, dict] = {
        'EMA_20': {'method': 'ema', 'period': 20},
        'SMA_50': {'method': 'sma', 'period': 50},
    }
    
    def __init__(
        self,
        ticker: str,
        root_path: str = 'data/raw',
        flag: Literal['train', 'val', 'test'] = 'train',
        seq_len: int = 14,
        pred_len: int = 5,
        input_features: Optional[List[str]] = None,
        target_features: Optional[List[str]] = None,
        scale: bool = True,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        ma_targets: Optional[List[str]] = None,
        return_targets: bool = False,
        start_date: Optional[str] = "2022-01-01",
        end_date: Optional[str] = None,
        train_end_date: Optional[str] = None,
        val_end_date: Optional[str] = None,
    ):
        """
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            root_path: Path to raw data folder containing {ticker}.csv
            flag: 'train', 'val', or 'test'
            seq_len: Length of input sequence (lookback window)
            pred_len: Length of prediction horizon
            input_features: List of input feature columns (default: OHLCV)
            target_features: List of target columns to predict (default: High, Close)
            scale: Whether to apply StandardScaler
            train_ratio: Ratio of data for training (default: 0.7)
            val_ratio: Ratio of data for validation (default: 0.15)
            ma_targets: List of MA target names to predict (e.g. ['EMA_20', 'SMA_50']).
                        Keys must exist in MA_CONFIGS. Pass None or [] to disable.
            return_targets: If True, seq_y is the % return of each target relative
                        to the last Close in the input window (the "anchor"); if
                        False, seq_y is the standard-scaled absolute price.
            start_date: Filter data to rows on or after this date (default: '2022-01-01').
                        Set to None to use all available data.
            end_date: Optional upper bound; keeps rows strictly on or before it.
                        Used by walk-forward folds to cap each fold's window.
            train_end_date / val_end_date: Optional explicit, date-based split
                        boundaries. When BOTH are given they replace
                        train_ratio/val_ratio: train is everything up to
                        train_end_date, val runs to val_end_date, and test is
                        whatever remains. This is what walk-forward evaluation
                        uses so each fold tests on a distinct calendar period.
        """
        self.ticker = ticker
        self.root_path = root_path
        self.flag = flag
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.scale = scale
        self.return_targets = return_targets
        self.start_date = start_date
        self.end_date = end_date
        self.train_end_date = train_end_date
        self.val_end_date = val_end_date

        # Default features (resolved in _load_data after reading the CSV)
        self._input_features_override = input_features
        self.input_features = input_features or self.OHLCV_COLUMNS.copy()
        self.target_features = target_features or self.DEFAULT_TARGETS.copy()
        
        # MA targets to append (validated later in _load_data)
        self.ma_targets = ma_targets or []
        for name in self.ma_targets:
            if name not in self.MA_CONFIGS:
                raise ValueError(
                    f"Unknown MA target '{name}'. "
                    f"Available: {list(self.MA_CONFIGS.keys())}"
                )
        
        # Validate target features are in input features
        for tf in self.target_features:
            if tf not in self.input_features:
                raise ValueError(f"Target feature '{tf}' must be in input_features")
        
        # Get indices of target features within input features
        self.target_indices = [self.input_features.index(tf) for tf in self.target_features]
        
        # Split ratios
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1.0 - train_ratio - val_ratio
        
        if self.test_ratio < 0:
            raise ValueError("train_ratio + val_ratio must be <= 1.0")
        
        # Type map for splits
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        
        # Scalers for input and target (separate for proper inverse transform)
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        
        self._load_data()
    
    @staticmethod
    def _compute_ma(close: pd.Series, method: str, period: int) -> pd.Series:
        """Compute a moving average from the Close column."""
        if method == 'sma':
            return close.rolling(window=period, min_periods=period).mean()
        elif method == 'ema':
            return close.ewm(span=period, min_periods=period, adjust=False).mean()
        else:
            raise ValueError(f"Unknown MA method: {method}")

    def _load_data(self):
        """Read the ticker CSV and build the sliding-window arrays for this split.

        Steps, in order:
          1. Read ``{ticker}.csv`` and filter to ``start_date`` onward.
          2. Forward/back-fill missing values.
          3. Compute any moving-average target columns from Close, then trim the
             leading warm-up rows those rolling windows produce.
          4. Resolve the final input-feature list (auto-detect the extended set).
          5. Split into train/val/test by time (no shuffle) and fit the scalers
             on the train split only, so val/test never leak into the statistics.
          6. Slice the scaled arrays to the current split into ``data_x``/``data_y``.
        """
        file_path = os.path.join(self.root_path, f"{self.ticker}.csv")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Data file not found: {file_path}\n"
                f"Run: python src/scripts/fetch_data.py to prepare data first."
            )
        
        df_raw = pd.read_csv(file_path)

        # Filter by start_date / end_date if specified
        if self.start_date and 'Date' in df_raw.columns:
            n_before = len(df_raw)
            df_raw = df_raw[df_raw['Date'] >= self.start_date].reset_index(drop=True)
            if self.flag == 'train' and n_before != len(df_raw):
                print(f"[{self.ticker}] Filtered to {self.start_date}+: "
                      f"{n_before} → {len(df_raw)} rows")
        if self.end_date and 'Date' in df_raw.columns:
            df_raw = df_raw[df_raw['Date'] <= self.end_date].reset_index(drop=True)

        # ffill only — bfill would pull FUTURE values backward (look-ahead leak).
        df_raw = df_raw.ffill().dropna(
            subset=['Open', 'High', 'Low', 'Close', 'Volume']
        ).reset_index(drop=True)
        
        # Compute moving-average target columns from Close before any
        # splitting so the rolling windows see the full history.
        for ma_name in self.ma_targets:
            cfg = self.MA_CONFIGS[ma_name]
            df_raw[ma_name] = self._compute_ma(
                df_raw['Close'], cfg['method'], cfg['period'],
            )
        
        # Trim leading NaN rows caused by MA warm-up.  The longest MA
        # window determines how many rows to drop.
        if self.ma_targets:
            max_period = max(
                self.MA_CONFIGS[n]['period'] for n in self.ma_targets
            )
            n_before = len(df_raw)
            df_raw = df_raw.iloc[max_period - 1:].reset_index(drop=True)
            if self.flag == 'train':
                print(f"[{self.ticker}] Trimmed {n_before - len(df_raw)} "
                      f"MA warm-up rows (max period={max_period})")
        
        # Append MA names to the target list (after base targets)
        all_targets = self.target_features + [
            n for n in self.ma_targets if n not in self.target_features
        ]
        self.target_features = all_targets
        
        # Auto-detect the extended column set (OHLCV + engineered features)
        # when no explicit input_features were provided by the caller.
        if self._input_features_override is None:
            has_extended = all(c in df_raw.columns for c in self.EXTENDED_COLUMNS)
            if has_extended:
                self.input_features = self.EXTENDED_COLUMNS.copy()
        
        # Validate required columns exist
        missing_cols = set(self.input_features) - set(df_raw.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in data: {missing_cols}")
        
        # Re-resolve target indices after potential feature list change
        self.target_indices = [
            self.input_features.index(tf)
            for tf in self.target_features
            if tf in self.input_features
        ]
        
        # Extract input features
        df_input = df_raw[self.input_features].copy()
        
        # Extract target features (High, Close + optional MAs)
        df_target = df_raw[self.target_features].copy()
        
        # Time-ordered split boundaries. Explicit dates (walk-forward) take
        # precedence; otherwise fall back to the ratio-based split.
        total_len = len(df_input)
        if self.train_end_date and self.val_end_date and 'Date' in df_raw.columns:
            dates = df_raw['Date'].values
            train_end = int((dates <= self.train_end_date).sum())
            val_end = int((dates <= self.val_end_date).sum())
        else:
            train_end = int(total_len * self.train_ratio)
            val_end = int(total_len * (self.train_ratio + self.val_ratio))

        border1s = [0, train_end, val_end]
        border2s = [train_end, val_end, total_len]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # unscaled Close for return-targets mode and inverse transforms
        close_col_idx = self.input_features.index('Close')
        self.raw_close = df_input.iloc[border1:border2, close_col_idx].values.astype(np.float64)

        if self.scale:
            # Fit on the train split only (no future leakage), then transform
            # every row with those train statistics; the split slice happens
            # afterwards. border1s[0]:border2s[0] is exactly the train range.
            train_x = df_input.iloc[border1s[0]:border2s[0]].values
            self.scaler_x.fit(train_x)
            data_x = self.scaler_x.transform(df_input.values)

            if not self.return_targets:
                # Price-target mode: scale targets the same way as inputs.
                train_y = df_target.iloc[border1s[0]:border2s[0]].values
                self.scaler_y.fit(train_y)
                data_y = self.scaler_y.transform(df_target.values)
            else:
                # Return-target mode: keep raw prices; __getitem__ turns them
                # into anchor-relative % returns at read time.
                data_y = df_target.values.astype(np.float64)
        else:
            data_x = df_input.values
            data_y = df_target.values

        # Slice to current split
        self.data_x = data_x[border1:border2]
        self.data_y = data_y[border1:border2]
        
        # Store metadata
        self.n_input_features = len(self.input_features)
        self.n_target_features = len(self.target_features)
        
        print(f"[{self.ticker}] {self.flag}: {len(self)} samples | "
              f"Input: {self.input_features} | Target: {self.target_features}")
    
    def __getitem__(self, index):
        """
        Get a single sample: seq_x [seq_len, enc_in], seq_y [pred_len, c_out].

        When return_targets=True, seq_y is % returns relative to anchor Close.
        """
        s_begin = index
        s_end = s_begin + self.seq_len       # input window end
        r_begin = s_end                      # target window start (no overlap)
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        if self.return_targets:
            # anchor = last Close price in the input window
            anchor = self.raw_close[s_end - 1]
            seq_y = (seq_y - anchor) / anchor

        return (
            torch.tensor(seq_x, dtype=torch.float32),
            torch.tensor(seq_y, dtype=torch.float32),
        )
    
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1
    
    def get_anchors(self) -> np.ndarray:
        """Return anchor Close prices for all samples (one per sample)."""
        n = len(self)
        return np.array([self.raw_close[i + self.seq_len - 1] for i in range(n)])

    def inverse_transform_x(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform input features back to original scale."""
        return self.scaler_x.inverse_transform(data)

    def inverse_transform_y(self, data: np.ndarray, anchors: np.ndarray | None = None) -> np.ndarray:
        """Inverse transform target features back to original scale.

        When return_targets=True, ``data`` contains percentage returns and
        ``anchors`` must be provided to convert back to absolute prices.
        """
        if self.return_targets:
            if anchors is None:
                raise ValueError("anchors required when return_targets=True")
            return anchors * (1.0 + data)
        return self.scaler_y.inverse_transform(data)
    
    @property
    def enc_in(self) -> int:
        """Number of input features (for model config)."""
        return self.n_input_features

    @property
    def c_out(self) -> int:
        """Number of output features (for model config)."""
        return self.n_target_features

    @property
    def denorm_indices(self) -> tuple:
        """
        Map each target channel to its closest input channel index.

        Used by TimesNet / TimeMixer to pick the right per-sample
        mean/std when denormalising predictions back to price scale.
        - Price targets (High, Close) → their exact input column index.
        - MA targets (EMA_20, SMA_50) → Close column index (same scale).
        """
        close_idx = self.input_features.index('Close')
        indices = []
        for tf in self.target_features:
            if tf in self.input_features:
                indices.append(self.input_features.index(tf))
            else:
                indices.append(close_idx)
        return tuple(indices)


# Backward-compatible aliases (the class used to be ParquetDataset; it now
# reads Yahoo Finance CSVs, so StockDataset is the canonical name).
ParquetDataset = StockDataset
YahooDataset = StockDataset
