"""
Parquet Dataset for stock price forecasting.

Input: OHLCV + optional Vwap/Transactions (5-7 features)
Output: High, Close + optional moving average predictions
"""
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
from typing import Dict, List, Optional, Literal


class ParquetDataset(Dataset):
    """
    Dataset for Parquet stock data.
    
    Predicts High and Close prices (and optionally moving averages) from
    OHLCV input for short-term forecasting.
    Uses date-based splits for proper time series handling.
    
    Input features: Open, High, Low, Close, Volume (5 features)
                    + Vwap, Transactions (7 features, when available)
    Output targets: High, Close (2 features)
                    + EMA_20, SMA_50 (4 features, when ma_targets enabled)
    """
    
    OHLCV_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume']
    EXTENDED_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume', 'Vwap', 'Transactions']
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
        seq_len: int = 30,
        pred_len: int = 1,
        input_features: Optional[List[str]] = None,
        target_features: Optional[List[str]] = None,
        scale: bool = True,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        ma_targets: Optional[List[str]] = None,
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
        """
        self.ticker = ticker
        self.root_path = root_path
        self.flag = flag
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.scale = scale
        
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
        """Load and preprocess the stock data."""
        file_path = os.path.join(self.root_path, f"{self.ticker}.csv")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Data file not found: {file_path}\n"
                f"Run: python scripts/fetch_data.py or "
                f"python scripts/resample_parquet.py to prepare data first."
            )
        
        df_raw = pd.read_csv(file_path)
        
        # Handle missing values with forward fill then backward fill
        df_raw = df_raw.ffill().bfill()
        
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
        
        # Auto-detect extended columns (Vwap, Transactions) when no
        # explicit input_features were provided by the caller.
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
        
        # Calculate split boundaries
        total_len = len(df_input)
        train_end = int(total_len * self.train_ratio)
        val_end = int(total_len * (self.train_ratio + self.val_ratio))
        
        border1s = [0, train_end, val_end]
        border2s = [train_end, val_end, total_len]
        
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.scale:
            # Fit scalers ONLY on training data
            train_x = df_input.iloc[border1s[0]:border2s[0]].values
            train_y = df_target.iloc[border1s[0]:border2s[0]].values
            
            self.scaler_x.fit(train_x)
            self.scaler_y.fit(train_y)
            
            # Transform all data
            data_x = self.scaler_x.transform(df_input.values)
            data_y = self.scaler_y.transform(df_target.values)
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
        Get a single sample.
        
        Returns:
            seq_x: Input sequence [seq_len, n_input_features]
            seq_y: Target sequence [pred_len, n_target_features]
        """
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len
        
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        
        return (
            torch.tensor(seq_x, dtype=torch.float32),
            torch.tensor(seq_y, dtype=torch.float32),
        )
    
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1
    
    def inverse_transform_x(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform input features back to original scale."""
        return self.scaler_x.inverse_transform(data)
    
    def inverse_transform_y(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform target features (High, Close) back to original scale."""
        return self.scaler_y.inverse_transform(data)
    
    @property
    def enc_in(self) -> int:
        """Number of input features (for model config)."""
        return self.n_input_features
    
    @property
    def c_out(self) -> int:
        """Number of output features (for model config)."""
        return self.n_target_features


# Backward-compatible alias
YahooDataset = ParquetDataset
