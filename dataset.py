"""
Yahoo Finance Dataset for stock price forecasting.

Input: OHLCV (Open, High, Low, Close, Volume) - 5 features
Output: High, Close predictions - 2 features
"""
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
from typing import List, Optional, Literal


class YahooDataset(Dataset):
    """
    Dataset for Yahoo Finance stock data.
    
    Predicts High and Close prices from OHLCV input for short-term forecasting.
    Uses date-based splits for proper time series handling.
    
    Input features: Open, High, Low, Close, Volume (5 features)
    Output targets: High, Close (2 features)
    """
    
    # Column names in Yahoo Finance data
    OHLCV_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume']
    DEFAULT_TARGETS = ['High', 'Close']
    
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
        """
        self.ticker = ticker
        self.root_path = root_path
        self.flag = flag
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.scale = scale
        
        # Default features
        self.input_features = input_features or self.OHLCV_COLUMNS.copy()
        self.target_features = target_features or self.DEFAULT_TARGETS.copy()
        
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
    
    def _load_data(self):
        """Load and preprocess the stock data."""
        file_path = os.path.join(self.root_path, f"{self.ticker}.csv")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Data file not found: {file_path}\n"
                f"Run: python scripts/fetch_data.py to download data first."
            )
        
        df_raw = pd.read_csv(file_path)
        
        # Handle missing values with forward fill then backward fill
        df_raw = df_raw.ffill().bfill()
        
        # Validate required columns exist
        missing_cols = set(self.input_features) - set(df_raw.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in data: {missing_cols}")
        
        # Extract input features (OHLCV)
        df_input = df_raw[self.input_features].copy()
        
        # Extract target features (High, Low)
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
