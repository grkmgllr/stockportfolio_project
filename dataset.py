import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

class ETTh1Dataset(Dataset):
    def __init__(self, root_path='data/processed', flag='train', 
                 size=None, features='M', target='OT', scale=True):
        """
        Args:
            root_path: Path to processed data folder
            flag: 'train', 'val', or 'test'
            size: [seq_len, label_len, pred_len]
            features: 'M' for multivariate, 'S' for univariate
            target: Target column name (usually 'OT' for ETTh1)
            scale: Whether to apply StandardScaler
        """
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        
        # ETTh1 Split Guidelines (Standard benchmark splits)
        # Train: 12 months, Val: 4 months, Test: 4 months
        # ETTh1 has ~17,420 rows (24 * 30 * 24 approx)
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.root_path = root_path
        self.data_path = 'ETTh1_processed.csv'
        
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # ETTh1 standard splits indices
        border1s = [0, 12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:] # Exclude date
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            # Fit scaler ONLY on training data
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        return torch.tensor(seq_x, dtype=torch.float32), torch.tensor(seq_y, dtype=torch.float32)

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)