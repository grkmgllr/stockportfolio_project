import torch
import torch.nn as nn
import lightgbm as lgb
import numpy as np
import os
import pandas as pd
from typing import Optional, List
import copy

from models.TimeMixer.TimeMixer import TimeMixer, TimeMixerConfig


class HybridTimeMixerLGBM:
    """
    Hybrid model: TimeMixer (deep learning) + LightGBM (gradient boosting).

    Pipeline:
        Phase 1: Train TimeMixer on raw OHLCV sequences (regression).
        Phase 2: Extract features — latent embeddings + predictions + statistical.
        Phase 3: Train per-horizon LightGBM regressors + optional direction classifiers.

    Feature groups fed to LightGBM:
        - Latent:      TimeMixer penultimate-layer embeddings
        - Prediction:  mean/last/std/slope of TimeMixer's forecast horizon
        - Statistical: hand-crafted technical indicators from raw sequences
    """

    def __init__(self, timemixer_config: TimeMixerConfig, lgbm_params: Optional[dict] = None) -> None:
        """
        Initialize the HybridTimeMixerLGBM model.

        Args:
            timemixer_config (TimeMixerConfig):
                Configuration object for the underlying TimeMixer model.
            lgbm_params (dict, optional):
                Parameters for the LightGBM model.
        """
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

        self.timemixer = TimeMixer(timemixer_config).to(self.device)

        self.lgbm_params = lgbm_params if lgbm_params else {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 15,
            'max_depth': 4,
            'min_data_in_leaf': 20,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'lambda_l1': 1.0,
            'lambda_l2': 1.0,
            'verbose': -1,
        }
        self.models = []
        self.classifiers = []
        self.regression_model_metadata = []

    # ------------------------------------------------------------------
    #  Phase 1 – TimeMixer Training (with best-model checkpoint)
    # ------------------------------------------------------------------

    def fit_timemixer(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        epochs: int = 10,
        lr: float = 1e-3,
        verbose: bool = False,
    ) -> None:
        """
        Phase 1: Train the TimeMixer model using PyTorch.

        Saves the best model weights based on validation loss (if provided)
        or training loss.

        Args:
            train_loader (torch.utils.data.DataLoader):
                DataLoader containing the training data batches.
            val_loader (torch.utils.data.DataLoader, optional):
                DataLoader for validation monitoring.
            epochs (int):
                Number of training epochs.
            lr (float):
                Initial learning rate for the Adam optimizer.
        """
        if verbose:
            print("\n>>> Phase 1: Training TimeMixer (Deep Learning)...")

        # Use a separate shuffled loader for training to improve generalization
        shuffled_loader = torch.utils.data.DataLoader(
            train_loader.dataset,
            batch_size=train_loader.batch_size,
            shuffle=True,
        )

        optimizer = torch.optim.Adam(self.timemixer.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
        criterion = nn.MSELoss()

        best_loss = float('inf')
        best_state = None

        for epoch in range(epochs):
            self.timemixer.train()
            total_loss = 0
            for batch_x, batch_y in shuffled_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                preds = self.timemixer(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.timemixer.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()

            avg_train = total_loss / len(shuffled_loader)

            # Validation
            val_msg = ""
            monitor_loss = avg_train
            if val_loader is not None:
                val_loss = self._evaluate_timemixer(val_loader, criterion)
                val_msg = f" | Val Loss: {val_loss:.4f}"
                monitor_loss = val_loss

            scheduler.step(monitor_loss)
            current_lr = optimizer.param_groups[0]['lr']

            # Checkpoint best model
            if monitor_loss < best_loss:
                best_loss = monitor_loss
                best_state = copy.deepcopy(self.timemixer.state_dict())
                val_msg += " *"

            if verbose:
                print(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"Train Loss: {avg_train:.4f}{val_msg} | "
                    f"LR: {current_lr:.6f}"
                )

        # Restore best model
        if best_state is not None:
            self.timemixer.load_state_dict(best_state)
            if verbose:
                print(f">>> Restored best TimeMixer weights (loss={best_loss:.4f})")

    def _evaluate_timemixer(self, loader, criterion):
        self.timemixer.eval()
        total = 0
        with torch.no_grad():
            for bx, by in loader:
                bx, by = bx.to(self.device), by.to(self.device)
                total += criterion(self.timemixer(bx), by).item()
        return total / max(len(loader), 1)

    # ------------------------------------------------------------------
    #  Phase 2 – Feature Extraction
    # ------------------------------------------------------------------

    def extract_latent_features(self, data_loader: torch.utils.data.DataLoader) -> np.ndarray:
        """Extract latent embeddings from TimeMixer's last PDM block output."""
        self.timemixer.eval()
        embeddings: List[np.ndarray] = []

        with torch.no_grad():
            for batch_x, _ in data_loader:
                batch_x = batch_x.to(self.device)

                x_scales = self.timemixer.multiscale_inputs(batch_x)
                z_scales = self.timemixer.embed_multiscale(x_scales)
                for pdm in self.timemixer.pdm_blocks:
                    z_scales = pdm(z_scales)

                future_latent = self.timemixer.fmulti_predictor_mixing(z_scales)
                latent_vector = future_latent.mean(dim=1)
                embeddings.append(latent_vector.cpu().numpy())

        return np.concatenate(embeddings, axis=0)

    def extract_prediction_features(self, data_loader: torch.utils.data.DataLoader) -> np.ndarray:
        """
        Extract TimeMixer's actual forecast predictions as features.

        For each sample, computes mean, last, std, and slope of the
        predicted horizon, yielding ``C_out * 4`` features.
        """
        self.timemixer.eval()
        preds_list: List[np.ndarray] = []

        with torch.no_grad():
            for batch_x, _ in data_loader:
                batch_x = batch_x.to(self.device)
                preds = self.timemixer(batch_x)  # [B, H, C_out]

                p_mean = preds.mean(dim=1)
                p_last = preds[:, -1, :]
                p_std = preds.std(dim=1)
                p_slope = preds[:, -1, :] - preds[:, 0, :]

                batch_feat = torch.cat([p_mean, p_last, p_std, p_slope], dim=1)
                preds_list.append(batch_feat.cpu().numpy())

        return np.concatenate(preds_list, axis=0)

    @staticmethod
    def compute_statistical_features(data_loader: torch.utils.data.DataLoader) -> np.ndarray:
        """
        Compute rich statistical and technical features from raw input sequences.

        Per channel (C), computes 16 features:
            mean, std, min, max, last, first, slope, skewness,
            return, short_momentum, mid_momentum, volatility,
            range, up_ratio, rsi, position_in_range.

        Total features: ``C * 16``.
        """
        eps = 1e-8
        all_features: List[np.ndarray] = []

        for batch_x, _ in data_loader:
            x = batch_x.numpy()  # [B, T, C]
            B, T, C = x.shape

            # --- Basic statistics ---
            f_mean = x.mean(axis=1)
            f_std = x.std(axis=1)
            f_min = x.min(axis=1)
            f_max = x.max(axis=1)
            f_last = x[:, -1, :]
            f_first = x[:, 0, :]
            f_slope = f_last - f_first

            centered = x - f_mean[:, np.newaxis, :]
            m3 = (centered ** 3).mean(axis=1)
            f_skew = m3 / (f_std ** 3 + eps)

            # --- Returns & Momentum ---
            f_return = (f_last - f_first) / (np.abs(f_first) + eps)

            idx_short = min(5, T - 1)
            idx_mid = min(10, T - 1)
            f_mom_short = f_last - x[:, -(idx_short + 1), :]
            f_mom_mid = f_last - x[:, -(idx_mid + 1), :]

            # --- Volatility ---
            returns = np.diff(x, axis=1)  # [B, T-1, C]
            f_vol = returns.std(axis=1)

            # --- Range ---
            f_range = (f_max - f_min) / (np.abs(f_mean) + eps)

            # --- Trend strength (up ratio) ---
            f_up_ratio = (returns > 0).mean(axis=1)

            # --- RSI approximation ---
            gains = np.maximum(returns, 0)
            losses = np.maximum(-returns, 0)
            avg_gain = gains.mean(axis=1) + eps
            avg_loss = losses.mean(axis=1) + eps
            f_rsi = 1.0 - 1.0 / (1.0 + avg_gain / avg_loss)

            # --- Position in range ---
            f_pos = (f_last - f_min) / (f_max - f_min + eps)

            batch_feat = np.concatenate([
                f_mean, f_std, f_min, f_max, f_last, f_first, f_slope, f_skew,
                f_return, f_mom_short, f_mom_mid, f_vol,
                f_range, f_up_ratio, f_rsi, f_pos,
            ], axis=1)
            all_features.append(batch_feat)

        return np.concatenate(all_features, axis=0)

    @staticmethod
    def compute_raw_statistical_features(data_loader: torch.utils.data.DataLoader, dataset, close_idx_x=None):
        """Compute scale-free technical features from inverse-transformed (raw dollar) prices."""
        eps = 1e-8
        all_features = []
        feature_names = []

        if close_idx_x is None:
            close_idx_x = dataset.input_features.index('Close')

        for col_name in dataset.input_features:
            if col_name == 'Volume':
                feature_names.extend([
                    f"{col_name}_log_change_1d",
                    f"{col_name}_log_change_5d",
                    f"{col_name}_log_change_10d",
                    f"{col_name}_log_volatility",
                    f"{col_name}_log_pos_in_range",
                ])
            else:
                feature_names.extend([
                    f"{col_name}_pct_change_first_last",
                    f"{col_name}_pct_change_1d",
                    f"{col_name}_pct_change_3d",
                    f"{col_name}_pct_change_5d",
                    f"{col_name}_pct_change_10d",
                    f"{col_name}_volatility_full",
                    f"{col_name}_volatility_5d",
                    f"{col_name}_volatility_10d",
                    f"{col_name}_range_pct",
                    f"{col_name}_pos_in_range",
                    f"{col_name}_mean_pct_change",
                    f"{col_name}_std_pct_change",
                    f"{col_name}_up_ratio",
                ])
                
        feature_names.extend([
            "Close_log_return_1d",
            "Close_log_return_3d",
            "Close_log_return_5d",
            "Close_log_return_10d",
            "Close_volatility_5d",
            "Close_volatility_10d",
            "Close_ma_ratio_5",
            "Close_ma_ratio_10",
            "Close_ma_ratio_20",
            "Close_dist_from_high",
            "Close_dist_from_low",
        ])

        for batch_x, _ in data_loader:
            B, T, C = batch_x.shape
            x_flat = batch_x.reshape(-1, C).numpy()
            x_raw = dataset.inverse_transform_x(x_flat).reshape(B, T, C)
            
            batch_feats = []
            
            for c_idx, col_name in enumerate(dataset.input_features):
                if col_name == 'Volume':
                    v_raw = x_raw[:, :, c_idx]
                    v_log = np.log1p(np.maximum(v_raw, 0))
                    
                    v_last = v_log[:, -1]
                    v_min = v_log.min(axis=1)
                    v_max = v_log.max(axis=1)
                    
                    f_change_1d = (v_last - v_log[:, -2]) if T > 1 else np.zeros(B)
                    f_change_5d = (v_last - v_log[:, -6]) if T > 5 else np.zeros(B)
                    f_change_10d = (v_last - v_log[:, -11]) if T > 10 else np.zeros(B)
                    
                    v_diff = np.diff(v_log, axis=1)
                    f_vol = v_diff.std(axis=1) if T > 1 else np.zeros(B)
                    f_pos = (v_last - v_min) / (v_max - v_min + eps)
                    
                    batch_feats.extend([f_change_1d, f_change_5d, f_change_10d, f_vol, f_pos])
                    
                else:
                    c_raw = x_raw[:, :, c_idx]
                    
                    c_mean = c_raw.mean(axis=1)
                    c_min = c_raw.min(axis=1)
                    c_max = c_raw.max(axis=1)
                    c_last = c_raw[:, -1]
                    c_first = c_raw[:, 0]
                    
                    f_pct_first_last = (c_last - c_first) / (np.abs(c_first) + eps)
                    f_pct_1d = (c_last - c_raw[:, -2]) / (np.abs(c_raw[:, -2]) + eps) if T > 1 else np.zeros(B)
                    f_pct_3d = (c_last - c_raw[:, -4]) / (np.abs(c_raw[:, -4]) + eps) if T > 3 else np.zeros(B)
                    f_pct_5d = (c_last - c_raw[:, -6]) / (np.abs(c_raw[:, -6]) + eps) if T > 5 else np.zeros(B)
                    f_pct_10d = (c_last - c_raw[:, -11]) / (np.abs(c_raw[:, -11]) + eps) if T > 10 else np.zeros(B)
                    
                    c_returns = np.diff(c_raw, axis=1) / (np.abs(c_raw[:, :-1]) + eps)
                    f_vol_full = c_returns.std(axis=1) if T > 1 else np.zeros(B)
                    
                    f_vol_5d = c_returns[:, -5:].std(axis=1) if T > 5 else np.zeros(B)
                    f_vol_10d = c_returns[:, -10:].std(axis=1) if T > 10 else np.zeros(B)
                    
                    f_range_pct = (c_max - c_min) / (np.abs(c_mean) + eps)
                    f_pos = (c_last - c_min) / (c_max - c_min + eps)
                    
                    f_mean_pct = c_returns.mean(axis=1) if T > 1 else np.zeros(B)
                    f_std_pct = c_returns.std(axis=1) if T > 1 else np.zeros(B)
                    f_up_ratio = (c_returns > 0).mean(axis=1) if T > 1 else np.zeros(B)
                    
                    batch_feats.extend([
                        f_pct_first_last, f_pct_1d, f_pct_3d, f_pct_5d, f_pct_10d,
                        f_vol_full, f_vol_5d, f_vol_10d,
                        f_range_pct, f_pos, f_mean_pct, f_std_pct, f_up_ratio
                    ])
                    
            c_raw = x_raw[:, :, close_idx_x]
            c_last = c_raw[:, -1]
            
            c_log_1d = np.log(np.maximum(c_last / (c_raw[:, -2] + eps), eps)) if T > 1 else np.zeros(B)
            c_log_3d = np.log(np.maximum(c_last / (c_raw[:, -4] + eps), eps)) if T > 3 else np.zeros(B)
            c_log_5d = np.log(np.maximum(c_last / (c_raw[:, -6] + eps), eps)) if T > 5 else np.zeros(B)
            c_log_10d = np.log(np.maximum(c_last / (c_raw[:, -11] + eps), eps)) if T > 10 else np.zeros(B)
            
            c_ret_ser = np.diff(c_raw, axis=1) / (np.abs(c_raw[:, :-1]) + eps)
            c_vol_5d = c_ret_ser[:, -5:].std(axis=1) if T > 5 else np.zeros(B)
            c_vol_10d = c_ret_ser[:, -10:].std(axis=1) if T > 10 else np.zeros(B)
            
            c_ma_5 = c_raw[:, -5:].mean(axis=1) if T >= 5 else np.zeros(B)
            c_ma_r_5 = c_last / (c_ma_5 + eps) - 1 if T >= 5 else np.zeros(B)
            
            c_ma_10 = c_raw[:, -10:].mean(axis=1) if T >= 10 else np.zeros(B)
            c_ma_r_10 = c_last / (c_ma_10 + eps) - 1 if T >= 10 else np.zeros(B)
            
            c_ma_20 = c_raw[:, -20:].mean(axis=1) if T >= 20 else np.zeros(B)
            c_ma_r_20 = c_last / (c_ma_20 + eps) - 1 if T >= 20 else np.zeros(B)
            
            c_high = c_raw.max(axis=1)
            c_low = c_raw.min(axis=1)
            
            c_dist_high = c_last / (c_high + eps) - 1
            c_dist_low = c_last / (c_low + eps) - 1
            
            batch_feats.extend([
                c_log_1d, c_log_3d, c_log_5d, c_log_10d,
                c_vol_5d, c_vol_10d,
                c_ma_r_5, c_ma_r_10, c_ma_r_20,
                c_dist_high, c_dist_low
            ])
            
            feat_mat = np.column_stack(batch_feats)
            
            feat_mat = np.nan_to_num(feat_mat, nan=0.0, posinf=0.0, neginf=0.0)
            feat_mat = np.clip(feat_mat, -5, 5)
            
            all_features.append(feat_mat)

        return np.concatenate(all_features, axis=0), feature_names

    def _build_features(self, data_loader, external_features=None, dataset=None, use_raw_stats=True):
        """Concatenate latent + prediction + statistical features into one matrix."""
        f_latent = self.extract_latent_features(data_loader)
        f_pred = self.extract_prediction_features(data_loader)
        
        if use_raw_stats and dataset is not None:
            f_stat, stat_names = self.compute_raw_statistical_features(data_loader, dataset)
        else:
            f_stat = self.compute_statistical_features(data_loader)
            stat_names = [f"Stat_Norm_{i}" for i in range(f_stat.shape[1])]
            
        self.stat_feature_names = stat_names

        parts = [f_latent, f_pred, f_stat]
        if external_features is not None:
            parts.append(external_features)

        combined = np.hstack(parts)
        return combined, f_latent.shape[1], f_pred.shape[1], f_stat.shape[1]

    def _diagnose_and_save_features(self, feat, y, n_lat, n_pred, n_stat, split_name, output_dir=None, is_raw=True, verbose=False):
        """Validate feature ranges, log stats, and optionally save to disk."""
        n_ext = feat.shape[1] - (n_lat + n_pred + n_stat)
        
        if verbose:
            print(f"\n--- Feature Diagnostics: {split_name.upper()} ---")
            stat_type = "RAW_SCALE_FREE" if is_raw else "NORMALIZED"
            print(f"  Total Features: {feat.shape[1]} (Samples: {feat.shape[0]})")
            print(f"  Latent: {n_lat} | Prediction: {n_pred} | Statistical ({stat_type}): {n_stat} | External: {n_ext}")
        
        c_lat = feat[:, :n_lat]
        c_pred = feat[:, n_lat:n_lat+n_pred]
        c_stat = feat[:, n_lat+n_pred:n_lat+n_pred+n_stat]
        
        groups = [("Latent", c_lat), ("Prediction", c_pred), ("Statistical", c_stat)]
        if n_ext > 0:
            c_ext = feat[:, n_lat+n_pred+n_stat:]
            groups.append(("External", c_ext))
            
        for name, data in groups:
            if data.shape[1] > 0:
                if np.isnan(data).any():
                    raise ValueError(f"NaN values found in {name} features for {split_name} split!")
                if np.isinf(data).any():
                    raise ValueError(f"Inf values found in {name} features for {split_name} split!")
                if verbose:
                    print(f"  {name:12s} - Mean: {data.mean():>8.4f} | Std: {data.std():>8.4f} | Min: {data.min():>8.4f} | Max: {data.max():>8.4f}")
                
                if name == "Statistical":
                    if data.max() > 5 or data.min() < -5:
                        print("    WARNING: Statistical max/min bounds exceeded [-5, 5]!")
                    if verbose:
                        if data.max() > 5 or data.min() < -5:
                            print("    WARNING: Statistical max/min bounds exceeded [-5, 5]!")
                        else:
                            print("    OK: Statistical features within [-5, 5]")

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            np.save(os.path.join(output_dir, f"{split_name}_features.npy"), feat)
            if y is not None:
                np.save(os.path.join(output_dir, f"y_{split_name}.npy"), y)
            if verbose:
                print(f"  Saved {split_name}_features.npy to {output_dir}")
            
            if hasattr(self, 'stat_feature_names') and self.stat_feature_names:
                with open(os.path.join(output_dir, "statistical_feature_names.txt"), "w") as f:
                    for s in self.stat_feature_names:
                        f.write(f"{s}\n")

    # ------------------------------------------------------------------
    #  Phase 3 – LightGBM Training
    # ------------------------------------------------------------------

    def fit_lgbm(
        self,
        train_loader, y_labels,
        val_loader=None, y_val=None,
        external_features=None,
        output_dir=None,
        train_dataset=None,
        val_dataset=None,
        use_raw_stats=True,
        use_direction_classifier=True,
        use_regression_variant_search=False,
        verbose=False
    ) -> None:
        """Train one LightGBM regressor per forecast day on combined features."""
        if verbose:
            print("\n>>> Phase 2: Extracting features from TimeMixer...")
        train_feat, n_lat, n_pred, n_stat = self._build_features(train_loader, external_features, dataset=train_dataset, use_raw_stats=use_raw_stats)
        
        self._diagnose_and_save_features(train_feat, y_labels, n_lat, n_pred, n_stat, "train", output_dir, is_raw=(use_raw_stats and train_dataset is not None), verbose=verbose)

        pred_len = y_labels.shape[1]
        self.models = []
        self.regression_model_metadata = []

        if verbose:
            print(
                f"\n>>> Phase 3: Training {pred_len} LightGBM Models (Multi-step Forecast)... "
                f"(Latent: {n_lat} + Pred: {n_pred} + Stat: {n_stat}"
                f" = {train_feat.shape[1]} total features)"
            )

        if val_loader is not None and y_val is not None:
            val_feat, _, _, _ = self._build_features(val_loader, external_features=None, dataset=val_dataset, use_raw_stats=use_raw_stats)
            self._diagnose_and_save_features(val_feat, y_val, n_lat, n_pred, n_stat, "val", output_dir, is_raw=(use_raw_stats and val_dataset is not None), verbose=verbose)

        candidates = []
        if use_regression_variant_search and val_loader is not None and y_val is not None:
            transforms = ['raw_return', 'demean_mean', 'demean_median', 'standardize_mean_std']
            lgbm_candidates = [
                {
                    'name': 'Candidate_1_Regression',
                    'params': {
                        'objective': 'regression', 'metric': 'rmse', 'learning_rate': 0.03,
                        'num_leaves': 15, 'max_depth': 4, 'min_data_in_leaf': 20,
                        'lambda_l1': 1.0, 'lambda_l2': 1.0, 'verbose': -1
                    }
                },
                {
                    'name': 'Candidate_2_Regression_L1',
                    'params': {
                        'objective': 'regression_l1', 'metric': 'l1', 'learning_rate': 0.03,
                        'num_leaves': 15, 'max_depth': 4, 'min_data_in_leaf': 20,
                        'lambda_l1': 1.0, 'lambda_l2': 2.0, 'verbose': -1
                    }
                },
                {
                    'name': 'Candidate_3_Huber',
                    'params': {
                        'objective': 'huber', 'metric': 'l1', 'learning_rate': 0.03,
                        'num_leaves': 15, 'max_depth': 4, 'min_data_in_leaf': 25,
                        'lambda_l1': 1.0, 'lambda_l2': 2.0, 'verbose': -1
                    }
                },
                {
                    'name': 'Candidate_4_Regression_Small',
                    'params': {
                        'objective': 'regression', 'metric': 'rmse', 'learning_rate': 0.02,
                        'num_leaves': 7, 'max_depth': 3, 'min_data_in_leaf': 30,
                        'lambda_l1': 2.0, 'lambda_l2': 3.0, 'verbose': -1
                    }
                }
            ]
        else:
            transforms = ['raw_return']
            lgbm_candidates = [{'name': 'Default', 'params': self.lgbm_params}]

        all_candidates_results = []
        selected_variants_results = []

        for i in range(pred_len):
            if verbose:
                print(f"\n  -> Training Models for Day {i+1}...")
            
            best_val_mae = float('inf')
            best_model = None
            best_meta = None
            
            for t_name in transforms:
                # Prepare transformed labels for training and validation
                y_train_h = np.ascontiguousarray(y_labels[:, i])
                y_val_h = np.ascontiguousarray(y_val[:, i]) if (val_loader is not None and y_val is not None) else None
                
                baseline = 0.0
                scale = 1.0
                
                if t_name == 'demean_mean':
                    baseline = float(np.mean(y_train_h))
                    y_train_h = y_train_h - baseline
                    if y_val_h is not None:
                        y_val_h = y_val_h - baseline
                elif t_name == 'demean_median':
                    baseline = float(np.median(y_train_h))
                    y_train_h = y_train_h - baseline
                    if y_val_h is not None:
                        y_val_h = y_val_h - baseline
                elif t_name == 'standardize_mean_std':
                    baseline = float(np.mean(y_train_h))
                    scale = float(np.std(y_train_h)) + 1e-8
                    y_train_h = (y_train_h - baseline) / scale
                    if y_val_h is not None:
                        y_val_h = (y_val_h - baseline) / scale
                        
                train_data = lgb.Dataset(train_feat, label=y_train_h)
                
                for cand in lgbm_candidates:
                    valid_sets = [train_data]
                    valid_names = ["train"]
                    callbacks = [lgb.log_evaluation(period=0)]
                    
                    if y_val_h is not None:
                        val_data = lgb.Dataset(val_feat, label=y_val_h, reference=train_data)
                        valid_sets.append(val_data)
                        valid_names.append("valid")
                        callbacks.append(lgb.early_stopping(stopping_rounds=30, verbose=False))
                        
                    model = lgb.train(
                        cand['params'], train_data,
                        num_boost_round=500,
                        valid_sets=valid_sets,
                        valid_names=valid_names,
                        callbacks=callbacks,
                    )
                    
                    # Evaluate on validation set if available
                    if y_val_h is not None:
                        pred_val_transformed = model.predict(val_feat)
                        # Inverse transform
                        pred_val_return = pred_val_transformed * scale + baseline
                        
                        y_val_orig = y_val[:, i]
                        val_return_mae = np.mean(np.abs(y_val_orig - pred_val_return))
                        
                        val_pred_up = (pred_val_return > 0).astype(int)
                        val_actual_up = (y_val_orig > 0).astype(int)
                        val_pred_up_rate = np.mean(val_pred_up)
                        val_actual_up_rate = np.mean(val_actual_up)
                        val_dir_acc = np.mean(val_pred_up == val_actual_up)
                        
                        cand_result = {
                            'horizon': i + 1,
                            'target_transform': t_name,
                            'objective_name': cand['name'],
                            'baseline': baseline,
                            'scale': scale,
                            'val_return_mae': val_return_mae,
                            'val_directional_accuracy': val_dir_acc,
                            'val_pred_up_rate': val_pred_up_rate,
                            'val_actual_up_rate': val_actual_up_rate,
                            'best_iteration': model.best_iteration if hasattr(model, 'best_iteration') else -1,
                            'selected': False
                        }
                        
                        # Selection logic
                        is_best = False
                        if val_return_mae < best_val_mae - 1e-6:
                            is_best = True
                        elif abs(val_return_mae - best_val_mae) <= 1e-6:
                            # Tie-breaker 1: directional accuracy
                            if cand_result['val_directional_accuracy'] > best_meta['val_directional_accuracy'] + 1e-4:
                                is_best = True
                            elif abs(cand_result['val_directional_accuracy'] - best_meta['val_directional_accuracy']) <= 1e-4:
                                # Tie-breaker 2: pred_up_rate closer to actual
                                diff_new = abs(cand_result['val_pred_up_rate'] - cand_result['val_actual_up_rate'])
                                diff_old = abs(best_meta['val_pred_up_rate'] - best_meta['val_actual_up_rate'])
                                if diff_new < diff_old - 1e-4:
                                    is_best = True
                                elif abs(diff_new - diff_old) <= 1e-4:
                                    # Tie-breaker 3: fewer boosting rounds
                                    if cand_result['best_iteration'] < best_meta['best_iteration']:
                                        is_best = True
                                        
                        if is_best or best_model is None:
                            best_val_mae = val_return_mae
                            best_model = model
                            best_meta = cand_result
                            
                        all_candidates_results.append(cand_result)
                    else:
                        # No validation data, just use the first candidate we train
                        best_model = model
                        best_meta = {
                            'horizon': i + 1,
                            'target_transform': t_name,
                            'objective_name': cand['name'],
                            'baseline': baseline,
                            'scale': scale,
                            'val_return_mae': np.nan,
                            'val_directional_accuracy': np.nan,
                            'val_pred_up_rate': np.nan,
                            'val_actual_up_rate': np.nan,
                            'best_iteration': model.best_iteration if hasattr(model, 'best_iteration') else -1,
                            'selected': True
                        }
                        all_candidates_results.append(best_meta)
                        break # Break lgbm_candidates loop
                if y_val_h is None:
                    break # Break transforms loop
            
            # Save best for this horizon
            if best_meta is not None:
                best_meta['selected'] = True
                selected_variants_results.append(best_meta)
                if verbose:
                    if y_val_h is not None:
                        print(f"    Selected: {best_meta['objective_name']} with {best_meta['target_transform']} | Val MAE: {best_meta['val_return_mae']:.5f} | Dir Acc: {best_meta['val_directional_accuracy']:.4f}")
                    else:
                        print(f"    Selected: {best_meta['objective_name']} with {best_meta['target_transform']}")
            self.models.append(best_model)
            self.regression_model_metadata.append(best_meta)
            
            # Directional Classifier
            if use_direction_classifier:
                if verbose:
                    print(f"  -> Training Classifier for Day {i+1}...")
                y_dir_train = (y_labels[:, i] > 0).astype(int)
                num_pos = np.sum(y_dir_train)
                num_neg = len(y_dir_train) - num_pos
                
                if num_pos == 0 or num_neg == 0:
                    if verbose:
                        print(f"  -> Skipping Classifier Day {i+1}: Missing classes.")
                    self.classifiers.append(None)
                    continue
                    
                scale_pos_weight = num_neg / num_pos
                
                clf_params = {
                    'objective': 'binary',
                    'metric': 'binary_logloss',
                    'learning_rate': 0.03,
                    'num_leaves': 7,
                    'max_depth': 3,
                    'min_data_in_leaf': 30,
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'lambda_l1': 1.0,
                    'lambda_l2': 2.0,
                    'verbose': -1,
                    'scale_pos_weight': scale_pos_weight
                }
                
                train_data_clf = lgb.Dataset(train_feat, label=np.ascontiguousarray(y_dir_train))
                valid_sets_clf = [train_data_clf]
                valid_names_clf = ["train"]
                callbacks_clf = [lgb.log_evaluation(period=0)]
                
                if val_loader is not None and y_val is not None:
                    y_dir_val = (y_val[:, i] > 0).astype(int)
                    val_data_clf = lgb.Dataset(val_feat, label=np.ascontiguousarray(y_dir_val), reference=train_data_clf)
                    valid_sets_clf.append(val_data_clf)
                    valid_names_clf.append("valid")
                    callbacks_clf.append(lgb.early_stopping(stopping_rounds=30, verbose=False))
                    
                model_clf = lgb.train(
                    clf_params, train_data_clf,
                    num_boost_round=500,
                    valid_sets=valid_sets_clf,
                    valid_names=valid_names_clf,
                    callbacks=callbacks_clf,
                )
                self.classifiers.append(model_clf)
                if verbose:
                    if hasattr(model_clf, 'best_iteration') and model_clf.best_iteration > 0:
                        print(f"  -> Classifier Day {i+1} Best Iteration: {model_clf.best_iteration}")

        if output_dir and all_candidates_results:
            pd.DataFrame(all_candidates_results).to_csv(os.path.join(output_dir, "regression_variant_candidates.csv"), index=False)
            pd.DataFrame(selected_variants_results).to_csv(os.path.join(output_dir, "regression_variant_selection.csv"), index=False)

        if verbose:
            print(">>> Hybrid Multi-step Training Complete!")

    # ------------------------------------------------------------------
    #  End-to-End fit & predict
    # ------------------------------------------------------------------

    def fit(self, train_loader, y_labels, val_loader=None, y_val=None,
            external_features=None, tm_epochs=10, output_dir=None,
            train_dataset=None, val_dataset=None, use_raw_stats=True, use_direction_classifier=True,
            use_regression_variant_search=False, verbose=False):
        """End-to-End training pipeline."""
        self.fit_timemixer(train_loader, val_loader=val_loader, epochs=tm_epochs, verbose=verbose)
        self.fit_lgbm(train_loader, y_labels, val_loader=val_loader, y_val=y_val,
                      external_features=external_features, output_dir=output_dir,
                      train_dataset=train_dataset, val_dataset=val_dataset, 
                      use_raw_stats=use_raw_stats, use_direction_classifier=use_direction_classifier,
                      use_regression_variant_search=use_regression_variant_search,
                      verbose=verbose)

    def predict(self, test_loader, external_features=None, output_dir=None, y_test=None,
                test_dataset=None, use_raw_stats=True, split_name="test", verbose=False) -> np.ndarray:
        """Run full hybrid pipeline: extract features then predict with LightGBM. Returns [N, pred_len]."""
        if not self.models:
            raise ValueError("LightGBM models are not trained yet! Call fit() first.")

        test_feat, n_lat, n_pred, n_stat = self._build_features(test_loader, external_features, dataset=test_dataset, use_raw_stats=use_raw_stats)
        
        self._diagnose_and_save_features(test_feat, y_test, n_lat, n_pred, n_stat, split_name, output_dir, is_raw=(use_raw_stats and test_dataset is not None), verbose=verbose)
        
        preds = []
        for i, model in enumerate(self.models):
            pred_h = model.predict(test_feat)
            if hasattr(self, 'regression_model_metadata') and i < len(self.regression_model_metadata) and self.regression_model_metadata[i] is not None:
                meta = self.regression_model_metadata[i]
                baseline = meta.get('baseline', 0.0)
                scale = meta.get('scale', 1.0)
                pred_h = pred_h * scale + baseline
            preds.append(pred_h)
            
        # Stack to [N, pred_len]
        return np.column_stack(preds)

    def predict_direction_proba(self, test_loader, external_features=None, test_dataset=None, use_raw_stats=True, verbose=False) -> np.ndarray:
        """Predict P(price goes up) per horizon day using the direction classifiers. Returns [N, pred_len]."""
        if not self.classifiers:
            raise ValueError("LightGBM classifiers are not trained yet! Ensure use_direction_classifier=True during fit.")
            
        test_feat, _, _, _ = self._build_features(test_loader, external_features, dataset=test_dataset, use_raw_stats=use_raw_stats)
        
        preds = []
        for clf in self.classifiers:
            if clf is None:
                # If skipping due to missing classes during training, default to 0.5 or 1.0 depending on logic, let's use 0.5
                preds.append(np.full(len(test_feat), 0.5))
            else:
                preds.append(clf.predict(test_feat))
                
        return np.column_stack(preds)
