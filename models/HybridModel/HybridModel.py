import torch
import torch.nn as nn
import lightgbm as lgb
import numpy as np
from typing import Optional, List
import copy

from models.TimeMixer.TimeMixer import TimeMixer, TimeMixerConfig


class HybridTimeMixerLGBM:
    """
    End-to-End Hybrid Model combining TimeMixer and LightGBM.

    Workflow
    --------
    1. Train TimeMixer for time-series forecasting (regression).
    2. Extract deep latent embeddings from the trained TimeMixer.
    3. Extract TimeMixer's actual predictions as additional features.
    4. Compute rich hand-crafted statistical/technical features from raw sequences.
    5. Concatenate all feature groups and train LightGBM for classification.
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

    # ------------------------------------------------------------------
    #  Phase 1 – TimeMixer Training (with best-model checkpoint)
    # ------------------------------------------------------------------

    def fit_timemixer(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        epochs: int = 10,
        lr: float = 1e-3,
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

            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train Loss: {avg_train:.4f}{val_msg} | "
                f"LR: {current_lr:.6f}"
            )

        # Restore best model
        if best_state is not None:
            self.timemixer.load_state_dict(best_state)
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
        """Extract deep latent embeddings from TimeMixer's penultimate layer."""
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

    def _build_features(self, data_loader, external_features=None):
        """Build the full feature matrix for a given DataLoader."""
        f_latent = self.extract_latent_features(data_loader)
        f_pred = self.extract_prediction_features(data_loader)
        f_stat = self.compute_statistical_features(data_loader)

        parts = [f_latent, f_pred, f_stat]
        if external_features is not None:
            parts.append(external_features)

        combined = np.hstack(parts)
        return combined, f_latent.shape[1], f_pred.shape[1], f_stat.shape[1]

    # ------------------------------------------------------------------
    #  Phase 3 – LightGBM Training
    # ------------------------------------------------------------------

    def fit_lgbm(
        self,
        train_loader, y_labels,
        val_loader=None, y_val=None,
        external_features=None,
    ) -> None:
        """
        Phase 3: Train LightGBM using combined features.
        Trains a separate LightGBM model for each day in the forecast horizon.
        """
        print("\n>>> Phase 2: Extracting features from TimeMixer...")
        train_feat, n_lat, n_pred, n_stat = self._build_features(train_loader, external_features)

        pred_len = y_labels.shape[1]
        self.models = []

        print(
            f"\n>>> Phase 3: Training {pred_len} LightGBM Models (Multi-step Forecast)... "
            f"(Latent: {n_lat} + Pred: {n_pred} + Stat: {n_stat}"
            f" = {train_feat.shape[1]} total features)"
        )

        if val_loader is not None and y_val is not None:
            val_feat, _, _, _ = self._build_features(val_loader)

        for i in range(pred_len):
            print(f"  -> Training Model for Day {i+1}...")
            # Use ascontiguousarray to prevent LightGBM warning about memory double cost
            train_data = lgb.Dataset(train_feat, label=np.ascontiguousarray(y_labels[:, i]))

            callbacks = [lgb.log_evaluation(period=0)]
            valid_sets = [train_data]
            valid_names = ["train"]

            if val_loader is not None and y_val is not None:
                val_data = lgb.Dataset(val_feat, label=np.ascontiguousarray(y_val[:, i]), reference=train_data)
                valid_sets.append(val_data)
                valid_names.append("valid")
                # Add early stopping for each day's model independently
                callbacks.append(lgb.early_stopping(stopping_rounds=30, verbose=False))

            model = lgb.train(
                self.lgbm_params, train_data,
                num_boost_round=500,
                valid_sets=valid_sets,
                valid_names=valid_names,
                callbacks=callbacks,
            )
            self.models.append(model)
            
        print(">>> Hybrid Multi-step Training Complete!")

    # ------------------------------------------------------------------
    #  End-to-End fit & predict
    # ------------------------------------------------------------------

    def fit(self, train_loader, y_labels, val_loader=None, y_val=None,
            external_features=None, tm_epochs=10):
        """End-to-End training pipeline."""
        self.fit_timemixer(train_loader, val_loader=val_loader, epochs=tm_epochs)
        self.fit_lgbm(train_loader, y_labels, val_loader=val_loader, y_val=y_val,
                      external_features=external_features)

    def predict(self, test_loader, external_features=None) -> np.ndarray:
        """Generates multi-step predictions using the complete hybrid pipeline."""
        if not self.models:
            raise ValueError("LightGBM models are not trained yet! Call fit() first.")

        test_feat, _, _, _ = self._build_features(test_loader, external_features)
        
        preds = []
        for model in self.models:
            preds.append(model.predict(test_feat))
            
        # Stack to [N, pred_len]
        return np.column_stack(preds)
