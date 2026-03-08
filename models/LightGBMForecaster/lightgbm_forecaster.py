"""
LightGBM-based primary forecaster for stock price prediction.

Uses gradient boosting on hand-crafted tabular features (lagged prices,
rolling statistics, technical indicators) to predict High, Close, and
optional moving averages (EMA_20, SMA_50).

Architecture
------------
- **Delta-based prediction**: Instead of predicting absolute prices
  (which is impossible from scale-invariant features), the model
  predicts the *change* from the last known Close price.  At inference
  the predicted delta is added back to the anchor Close to recover
  the absolute price.
- **Direct multi-step strategy**: For each (forecast step, target) pair,
  a separate LightGBM regressor is trained.  With ``pred_len=5`` and 4
  targets this produces 20 small models, each training in milliseconds.
- **Feature engineering**: Converts the raw OHLCV lookback window into
  ~35-40 scale-invariant tabular features (returns, ratios, RSI, MACD,
  ATR, Bollinger width, rolling stats, price position).
- **Output format**: Predictions are shaped ``[N, pred_len, n_targets]``
  in original price scale, identical to the DL models, so the downstream
  meta-labeling pipeline works unchanged.

Role in the pipeline
--------------------
    AAPL.csv  -->  LightGBMForecaster.fit()
                          |
                   LightGBMForecaster.predict()
                          |
                          v
                   predictions.npy  (same shape as DL model output)
                          |
                          v
                   generate_meta_labels.py  (unchanged)
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
import os
from typing import Dict, List, Optional, Any


class LightGBMForecaster:
    """
    Multi-target, multi-step LightGBM forecaster for stock prices.

    Trains one ``lgb.LGBMRegressor`` per (forecast step, target) pair
    using a direct forecasting strategy.  Features are engineered from
    raw OHLCV data so the model benefits from domain knowledge that DL
    models must learn from scratch.

    Attributes
    ----------
    seq_len : int
        Number of historical bars used to compute features.
    pred_len : int
        Number of future bars to predict.
    target_features : List[str]
        Ordered list of target column names.
    feature_names : List[str]
        Names of the engineered features (set after ``fit``).
    models : Dict[tuple, lgb.LGBMRegressor]
        Trained regressors keyed by ``(step, target_name)``.
    """

    DEFAULT_PARAMS: Dict[str, Any] = {
        "objective": "regression",
        "metric": "mae",
        "boosting_type": "gbdt",
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": -1,
    }

    def __init__(
        self,
        seq_len: int = 30,
        pred_len: int = 5,
        lgb_params: Optional[Dict[str, Any]] = None,
    ):
        self.seq_len = seq_len
        self.pred_len = pred_len

        self._params = self.DEFAULT_PARAMS.copy()
        if lgb_params is not None:
            self._params.update(lgb_params)

        self.target_features: List[str] = []
        self.feature_names: List[str] = []
        self.models: Dict[tuple, lgb.LGBMRegressor] = {}

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create tabular features from an OHLCV DataFrame.

        All features are backward-looking (no future leakage).  Most are
        scale-invariant (ratios, returns, bounded oscillators) so the
        model generalises across different price levels.

        Args:
            df: DataFrame with at least Open, High, Low, Close, Volume.

        Returns:
            DataFrame of engineered features aligned to the same index.
        """
        feat = pd.DataFrame(index=df.index)

        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        open_ = df["Open"]
        volume = df["Volume"]

        # -- Returns at key lags --
        for lag in [1, 2, 3, 5, 10, 20]:
            feat[f"ret_{lag}d"] = close.pct_change(lag)

        # -- Intraday price ratios --
        feat["high_close_ratio"] = high / close
        feat["low_close_ratio"] = low / close
        feat["open_close_ratio"] = open_ / close
        feat["high_low_range"] = (high - low) / close

        # -- Volume dynamics --
        feat["vol_change_1d"] = volume.pct_change(1)
        feat["vol_change_5d"] = volume.pct_change(5)
        vol_ma20 = volume.rolling(20, min_periods=1).mean()
        feat["vol_ma_ratio"] = volume / vol_ma20.replace(0, np.nan)

        # -- Rolling return statistics --
        rets = close.pct_change()
        for w in [5, 10, 20]:
            feat[f"ret_mean_{w}d"] = rets.rolling(w, min_periods=w).mean()
            feat[f"ret_std_{w}d"] = rets.rolling(w, min_periods=w).std()

        # -- Price position within recent range --
        for w in [10, 20]:
            roll_high = high.rolling(w, min_periods=w).max()
            roll_low = low.rolling(w, min_periods=w).min()
            denom = (roll_high - roll_low).replace(0, np.nan)
            feat[f"price_pos_{w}d"] = (close - roll_low) / denom

        # -- RSI (14) --
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss_val = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        avg_loss = loss_val.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        feat["rsi"] = 100.0 - (100.0 / (1.0 + rs))

        # -- MACD (12, 26, 9) --
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        feat["macd"] = (ema12 - ema26) / close
        feat["macd_signal"] = feat["macd"].ewm(span=9, adjust=False).mean()
        feat["macd_hist"] = feat["macd"] - feat["macd_signal"]

        # -- ATR (14), normalised --
        prev_close = close.shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        feat["atr"] = tr.rolling(14, min_periods=14).mean() / close

        # -- Bollinger Band width --
        sma20 = close.rolling(20, min_periods=20).mean()
        std20 = close.rolling(20, min_periods=20).std()
        feat["bb_width"] = (2 * std20) / sma20.replace(0, np.nan)

        # -- Vwap / Transactions (when available) --
        if "Vwap" in df.columns:
            feat["vwap_ratio"] = df["Vwap"] / close
        if "Transactions" in df.columns:
            txn = df["Transactions"].replace(0, np.nan)
            feat["txn_change"] = df["Transactions"].pct_change(1)
            feat["avg_trade_size"] = volume / txn

        # -- Day of week (when Date column exists) --
        if "Date" in df.columns:
            feat["day_of_week"] = pd.to_datetime(df["Date"]).dt.dayofweek

        return feat

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        target_features: List[str],
        early_stopping_rounds: int = 50,
    ) -> "LightGBMForecaster":
        """
        Train the forecaster on raw OHLCV DataFrames.

        Args:
            df_train: Training split (raw OHLCV + optional MA columns).
            df_val: Validation split (immediately follows training data).
            target_features: Ordered target column names
                (e.g. ``['High', 'Close', 'EMA_20', 'SMA_50']``).
            early_stopping_rounds: Stop if val MAE doesn't improve.

        Returns:
            self
        """
        self.target_features = target_features

        # Concatenate train+val for feature engineering (so rolling
        # windows at the start of val are computed correctly), then
        # split back.
        df_full = pd.concat([df_train, df_val], ignore_index=True)
        feat_full = self.engineer_features(df_full)
        self.feature_names = list(feat_full.columns)

        n_train = len(df_train)
        n_val = len(df_val)

        # Usable indices: must have seq_len history AND pred_len future
        train_start = self.seq_len
        train_end = n_train - self.pred_len
        val_start = max(n_train, self.seq_len)
        val_end = n_train + n_val - self.pred_len

        if train_end <= train_start:
            raise ValueError(
                f"Not enough training data: need at least "
                f"seq_len({self.seq_len}) + pred_len({self.pred_len}) rows."
            )

        X_feat = feat_full.values.astype(np.float64)

        X_feat = np.nan_to_num(X_feat, nan=0.0, posinf=0.0, neginf=0.0)

        X_train = pd.DataFrame(X_feat[train_start:train_end], columns=self.feature_names)
        X_val = (pd.DataFrame(X_feat[val_start:val_end], columns=self.feature_names)
                 if val_end > val_start else None)

        n_models = self.pred_len * len(target_features)

        print(f"\n{'='*60}")
        print(f"LightGBM Forecaster Training")
        print(f"{'='*60}")
        print(f"Features: {len(self.feature_names)}")
        print(f"Train samples: {len(X_train)}")
        if X_val is not None:
            print(f"Val samples:   {len(X_val)}")
        print(f"Targets: {target_features}")
        print(f"Pred steps: {self.pred_len}  |  Models to train: {n_models}")
        print(f"{'='*60}\n")

        self.models = {}

        # Anchor price: the Close at each sample position.
        # Targets become deltas: target_price - anchor.
        anchor_col = df_full["Close"].values

        for step in range(1, self.pred_len + 1):
            for target_name in target_features:
                target_col = df_full[target_name].values

                anchor_train = anchor_col[train_start:train_end]
                y_train = (target_col[train_start + step : train_end + step]
                           - anchor_train)

                model = lgb.LGBMRegressor(**self._params)

                fit_kwargs: Dict[str, Any] = {}
                if X_val is not None and val_end > val_start:
                    anchor_val = anchor_col[val_start:val_end]
                    y_val = (target_col[val_start + step : val_end + step]
                             - anchor_val)
                    fit_kwargs["eval_set"] = [(X_val, y_val)]
                    fit_kwargs["callbacks"] = [
                        lgb.early_stopping(early_stopping_rounds, verbose=False),
                        lgb.log_evaluation(period=0),
                    ]

                model.fit(X_train, y_train, **fit_kwargs)

                key = (step, target_name)
                self.models[key] = model

                best_iter = getattr(model, "best_iteration_", model.n_estimators)
                val_score = ""
                if X_val is not None:
                    anchor_val = anchor_col[val_start:val_end]
                    delta_pred = model.predict(X_val)
                    abs_pred = delta_pred + anchor_val
                    abs_true = target_col[val_start + step : val_end + step]
                    mae = np.mean(np.abs(abs_pred - abs_true))
                    val_score = f"  val_MAE=${mae:.2f}"

                print(f"  step={step} target={target_name:8s}  "
                      f"best_iter={best_iter}{val_score}")

        print(f"\nTraining complete. {len(self.models)} models trained.\n")
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict future values from raw OHLCV data.

        Internally the model predicts deltas from the anchor Close
        price, then converts back to absolute prices.

        Args:
            df: DataFrame with OHLCV (+ optional extended) columns.
                Must have at least ``seq_len`` rows of history before
                the first predictable position.

        Returns:
            np.ndarray of shape ``[N, pred_len, n_targets]`` in
            original price scale.  ``N`` is the number of valid
            prediction positions (rows where both enough history
            and enough future exist).
        """
        feat = self.engineer_features(df)
        X = feat.values.astype(np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        start = self.seq_len
        end = len(df) - self.pred_len
        n_samples = end - start

        if n_samples <= 0:
            raise ValueError("Not enough data for prediction.")

        X_pred = pd.DataFrame(X[start:end], columns=self.feature_names)
        n_targets = len(self.target_features)

        # Anchor: Close price at each prediction position
        anchor = df["Close"].values[start:end]

        preds = np.zeros((n_samples, self.pred_len, n_targets), dtype=np.float64)

        for step in range(1, self.pred_len + 1):
            for t_idx, target_name in enumerate(self.target_features):
                model = self.models[(step, target_name)]
                delta = model.predict(X_pred)
                preds[:, step - 1, t_idx] = anchor + delta

        return preds

    def get_ground_truth(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract ground-truth target values aligned to prediction positions.

        Returns:
            np.ndarray of shape ``[N, pred_len, n_targets]``.
        """
        start = self.seq_len
        end = len(df) - self.pred_len
        n_samples = end - start
        n_targets = len(self.target_features)

        trues = np.zeros((n_samples, self.pred_len, n_targets), dtype=np.float64)

        for step in range(1, self.pred_len + 1):
            for t_idx, target_name in enumerate(self.target_features):
                trues[:, step - 1, t_idx] = (
                    df[target_name].values[start + step : end + step]
                )

        return trues

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def feature_importance(
        self, importance_type: str = "gain",
    ) -> Dict[str, float]:
        """
        Aggregate feature importances across all sub-models.

        Returns:
            Dict mapping feature names to summed importance scores,
            sorted descending.
        """
        totals = np.zeros(len(self.feature_names), dtype=np.float64)

        for model in self.models.values():
            totals += model.feature_importances_

        name_score = dict(zip(self.feature_names, totals))
        return dict(sorted(name_score.items(), key=lambda x: x[1], reverse=True))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save all sub-models and metadata to a single joblib file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "seq_len": self.seq_len,
            "pred_len": self.pred_len,
            "target_features": self.target_features,
            "feature_names": self.feature_names,
            "params": self._params,
            "models": self.models,
        }
        joblib.dump(payload, path)
        print(f"LightGBM forecaster saved to {path}")

    @classmethod
    def load(cls, path: str) -> "LightGBMForecaster":
        """Load a previously saved forecaster."""
        payload = joblib.load(path)
        instance = cls(
            seq_len=payload["seq_len"],
            pred_len=payload["pred_len"],
            lgb_params=payload["params"],
        )
        instance.target_features = payload["target_features"]
        instance.feature_names = payload["feature_names"]
        instance.models = payload["models"]
        print(f"LightGBM forecaster loaded from {path}")
        return instance
