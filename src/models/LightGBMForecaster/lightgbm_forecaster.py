"""
LightGBM-based primary forecaster for stock price prediction.

Uses gradient boosting on hand-crafted tabular features (lagged prices,
rolling statistics, technical indicators) to predict High, Close, and
optional moving averages (EMA_20, SMA_50).

Architecture
------------
- **Return-based prediction**: Instead of predicting absolute prices
  (which is impossible from scale-invariant features), the model
  predicts the *percentage return* from the last known Close price:
  ``(future - anchor) / anchor``.  At inference the predicted return
  is applied to the anchor Close to recover the absolute price:
  ``anchor * (1 + predicted_return)``.
- **Direct multi-step strategy**: For each (forecast step, target) pair,
  a separate LightGBM regressor is trained.  With ``pred_len=5`` and 4
  targets this produces 20 small models, each training in milliseconds.
- **Feature engineering**: Derives ~28 mostly scale-invariant tabular
  features from the OHLCV history (returns, ratios, RSI, MACD, ATR,
  Bollinger width, rolling stats, price position, day-of-week). A few
  extra features appear only if legacy Vwap/Transactions columns exist.
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
                   meta/generate.py  (unchanged)
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
        "n_estimators": 2000,
        "learning_rate": 0.01,
        "num_leaves": 15,
        "max_depth": 5,
        "min_child_samples": 20,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "reg_alpha": 1.0,
        "reg_lambda": 1.0,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": -1,
    }

    def __init__(
        self,
        seq_len: int = 30,
        pred_len: int = 5,
        lgb_params: Optional[Dict[str, Any]] = None,
        target_mode: str = "price",
    ):
        """
        target_mode:
          "price" — legacy: predict next-``pred_len`` High & Close prices as
                    per-step returns off the Close[t] anchor (feeds Stage-2 as-is).
          "range" — predict the forward-averaged upside/downside band, vol-
                    normalised, off the Close[t] anchor:
                      up = (mean(High[t+1..t+H]) - Close[t]) / Close[t] / sigma[t]
                      dn = (mean(Low [t+1..t+H]) - Close[t]) / Close[t] / sigma[t]
                    sigma = volatility_20. One model per channel (upside/downside).
        """
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.target_mode = target_mode

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
        Select the canonical causal feature set (see :mod:`features`).

        The features are the single source of truth shared with the neural
        models, so the comparison is on identical inputs. They are normally
        pre-computed in the ticker CSV; if a caller passes a bare OHLCV frame
        we derive them on the fly.

        Args:
            df: DataFrame with the :data:`features.FEATURE_COLUMNS` columns, or
                at least ``Open, High, Low, Close, Volume`` to derive them.

        Returns:
            DataFrame of features aligned positionally to ``df``.
        """
        from features import FEATURE_COLUMNS, add_features

        if not set(FEATURE_COLUMNS).issubset(df.columns):
            df = add_features(df, drop_warmup=False)
        # Raw OHLCV + the canonical features — the same columns the neural nets
        # receive, so the LightGBM-vs-neural comparison is on identical inputs.
        cols = ["Open", "High", "Low", "Close", "Volume"] + list(FEATURE_COLUMNS)
        feat = df[cols].copy()

        # Optional cross-sectional / market context (panel add-on, opt-in).
        from features_cross import CROSS_FEATURE_COLUMNS
        for col in CROSS_FEATURE_COLUMNS:
            if col in df.columns:
                feat[col] = df[col].values

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
        Train the forecaster on one ticker's raw OHLCV DataFrames.

        Targets are percentage returns: (future_price - anchor) / anchor.
        This makes targets scale-invariant, matching the features.
        """
        self.target_features = target_features
        arrays = self._build_arrays(df_train, df_val, target_features)
        self._print_training_header([arrays])
        self._fit_from_arrays(arrays, early_stopping_rounds)
        return self

    def fit_pooled(
        self,
        train_dfs: List[pd.DataFrame],
        val_dfs: List[pd.DataFrame],
        target_features: List[str],
        early_stopping_rounds: int = 50,
        ticker_labels: Optional[List[str]] = None,
    ) -> "LightGBMForecaster":
        """
        Train one model on multiple tickers without leaking across boundaries.

        Feature engineering (rolling / ewm / lag) and target shifting are
        performed **inside each ticker's frame** first — only the resulting
        (X, y) arrays are concatenated.  Naively concatenating the raw
        DataFrames instead makes the last rows of ticker A pull rolling
        state and even future-target values out of ticker B.
        """
        assert len(train_dfs) == len(val_dfs)
        self.target_features = target_features

        per_ticker = [
            self._build_arrays(dt, dv, target_features)
            for dt, dv in zip(train_dfs, val_dfs)
        ]
        pooled = self._pool_arrays(per_ticker)
        self._print_training_header(per_ticker, ticker_labels=ticker_labels)
        self._fit_from_arrays(pooled, early_stopping_rounds)
        return self

    # ------------------------------------------------------------------
    # Internals shared by fit / fit_pooled
    # ------------------------------------------------------------------

    def _build_arrays(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        target_features: List[str],
    ) -> Dict[str, Any]:
        """Compute features and returns for one ticker's train+val split.

        Feature engineering runs on ``concat(df_train, df_val)`` so that
        rolling windows at the start of val are seeded by the tail of
        train — this is safe because both come from the same ticker.
        """
        df_full = pd.concat([df_train, df_val], ignore_index=True)
        feat_full = self.engineer_features(df_full)
        feature_names = list(feat_full.columns)

        n_train = len(df_train)
        n_val = len(df_val)

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

        X_train = X_feat[train_start:train_end]
        X_val = X_feat[val_start:val_end] if val_end > val_start else None

        anchor_col = df_full["Close"].values
        anchor_train = anchor_col[train_start:train_end]
        anchor_val = anchor_col[val_start:val_end] if val_end > val_start else None

        y_train: Dict[tuple, np.ndarray] = {}
        y_val: Dict[tuple, np.ndarray] = {}
        y_val_abs: Dict[tuple, np.ndarray] = {}

        for step in range(1, self.pred_len + 1):
            for target_name in target_features:
                target_col = df_full[target_name].values
                y_train[(step, target_name)] = (
                    (target_col[train_start + step:train_end + step] - anchor_train)
                    / anchor_train
                )
                if X_val is not None:
                    abs_true = target_col[val_start + step:val_end + step]
                    y_val_abs[(step, target_name)] = abs_true
                    y_val[(step, target_name)] = (abs_true - anchor_val) / anchor_val

        return {
            "X_train": X_train,
            "X_val": X_val,
            "anchor_val": anchor_val,
            "y_train": y_train,
            "y_val": y_val,
            "y_val_abs": y_val_abs,
            "feature_names": feature_names,
        }

    # ------------------------------------------------------------------
    # Range mode (upside/downside band, vol-normalised) — self-contained
    # path so the legacy "price" mode is untouched.
    # ------------------------------------------------------------------

    def _build_range_arrays(self, df_train: pd.DataFrame, df_val: pd.DataFrame) -> Dict[str, Any]:
        """Per-ticker (X, upside, downside) arrays for train and val splits."""
        df = pd.concat([df_train, df_val], ignore_index=True)
        feat = self.engineer_features(df)
        X = np.nan_to_num(feat.values.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        c = df["Close"].values.astype(np.float64)
        h = df["High"].values.astype(np.float64)
        lo = df["Low"].values.astype(np.float64)
        sig = df["volatility_20"].values.astype(np.float64)
        H = self.pred_len

        n_train, n_val = len(df_train), len(df_val)
        blocks = {
            "train": range(self.seq_len, n_train - H),
            "val":   range(max(n_train, self.seq_len), n_train + n_val - H),
        }
        out = {"feature_names": list(feat.columns)}
        for name, rng in blocks.items():
            Xs, ups, dns = [], [], []
            for i in rng:
                s = sig[i]
                if not np.isfinite(s) or s <= 0 or i + H >= len(df):
                    continue
                up = (h[i+1:i+1+H].mean() - c[i]) / c[i] / s
                dn = (lo[i+1:i+1+H].mean() - c[i]) / c[i] / s
                Xs.append(X[i]); ups.append(up); dns.append(dn)
            out[f"X_{name}"] = np.array(Xs) if Xs else None
            out[f"up_{name}"] = np.array(ups)
            out[f"dn_{name}"] = np.array(dns)
        return out

    def fit_pooled_range(self, train_dfs, val_dfs, early_stopping_rounds=200,
                         ticker_labels=None):
        """Train upside & downside regressors pooled over many tickers."""
        assert len(train_dfs) == len(val_dfs)
        self.target_features = ["upside", "downside"]
        per = [self._build_range_arrays(dt, dv) for dt, dv in zip(train_dfs, val_dfs)]
        self.feature_names = per[0]["feature_names"]

        def cat(key):
            parts = [p[key] for p in per if p[key] is not None and len(p[key])]
            return np.concatenate(parts) if parts else None
        Xtr, Xva = cat("X_train"), cat("X_val")
        ytr = {"upside": cat("up_train"), "downside": cat("dn_train")}
        yva = {"upside": cat("up_val"),   "downside": cat("dn_val")}

        print(f"\n{'='*60}\nLightGBM RANGE mode | features={len(self.feature_names)} "
              f"| train={len(Xtr)} val={0 if Xva is None else len(Xva)}\n{'='*60}")
        self.models = {}
        Xtr_df = pd.DataFrame(Xtr, columns=self.feature_names)
        Xva_df = pd.DataFrame(Xva, columns=self.feature_names) if Xva is not None else None
        for ch in self.target_features:
            m = lgb.LGBMRegressor(**self._params)
            kw = {}
            if Xva_df is not None:
                kw = dict(eval_set=[(Xva_df, yva[ch])],
                          callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False),
                                     lgb.log_evaluation(0)])
            m.fit(Xtr_df, ytr[ch], **kw)
            self.models[("range", ch)] = m
            print(f"  {ch:9s} best_iter={getattr(m,'best_iteration_',m.n_estimators)}")
        return self

    def predict_range(self, df: pd.DataFrame):
        """Return (preds[N,2], anchor[N], sigma[N]); channels = [upside, downside]."""
        feat = self.engineer_features(df)
        X = np.nan_to_num(feat.values.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        start, end = self.seq_len, len(df) - self.pred_len
        Xp = pd.DataFrame(X[start:end], columns=self.feature_names)
        anchor = df["Close"].values[start:end].astype(np.float64)
        sigma = df["volatility_20"].values[start:end].astype(np.float64)
        preds = np.column_stack([
            self.models[("range", "upside")].predict(Xp),
            self.models[("range", "downside")].predict(Xp),
        ])
        return preds, anchor, sigma

    def range_ground_truth(self, df: pd.DataFrame) -> np.ndarray:
        """True vol-normalised [upside, downside] aligned to predict_range."""
        c = df["Close"].values.astype(np.float64)
        h = df["High"].values.astype(np.float64)
        lo = df["Low"].values.astype(np.float64)
        sig = df["volatility_20"].values.astype(np.float64)
        H = self.pred_len
        start, end = self.seq_len, len(df) - self.pred_len
        up = np.array([(h[i+1:i+1+H].mean() - c[i]) / c[i] / sig[i] for i in range(start, end)])
        dn = np.array([(lo[i+1:i+1+H].mean() - c[i]) / c[i] / sig[i] for i in range(start, end)])
        return np.column_stack([up, dn])

    def _pool_arrays(self, per_ticker: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Concatenate the per-ticker (X, y) arrays for a joint fit."""
        feature_names = per_ticker[0]["feature_names"]
        for pt in per_ticker[1:]:
            if pt["feature_names"] != feature_names:
                raise ValueError(
                    "Ticker feature sets differ — cannot pool. "
                    "Check that all tickers have the same optional columns "
                    "(Vwap, Transactions, Date, MA targets)."
                )

        X_train = np.concatenate([pt["X_train"] for pt in per_ticker], axis=0)
        val_chunks = [pt["X_val"] for pt in per_ticker if pt["X_val"] is not None]
        X_val = np.concatenate(val_chunks, axis=0) if val_chunks else None

        anchor_val_chunks = [pt["anchor_val"] for pt in per_ticker
                             if pt["anchor_val"] is not None]
        anchor_val = np.concatenate(anchor_val_chunks) if anchor_val_chunks else None

        y_train: Dict[tuple, np.ndarray] = {}
        y_val: Dict[tuple, np.ndarray] = {}
        y_val_abs: Dict[tuple, np.ndarray] = {}
        keys = per_ticker[0]["y_train"].keys()
        for key in keys:
            y_train[key] = np.concatenate([pt["y_train"][key] for pt in per_ticker])
            val_ys = [pt["y_val"][key] for pt in per_ticker if key in pt["y_val"]]
            if val_ys:
                y_val[key] = np.concatenate(val_ys)
                y_val_abs[key] = np.concatenate(
                    [pt["y_val_abs"][key] for pt in per_ticker if key in pt["y_val_abs"]]
                )

        return {
            "X_train": X_train,
            "X_val": X_val,
            "anchor_val": anchor_val,
            "y_train": y_train,
            "y_val": y_val,
            "y_val_abs": y_val_abs,
            "feature_names": feature_names,
        }

    def _print_training_header(
        self,
        per_ticker: List[Dict[str, Any]],
        ticker_labels: Optional[List[str]] = None,
    ) -> None:
        """Emit the training banner shown before per-model training lines."""
        n_train_total = sum(len(pt["X_train"]) for pt in per_ticker)
        n_val_total = sum(
            len(pt["X_val"]) for pt in per_ticker if pt["X_val"] is not None
        )
        n_models = self.pred_len * len(self.target_features)

        print(f"\n{'=' * 60}")
        print("LightGBM Forecaster Training")
        print(f"{'=' * 60}")
        print(f"Features: {len(per_ticker[0]['feature_names'])}")
        if len(per_ticker) > 1:
            print(f"Pooled tickers: {len(per_ticker)}"
                  + (f" ({', '.join(ticker_labels)})" if ticker_labels else ""))
            for i, pt in enumerate(per_ticker):
                tag = ticker_labels[i] if ticker_labels else f"#{i}"
                val_len = len(pt["X_val"]) if pt["X_val"] is not None else 0
                print(f"  {tag:8s} train={len(pt['X_train'])} val={val_len}")
        print(f"Train samples: {n_train_total}")
        print(f"Val samples:   {n_val_total}")
        print(f"Targets: {self.target_features}")
        print(f"Pred steps: {self.pred_len}  |  Models to train: {n_models}")
        print("Target type: percentage return (scale-invariant)")
        print(f"{'=' * 60}\n")

    def _fit_from_arrays(
        self, arrays: Dict[str, Any], early_stopping_rounds: int,
    ) -> None:
        """Train one LGBMRegressor per (step, target) from pre-built arrays."""
        self.feature_names = arrays["feature_names"]
        X_train = pd.DataFrame(arrays["X_train"], columns=self.feature_names)
        X_val = (
            pd.DataFrame(arrays["X_val"], columns=self.feature_names)
            if arrays["X_val"] is not None else None
        )
        anchor_val = arrays["anchor_val"]

        self.models = {}
        for step in range(1, self.pred_len + 1):
            for target_name in self.target_features:
                key = (step, target_name)
                model = lgb.LGBMRegressor(**self._params)

                fit_kwargs: Dict[str, Any] = {}
                if X_val is not None and key in arrays["y_val"]:
                    fit_kwargs["eval_set"] = [(X_val, arrays["y_val"][key])]
                    fit_kwargs["callbacks"] = [
                        lgb.early_stopping(early_stopping_rounds, verbose=False),
                        lgb.log_evaluation(period=0),
                    ]

                model.fit(X_train, arrays["y_train"][key], **fit_kwargs)
                self.models[key] = model

                best_iter = getattr(model, "best_iteration_", model.n_estimators)
                val_score = ""
                if X_val is not None and key in arrays["y_val_abs"]:
                    pct_pred = model.predict(X_val)
                    abs_pred = anchor_val * (1.0 + pct_pred)
                    mae = float(np.mean(np.abs(abs_pred - arrays["y_val_abs"][key])))
                    val_score = f"  val_MAE=${mae:.2f}"

                print(f"  step={step} target={target_name:8s}  "
                      f"best_iter={best_iter}{val_score}")

        print(f"\nTraining complete. {len(self.models)} models trained.\n")

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict future values from raw OHLCV data.

        Internally the model predicts percentage returns from the anchor
        Close price, then converts back to absolute prices.

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
                pct_return = model.predict(X_pred)
                preds[:, step - 1, t_idx] = anchor * (1.0 + pct_return)

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
            "target_mode": self.target_mode,
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
            target_mode=payload.get("target_mode", "price"),
        )
        instance.target_features = payload["target_features"]
        instance.feature_names = payload["feature_names"]
        instance.models = payload["models"]
        print(f"LightGBM forecaster loaded from {path}")
        return instance
