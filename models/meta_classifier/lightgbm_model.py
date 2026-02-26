"""
LightGBM Meta-Classifier for filtering primary model signals.

This module wraps a LightGBM binary classifier behind a clean interface
that integrates with the rest of the meta-labeling pipeline.  The model
is trained to predict the *probability* that a signal from the primary
model (TimeMixer) will result in a profitable trade (as defined by the
Triple Barrier Method).

Design goals
------------
- Provide ``fit``, ``predict_proba``, ``save``, and ``load`` methods
  that are consistent with the scikit-learn estimator API.
- Force the use of ``PurgedKFold`` during training to prevent data
  leakage from overlapping event windows.
- Output calibrated probabilities via ``predict_proba()`` rather than
  hard classifications, so that downstream code can use the confidence
  score for position sizing, threshold tuning, or risk management.
- Log per-fold metrics during cross-validated training so the user can
  diagnose overfitting or class imbalance issues.

Role in the pipeline
--------------------
    generate_meta_labels.py  -->  meta_labels_{ticker}.csv
                                        |
                                        v
                                  train_meta.py
                                        |
                                  MetaClassifier.fit()
                                  (uses PurgedKFold internally)
                                        |
                                        v
                                  MetaClassifier.predict_proba()
                                  --> confidence scores [0, 1]
"""

import numpy as np
import lightgbm as lgb
import joblib
import os
from typing import Optional, Dict, List, Any

from trading_logic.purged_cv import PurgedKFold


class MetaClassifier:
    """
    LightGBM-based meta-classifier for trade signal filtering.

    Wraps a ``lightgbm.LGBMClassifier`` with built-in support for
    ``PurgedKFold`` cross-validation.  The classifier learns to predict
    the probability that a signal from the primary forecasting model
    will be profitable.

    Attributes
    ----------
    model : lgb.LGBMClassifier
        The underlying LightGBM classifier instance.
    cv_results : List[Dict[str, float]]
        Per-fold cross-validation metrics collected during ``fit()``.
    feature_names : List[str]
        Names of the features used during training (stored for
        reproducibility and feature importance analysis).

    Notes
    -----
    - The model uses ``objective='binary'`` and ``metric='binary_logloss'``
      by default, which are appropriate for the binary meta-labeling task.
    - ``is_unbalance=True`` is set by default to handle the typical class
      imbalance in financial labels (more losing trades than winning ones).
    - All hyperparameters can be overridden via the ``lgb_params``
      dictionary passed to ``__init__``.
    """

    DEFAULT_PARAMS: Dict[str, Any] = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "is_unbalance": True,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": -1,
    }

    def __init__(self, lgb_params: Optional[Dict[str, Any]] = None):
        """
        Initialize the meta-classifier.

        Args:
            lgb_params (Dict[str, Any], optional):
                LightGBM hyperparameters.  Any key provided here
                overrides the corresponding entry in ``DEFAULT_PARAMS``.
                If None, the defaults are used unchanged.
        """
        params = self.DEFAULT_PARAMS.copy()
        if lgb_params is not None:
            params.update(lgb_params)

        self.model = lgb.LGBMClassifier(**params)
        self.cv_results: List[Dict[str, float]] = []
        self.feature_names: List[str] = []

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        t_start: np.ndarray,
        t_end: np.ndarray,
        n_splits: int = 5,
        n_embargo: int = 5,
        feature_names: Optional[List[str]] = None,
        early_stopping_rounds: int = 50,
    ) -> "MetaClassifier":
        """
        Train the meta-classifier with Purged K-Fold cross-validation.

        The model is trained ``n_splits`` times.  For each fold the
        training set is purged of overlapping events and embargoed.
        Per-fold metrics (accuracy, log-loss) are recorded in
        ``self.cv_results``.  After cross-validation, the final model
        is retrained on the *full* dataset.

        Training pipeline
        -----------------
        1. Create a ``PurgedKFold`` splitter from the event spans.
        2. For each fold:
           a. Extract purged train and test indices.
           b. Train a LightGBM model with early stopping on the
              test fold.
           c. Record accuracy and log-loss on the held-out fold.
        3. Retrain the final model on the full dataset (without
           early stopping) using the best ``n_estimators`` from CV.

        Args:
            X (np.ndarray):
                Feature matrix of shape ``[N, F]``.
            y (np.ndarray):
                Binary target array of shape ``[N]`` (0 or 1).
            t_start (np.ndarray):
                Event start indices of shape ``[N]`` (from triple barrier).
            t_end (np.ndarray):
                Event end indices of shape ``[N]`` (from triple barrier).
            n_splits (int):
                Number of cross-validation folds (default 5).
            n_embargo (int):
                Number of bars to embargo after each test fold (default 5).
            feature_names (List[str], optional):
                Feature column names for interpretability.
            early_stopping_rounds (int):
                Stop training a fold if the test metric does not improve
                for this many rounds (default 50).

        Returns:
            MetaClassifier:
                ``self``, to allow method chaining.
        """
        self.feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]
        self.cv_results = []

        cv = PurgedKFold(
            n_splits=n_splits,
            n_embargo=n_embargo,
            t_start=t_start,
            t_end=t_end,
        )

        best_iterations = []

        print(f"\n{'='*60}")
        print(f"Meta-Classifier Training (PurgedKFold, {n_splits} folds)")
        print(f"{'='*60}")
        print(f"Samples: {len(X)}  |  Features: {X.shape[1]}")
        print(f"Class balance: {int(y.sum())} positive / {int(len(y) - y.sum())} negative")
        print(f"Embargo: {n_embargo} bars")
        print(f"{'='*60}\n")

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            fold_model = lgb.LGBMClassifier(**self.model.get_params())

            fold_model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                callbacks=[
                    lgb.early_stopping(early_stopping_rounds, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )

            best_iter = fold_model.best_iteration_
            best_iterations.append(best_iter)

            y_pred_proba = fold_model.predict_proba(X_test)[:, 1]
            y_pred = (y_pred_proba >= 0.5).astype(int)

            accuracy = float(np.mean(y_pred == y_test))
            logloss = float(-np.mean(
                y_test * np.log(np.clip(y_pred_proba, 1e-15, 1)) +
                (1 - y_test) * np.log(np.clip(1 - y_pred_proba, 1e-15, 1))
            ))

            fold_result = {
                "fold": fold_idx,
                "train_size": len(train_idx),
                "test_size": len(test_idx),
                "accuracy": accuracy,
                "logloss": logloss,
                "best_iteration": best_iter,
            }
            self.cv_results.append(fold_result)

            print(f"  Fold {fold_idx}: acc={accuracy:.4f}  "
                  f"logloss={logloss:.4f}  "
                  f"best_iter={best_iter}  "
                  f"train={len(train_idx)} test={len(test_idx)}")

        # Summary
        mean_acc = np.mean([r["accuracy"] for r in self.cv_results])
        mean_ll = np.mean([r["logloss"] for r in self.cv_results])
        avg_best_iter = int(np.mean(best_iterations))

        print(f"\n  CV Mean:  acc={mean_acc:.4f}  logloss={mean_ll:.4f}")
        print(f"  Avg best iteration: {avg_best_iter}")

        # Retrain on full dataset with averaged best iteration count
        print(f"\n  Retraining final model on all {len(X)} samples "
              f"(n_estimators={avg_best_iter})...")

        final_params = self.model.get_params()
        final_params["n_estimators"] = max(avg_best_iter, 10)
        self.model = lgb.LGBMClassifier(**final_params)
        self.model.fit(X, y)

        print("  Done.\n")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the probability that each signal is profitable.

        Returns the positive-class probability (class 1 = take-profit),
        which can be used directly as a confidence score for position
        sizing or threshold-based filtering.

        Args:
            X (np.ndarray):
                Feature matrix of shape ``[N, F]``.

        Returns:
            np.ndarray:
                Array of shape ``[N]`` with values in [0, 1].
                Higher values indicate higher confidence that the
                primary model's signal will be profitable.
        """
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict hard classifications using a configurable threshold.

        Args:
            X (np.ndarray):
                Feature matrix of shape ``[N, F]``.
            threshold (float):
                Decision boundary (default 0.5).  Signals with
                ``predict_proba >= threshold`` are classified as 1.

        Returns:
            np.ndarray:
                Binary array of shape ``[N]`` (0 or 1).
        """
        return (self.predict_proba(X) >= threshold).astype(int)

    def feature_importance(self, importance_type: str = "gain") -> Dict[str, float]:
        """
        Return feature importances as a name-to-score mapping.

        Args:
            importance_type (str):
                Type of importance: ``'gain'`` (default) or ``'split'``.

        Returns:
            Dict[str, float]:
                Dictionary mapping feature names to importance scores,
                sorted in descending order.
        """
        importances = self.model.feature_importances_
        name_score = dict(zip(self.feature_names, importances))
        return dict(sorted(name_score.items(), key=lambda x: x[1], reverse=True))

    def save(self, path: str) -> None:
        """
        Save the trained model and metadata to disk.

        Args:
            path (str):
                Output file path (recommended extension: ``.joblib``).
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "model": self.model,
            "cv_results": self.cv_results,
            "feature_names": self.feature_names,
        }
        joblib.dump(payload, path)
        print(f"Meta-classifier saved to {path}")

    @classmethod
    def load(cls, path: str) -> "MetaClassifier":
        """
        Load a previously saved meta-classifier from disk.

        Args:
            path (str):
                Path to the ``.joblib`` file.

        Returns:
            MetaClassifier:
                Restored instance with model, CV results, and feature names.
        """
        payload = joblib.load(path)
        instance = cls()
        instance.model = payload["model"]
        instance.cv_results = payload["cv_results"]
        instance.feature_names = payload["feature_names"]
        print(f"Meta-classifier loaded from {path}")
        return instance
