"""Single source of truth for filesystem locations used by the pipeline."""

import os

RESULTS_ROOT = "results"
CHECKPOINTS_ROOT = "checkpoints"
DATA_ROOT = "data/raw"
META_ROOT = "data/meta"
META_CHECKPOINTS_ROOT = os.path.join(CHECKPOINTS_ROOT, "meta")


def results_dir(ticker: str, model: str, create: bool = True) -> str:
    """Path to ``results/{ticker}/{model}/`` — canonical prediction output layout."""
    path = os.path.join(RESULTS_ROOT, ticker, model)
    if create:
        os.makedirs(path, exist_ok=True)
    return path


def forecaster_checkpoint(ticker: str, model: str) -> str:
    """Path to a primary-forecaster checkpoint file."""
    if model == "LightGBM":
        return os.path.join(CHECKPOINTS_ROOT, f"{ticker}_LightGBM_best.joblib")
    return os.path.join(CHECKPOINTS_ROOT, f"{ticker}_{model}_best.pt")


def meta_labels_path(ticker: str, model: str) -> str:
    """Path to the meta-label CSV produced by the labelling step."""
    return os.path.join(META_ROOT, f"meta_labels_{ticker}_{model}.csv")


def meta_predictions_path(ticker: str, model: str) -> str:
    """Path to the meta-classifier's per-row prediction CSV."""
    return os.path.join(META_ROOT, f"meta_predictions_{ticker}_{model}.csv")


def meta_classifier_checkpoint(ticker: str, model: str) -> str:
    """Path to a trained meta-classifier joblib."""
    return os.path.join(META_CHECKPOINTS_ROOT, f"{ticker}_{model}_meta_clf.joblib")
