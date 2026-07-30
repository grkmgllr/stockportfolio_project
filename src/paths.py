"""Single source of truth for filesystem locations used by the pipeline."""

import os

RESULTS_ROOT = "results"
CHECKPOINTS_ROOT = "checkpoints"
DATA_ROOT = "data/raw"


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
