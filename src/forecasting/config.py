"""Training / test configuration dataclasses for PyTorch forecasters."""

from dataclasses import dataclass, field
from typing import Literal

import torch

from paths import CHECKPOINTS_ROOT, DATA_ROOT


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class TrainingConfig:
    """Hyperparameters and runtime settings for one training run."""
    model_name: Literal["TimeMixer", "TimesNet", "LightGBM", "StockMixer"] = "TimesNet"

    ticker: str = "AAPL"
    data_root: str = DATA_ROOT
    seq_len: int = 14
    pred_len: int = 5

    batch_size: int = 32
    epochs: int = 200
    learning_rate: float = 2e-4
    weight_decay: float = 1e-5
    patience: int = 30
    grad_clip: float = 1.0
    scheduler: Literal["cosine", "step", "none"] = "cosine"
    scheduler_step_size: int = 10
    scheduler_gamma: float = 0.5

    device: str = field(default_factory=_default_device)
    checkpoint_dir: str = CHECKPOINTS_ROOT


@dataclass
class TestConfig:
    """Runtime settings for evaluating a trained forecaster."""
    model_name: Literal["TimeMixer", "TimesNet", "LightGBM", "StockMixer"] = "TimesNet"
    ticker: str = "AAPL"
    data_root: str = DATA_ROOT
    seq_len: int = 14
    pred_len: int = 5
    batch_size: int = 32
    checkpoint_dir: str = CHECKPOINTS_ROOT
    device: str = field(default_factory=_default_device)
