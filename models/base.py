from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn.functional as f


class ForecastModel(ABC, torch.nn.Module):
    """Abstract base class for forecasting models.

    Subclasses must implement `forward`. The recommended public
    interface uses shapes:
      - inputs:  [b, in_len, c]
      - outputs: [b, out_len, c] (for forecasting-style tasks)

    Classification or other tasks may return different shapes and
    should document them in the subclass.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Run a forward pass.

        Returns:
            torch.Tensor: Model output. For forecasting tasks, shape
            is typically [b, out_len, c].
        """

    def loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Default regression loss (MSE). Override in subclasses if needed."""
        return f.mse_loss(preds, targets)
