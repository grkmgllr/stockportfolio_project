from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn.functional as f


class ForecastModel(ABC, torch.nn.Module):
    """
    Abstract base class for forecasting models.

    This class defines the *minimal interface contract* for all forecasting
    models used in this project. It combines PyTorch's `nn.Module` with Python's
    abstract base class (ABC) mechanism to enforce a consistent API across
    different model implementations.

    Design goals
    ------------
    - Provide a unified interface for time-series forecasting models.
    - Enforce implementation of a `forward` method in subclasses.
    - Offer a default regression loss suitable for forecasting tasks.
    - Remain agnostic to specific architectures (e.g., TimesNet, Transformer).

    Shape contract
    --------------
    Unless explicitly stated otherwise by a subclass, forecasting models
    following this base class are expected to obey the following convention:

    Inputs:
        x : torch.Tensor
            Shape [B, T_in, C]
            where:
                B = batch size,
                T_in = input (historical) sequence length,
                C = number of input channels/features.

    Outputs:
        y_pred : torch.Tensor
            Shape [B, T_out, C]
            where T_out is the forecast horizon length.

    Notes
    -----
    - This base class does *not* assume any particular loss beyond regression.
    - Subclasses may override `loss` if a different objective is required.
    - Tasks such as classification or anomaly detection should either:
        (a) clearly document deviations from this contract, or
        (b) use a different base class.
    """

    def __init__(self) -> None:
        """
        Initialize the forecasting model base.

        This constructor simply initializes the underlying `nn.Module`.
        Subclasses are responsible for defining all learnable parameters.
        """
        super().__init__()

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        Forward pass of the forecasting model.

        This method must be implemented by all subclasses and defines how
        input tensors are transformed into model outputs.

        Args:
            *args:
                Positional arguments required by the model. For forecasting
                models, this typically includes the input time-series tensor.
            **kwargs:
                Optional keyword arguments such as time features, masks, or
                auxiliary inputs.

        Returns:
            torch.Tensor:
                Model output tensor. For forecasting tasks, this is typically
                a tensor of shape [B, T_out, C].

        Raises:
            NotImplementedError:
                If the subclass does not implement this method.
        """
        raise NotImplementedError

    def loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the default loss for forecasting tasks.

        The default implementation uses Mean Squared Error (MSE), which is
        commonly employed in regression-based time-series forecasting.

        Args:
            preds (torch.Tensor):
                Model predictions with shape [B, T_out, C].
            targets (torch.Tensor):
                Ground-truth target values with shape [B, T_out, C].

        Returns:
            torch.Tensor:
                Scalar loss value.

        Notes
        -----
        - Subclasses may override this method to implement alternative loss
          functions (e.g., MAE, Huber loss, probabilistic losses).
        """
        return f.mse_loss(preds, targets)
