from __future__ import annotations

from typing import Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

PaddingMode = Literal["replicate", "reflect"]

class MovingAverage(nn.Module):
    """
    Moving average smoother over the time dimension.

    Input:  time_series [B, T, D]
    Output: trend       [B, T, D]
    """

    def __init__(
        self,
        kernel_size: int,
        padding_mode: PaddingMode = "replicate",
        center: bool = True,
    ) -> None:
        super().__init__()

        if kernel_size < 1:
            raise ValueError(f"kernel_size must be >= 1, got {kernel_size}.")
        if center and (kernel_size % 2 == 0):
            raise ValueError(f"kernel_size must be odd when center=True, got {kernel_size}.")
        if padding_mode not in ("replicate", "reflect"):
            raise ValueError(f"padding_mode must be 'replicate' or 'reflect', got {padding_mode}.")

        self.kernel_size = kernel_size
        self.padding_mode = padding_mode
        self.center = center

    def forward(self, time_series: torch.Tensor) -> torch.Tensor:
        # Pooling expects [B, Channels, Time] but our contract is [B, Time, Features]
        x = time_series.transpose(1, 2)  # [B, D, T]

        if self.center:
            pad_left = self.kernel_size // 2
            pad_right = self.kernel_size // 2
        else:
            # Causal smoothing (only past influences current)
            pad_left = self.kernel_size - 1
            pad_right = 0

        x_padded = F.pad(x, (pad_left, pad_right), mode=self.padding_mode)

        trend = F.avg_pool1d(
            x_padded,
            kernel_size=self.kernel_size,
            stride=1,
        )

        return trend.transpose(1, 2)  # [B, T, D]


class SeriesDecomposition(nn.Module):
    """
    Decomposition block for TimeMixer:
    x = seasonal + trend
    trend = MovingAverage(x)
    seasonal = x - trend
    """

    def __init__(
        self,
        kernel_size: int,
        padding_mode: PaddingMode = "replicate",
        center: bool = True,
        validate_shapes: bool = True,
    ) -> None:
        super().__init__()
        self.validate_shapes = validate_shapes
        self.moving_average = MovingAverage(
            kernel_size=kernel_size,
            padding_mode=padding_mode,
            center=center,
        )

    def forward(self, time_series: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.validate_shapes:
            if not isinstance(time_series, torch.Tensor):
                raise TypeError(f"time_series must be a torch.Tensor, got {type(time_series)}.")
            if time_series.ndim != 3:
                raise ValueError(
                    f"time_series must have shape [B, T, D] (3D), got {tuple(time_series.shape)}."
                )

        trend_component = self.moving_average(time_series)
        seasonal_component = time_series - trend_component

        return seasonal_component, trend_component
