from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..base import ForecastModel
from .blocks import TimesBlock
from .layers import DataEmbedding


@dataclass(frozen=True)
class TimesNetForecastConfig:
    """
    Hyperparameter container for the forecasting-only TimesNet model.

    This configuration is intentionally scoped to *long-term forecasting*.
    It does not include task switches (e.g., classification/anomaly/imputation)
    and does not carry training-specific parameters (e.g., learning rate).

    The model consumes a fixed-length historical window of length `seq_len`
    and produces a forecast horizon of length `pred_len`.

    Notes
    -----
    - `seq_len` in this config must match the dataset's sequence length.
      A mismatch is treated as a configuration error and raises an exception.
    - `c_out` can be equal to `enc_in` (typical multivariate forecasting), but
      may also differ if you want to project to a different output dimension.

    Attributes
    ----------
    seq_len : int
        Historical lookback length (number of past time steps).
    pred_len : int
        Forecast horizon length (number of future time steps to predict).
    enc_in : int
        Number of input features/channels in the input time series.
    c_out : int
        Number of output features/channels in the forecast.
    d_model : int
        Model embedding dimension.
    d_ff : int
        Hidden channel dimension inside the TimesNet 2D-convolution block.
    e_layers : int
        Number of stacked TimesNet blocks.
    top_k : int
        Number of dominant periods selected via FFT in each block.
    num_kernels : int
        Number of parallel convolution branches in the Inception-style block.
    embed : str
        Temporal embedding mode passed to `DataEmbedding` (e.g., "fixed", "timeF").
    freq : str
        Frequency label passed to `DataEmbedding` for time feature handling.
    dropout : float
        Dropout probability used in embeddings (and potentially other modules).
    eps : float
        Small constant used for numerical stability in normalization.
    """

    # Task lengths
    seq_len: int = 96
    pred_len: int = 96

    # Input / output channels
    enc_in: int = 7
    c_out: int = 7

    # Model dimensions
    d_model: int = 64
    d_ff: int = 256
    e_layers: int = 2

    # TimesNet-specific
    top_k: int = 2
    num_kernels: int = 6

    # Embedding
    embed: str = "fixed"   # "fixed" or "timeF" etc.
    freq: str = "h"

    # Regularization
    dropout: float = 0.1

    # Numerical stability
    eps: float = 1e-5


class TimesNetForecastModel(ForecastModel):
    """
    Forecasting-only TimesNet implementation.

    This model implements the core TimesNet idea for time-series forecasting:
    (1) Normalize the input per-sample across time to stabilize training.
    (2) Embed the input using value + positional (+ optional temporal) embeddings.
    (3) Expand the time axis from `seq_len` to `seq_len + pred_len` using a linear
        projection (time alignment).
    (4) Apply a stack of TimesNet blocks. Each block:
        - identifies dominant periods using FFT,
        - folds the sequence into a 2D representation by each period,
        - applies Inception-style Conv2D processing,
        - unfolds back to 1D and aggregates reconstructions.
    (5) Project latent features back to `c_out` channels.
    (6) Denormalize outputs back to the original data scale.
    (7) Return only the last `pred_len` time steps as the forecast horizon.

    Shape contract
    --------------
    Inputs
    - x: Tensor of shape [B, seq_len, enc_in]
        Historical window of the time-series.
    - x_mark: Optional Tensor of shape [B, seq_len, K]
        Optional time features aligned to `x`. Can be None.

    Outputs
    - y_pred: Tensor of shape [B, pred_len, c_out]
        Forecast horizon in the *original* scale (after denormalization).

    Raises
    ------
    ValueError
        If the input tensor does not have rank 3 or if its time length does not
        match `cfg.seq_len`.
    """

    def __init__(self, cfg: TimesNetForecastConfig):
        """
        Initialize a forecasting-only TimesNet model.

        Args:
            cfg (TimesNetForecastConfig):
                Model hyperparameters and architectural settings.

        Notes
        -----
        - `align_time` is a linear layer that maps the time dimension from `seq_len`
        to `seq_len + pred_len` after transposing the tensor to [B, D, T].
        - The model uses `DataEmbedding` to add value + positional (+ optional time)
        embeddings before TimesNet blocks.
        """

        super().__init__()
        self.cfg = cfg
        self.total_len = cfg.seq_len + cfg.pred_len

        # Embedding: [b,t,c] -> [b,t,d_model]
        self.enc_embedding = DataEmbedding(
            c_in=cfg.enc_in,
            d_model=cfg.d_model,
            embed=cfg.embed,
            freq=cfg.freq,
            dropout=cfg.dropout,
        )

        # Align time dimension from seq_len -> (seq_len + pred_len)
        # Operates on the TIME dimension (after transpose).
        self.align_time = nn.Linear(cfg.seq_len, self.total_len)

        # TimesNet blocks
        self.blocks = nn.ModuleList(
            [
                TimesBlock(
                    d_model=cfg.d_model,
                    d_ff=cfg.d_ff,
                    k=cfg.top_k,
                    num_kernels=cfg.num_kernels,
                )
                for _ in range(cfg.e_layers)
            ]
        )

        self.norm = nn.LayerNorm(cfg.d_model)

        # Project back to output channels
        self.proj = nn.Linear(cfg.d_model, cfg.c_out, bias=True)

    # -------------------------
    # Normalization helpers
    # -------------------------

    def _normalize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Normalize a batch of sequences per-sample across the time axis.

        The normalization is computed independently for each sample and channel:
            mean[b, 1, c] = average over time
            std[b, 1, c]  = sqrt(var over time + eps)

        This is commonly used in forecasting models to improve numerical stability
        and make optimization less sensitive to scale.

        Args:
            x (torch.Tensor):
                Input sequence tensor of shape [B, T, C].

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - x_norm: Normalized tensor of shape [B, T, C].
                - mean:   Per-sample mean of shape [B, 1, C].
                - std:    Per-sample standard deviation of shape [B, 1, C].

        Notes
        -----
        The mean is detached to prevent gradients flowing through statistics.
        """

        mean = x.mean(dim=1, keepdim=True).detach()
        x0 = x - mean
        std = torch.sqrt(x0.var(dim=1, keepdim=True, unbiased=False) + self.cfg.eps)
        x_norm = x0 / std
        return x_norm, mean, std

    def _denormalize(self, y: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """
        Restore normalized outputs back to the original data scale.

        This performs the inverse transform of `_normalize`:
            y_orig = y * std + mean

        Args:
            y (torch.Tensor):
                Normalized predictions of shape [B, T, C].
            mean (torch.Tensor):
                Per-sample mean used for normalization, shape [B, 1, C].
            std (torch.Tensor):
                Per-sample std used for normalization, shape [B, 1, C].

        Returns:
            torch.Tensor:
                Denormalized predictions of shape [B, T, C].
        """

        return y * std + mean

    # -------------------------
    # Forecasting
    # -------------------------

    def forward(self, x: torch.Tensor, x_mark: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute the forecast horizon for an input historical window.

        Pipeline
        --------
        1) Validate input shape and time length.
        2) Normalize `x` per-sample across time.
        3) Apply `DataEmbedding` to obtain latent features of size `d_model`.
        4) Expand the time axis from `seq_len` to `seq_len + pred_len` using a
        linear projection over the time dimension (`align_time`).
        5) Process latent sequence with stacked TimesNet blocks.
        6) Project latent features to `c_out` channels.
        7) Denormalize predictions back to the original scale.
        8) Return only the last `pred_len` steps.

        Args:
            x (torch.Tensor):
                Historical input sequence of shape [B, seq_len, enc_in].
            x_mark (Optional[torch.Tensor]):
                Optional time features aligned to `x`, shape [B, seq_len, K].
                If None, the model uses only value + positional embeddings.

        Returns:
            torch.Tensor:
                Forecast horizon of shape [B, pred_len, c_out] in the *original*
                data scale (after denormalization).

        Raises:
            ValueError:
                - If `x` is not a rank-3 tensor.
                - If `x.shape[1] != self.cfg.seq_len` (sequence length mismatch).
        """

        if x.dim() != 3:
            raise ValueError(f"Expected x shape [b,t,c], got {tuple(x.shape)}")
        if x.size(1) != self.cfg.seq_len:
            raise ValueError(
                f"Expected seq_len={self.cfg.seq_len}, got t={x.size(1)}. "
                "Ensure dataset uses the same seq_len as config."
            )

        # 1) normalize
        x_norm, mean, std = self._normalize(x)

        # 2) embed
        enc = self.enc_embedding(x_norm, x_mark)  # [b, seq_len, d_model]

        # 3) time alignment: [b, seq_len, d] -> [b, total_len, d]
        enc = self.align_time(enc.transpose(1, 2)).transpose(1, 2)

        # 4) TimesBlocks
        for blk in self.blocks:
            enc = self.norm(blk(enc, self.total_len))

        # 5) project: [b, total_len, c_out]
        y = self.proj(enc)

        # 6) denormalize back to original scale
        y = self._denormalize(y, mean, std)

        # 7) return only forecast horizon
        return y[:, -self.cfg.pred_len :, :]
