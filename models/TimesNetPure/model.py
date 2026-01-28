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
    Configuration for TimesNet forecasting model.

    This project uses TimesNet ONLY for forecasting (no classification,
    anomaly detection, or imputation). Keep this config minimal and
    focused on the forecasting task.
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
    e_layers: int = 3

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
    TimesNet model for forecasting only.

    Inputs:
        x:      Tensor [batch, seq_len, enc_in]
        x_mark: Optional time features [batch, seq_len, k] (can be None)

    Outputs:
        y_pred: Tensor [batch, pred_len, c_out]
    """

    def __init__(self, cfg: TimesNetForecastConfig):
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
        Normalize per-sample across time (NS-Transformer style).

        Args:
            x: [b, t, c]

        Returns:
            x_norm: [b, t, c]
            mean:   [b, 1, c]
            std:    [b, 1, c]
        """
        mean = x.mean(dim=1, keepdim=True).detach()
        x0 = x - mean
        std = torch.sqrt(x0.var(dim=1, keepdim=True, unbiased=False) + self.cfg.eps)
        x_norm = x0 / std
        return x_norm, mean, std

    def _denormalize(self, y: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """
        Undo _normalize.

        Args:
            y:    [b, t, c]
            mean: [b, 1, c]
            std:  [b, 1, c]
        """
        return y * std + mean

    # -------------------------
    # Forecasting
    # -------------------------

    def forward(self, x: torch.Tensor, x_mark: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Run TimesNet forecasting.

        Args:
            x:      [b, seq_len, enc_in]
            x_mark: Optional [b, seq_len, k] time features (can be None)

        Returns:
            [b, pred_len, c_out] predictions in the ORIGINAL scale
            (after denormalization).
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
