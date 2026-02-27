"""
TimesNet block definitions.

This module implements the core building block used in TimesNet-style models.
The key idea is to discover dominant periodicities in a time series using FFT,
then transform the 1D sequence into a 2D representation aligned with each
period. A 2D convolutional block is applied to capture intra-period and
inter-period patterns, and the result is unfolded back to 1D.

Main components
---------------
TimesBlock
    A single TimesNet block that:
      (1) selects top-k dominant periods via FFT,
      (2) folds the sequence into 2D grids for each period,
      (3) applies Inception-style Conv2D processing,
      (4) unfolds back to 1D and aggregates reconstructions,
      (5) adds a residual connection.

All tensors follow the convention [B, T, C] where:
    B = batch size, T = time length, C = number of channels/features.
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as f
from beartype import beartype

from .types import BCRP, BTC
from .utils import fft_top_periods

# ---- Layout indices ----
# Original 3D: BTC = [B, T, C]
BTC_B: int = 0
BTC_T: int = 1
BTC_C: int = 2

# After reshape (before permute): BRPC = [B, Rows, Period, C]
BRPC_B: int = 0
BRPC_R: int = 1
BRPC_P: int = 2
BRPC_C: int = 3

# Folded target 4D: BCRP = [B, C, Rows, Period]
BCRP_B: int = 0
BCRP_C: int = 1
BCRP_R: int = 2
BCRP_P: int = 3


class TimesBlock(nn.Module):
    """
    A single TimesNet block for time-series feature transformation.

    This block implements the main TimesNet mechanism:
        (1) Estimate dominant periods using FFT magnitude statistics.
        (2) For each selected period p:
            - Fold the input sequence from 1D into a 2D grid of shape
              [rows, p] (padding with zeros if needed).
            - Apply an Inception-style Conv2D stack over the 2D grid.
            - Unfold the result back into a 1D sequence of length T.
        (3) Aggregate reconstructions across periods using a softmax weighting
            derived from FFT amplitudes.
        (4) Add a residual connection to preserve the original signal.

    Notes
    -----
    - Folding converts [B, T, C] into [B, C, rows, p] to enable 2D convolution.
    - The parameter `total_len` typically equals (seq_len + pred_len) and is used
      only to determine the folding grid size (rows) and required padding.

    Input / Output
    --------------
    Input:
        x : [B, T, C]
    Returns:
        out : [B, T, C]
    """

    @beartype
    def __init__(self, d_model: int, d_ff: int, k: int, num_kernels: int):
        """
        Initialize a TimesNet block.

        Args:
            d_model (int):
                Feature dimension of the input sequence (channels after embedding).
            d_ff (int):
                Hidden channel dimension inside the Conv2D stack.
            k (int):
                Number of dominant periods to select from FFT (top-k).
            num_kernels (int):
                Number of parallel convolution branches inside the Inception block.

        Raises:
            ValueError:
                If k < 1 or if any of (d_model, d_ff, num_kernels) is < 1.
        """

        super().__init__()
        from .layers import InceptionBlockV1  # avoid cycles

        if k < 1:
            raise ValueError("k (top_k) must be >= 1")
        if min(d_model, d_ff, num_kernels) < 1:
            raise ValueError("d_model, d_ff, num_kernels must be >= 1")

        self.k = k
        self.conv = nn.Sequential(
            InceptionBlockV1(d_model, d_ff, num_kernels=num_kernels),
            nn.GELU(),
            InceptionBlockV1(d_ff, d_model, num_kernels=num_kernels),
        )

    @beartype
    def fold_time(self, x: BTC, period: int, total_len: int) -> tuple[BCRP, int]:
        """
        Fold a 1D time sequence into a 2D grid aligned with a given period.

        The folding procedure reshapes the time axis into (rows, period) where:
            rows = ceil(target_len / period)
        and `target_len` is chosen as max(total_len, T). If T is not divisible by
        period, the sequence is padded with zeros along the time dimension.

        Args:
            x (BTC):
                Input tensor of shape [B, T, C].
            period (int):
                Candidate period length used for folding (p >= 1).
            total_len (int):
                Target length used to determine the folding grid (typically seq_len + pred_len).

        Returns:
            Tuple[BCRP, int]:
                - y: Folded tensor of shape [B, C, rows, period].
                - rows: Number of rows used in the folded representation.

        Raises:
            ValueError:
                If total_len < 1.
        """

        if total_len < 1:
            raise ValueError("total_len must be >= 1")
        b, t, c = x.shape
        p = max(int(period), 1)

        # Compute target rows regardless of divisibility
        target_len = max(total_len, t)
        rows = (target_len + p - 1) // p  # ceil(target_len / p)
        pad = rows * p - t  # pad from current length t

        if pad:
            x = cast(BTC, torch.cat([x, x.new_zeros(b, pad, c)], dim=BTC_T))

        # [B,T',C] -> [B,Rows,Period,C] -> [B,C,Rows,Period]
        y = (
            x.reshape(b, rows, p, c)
            .permute(BRPC_B, BRPC_C, BRPC_R, BRPC_P)  # (0,3,1,2)
            .contiguous()
        )
        return cast(BCRP, y), rows

    @beartype
    def unfold_time(self, y: BCRP, rows: int, period: int, total_len: int, t_orig: int) -> BTC:
        """
        Unfold a folded 2D representation back to a 1D time sequence.

        This is the inverse of `fold_time`. The tensor is permuted and reshaped
        back into [B, rows * period, C] and then cropped to the original time
        length `t_orig`.

        Args:
            y (BCRP):
                Folded tensor of shape [B, C, rows, period].
            rows (int):
                Number of rows used during folding.
            period (int):
                Period length used during folding.
            total_len (int):
                Target length used during folding (kept for API symmetry and validation).
            t_orig (int):
                Original time length to crop the unfolded output to.

        Returns:
            BTC:
                Unfolded tensor of shape [B, t_orig, C].

        Raises:
            ValueError:
                If any of (rows, period, total_len, t_orig) is < 1.
        """
        
        if rows < 1 or period < 1 or total_len < 1 or t_orig < 1:
            raise ValueError("rows, period, total_len, t_orig must be >= 1")
        b, c, _, p = y.shape

        out = (
            y.permute(BCRP_B, BCRP_R, BCRP_P, BCRP_C)  # [B,Rows,Period,C]
            .contiguous()
            .reshape(b, rows * p, c)  # [B,Rows*Period,C]
        )
        # Crop only to the original requested length
        out = out[:, :t_orig, :]
        return cast(BTC, out)

    @beartype
    def forward(self, x: BTC, total_len: int) -> BTC:
        """
        Run a forward pass of the TimesNet block.

        This method:
            (1) Computes dominant periods with FFT (`fft_top_periods`).
            (2) Reconstructs the sequence for each period via:
                    fold -> Conv2D stack -> unfold
            (3) Aggregates reconstructions using a softmax attention over FFT-based weights.
            (4) Adds the input residual (out + x).

        Args:
            x (BTC):
                Input tensor of shape [B, T, C].
            total_len (int):
                Target length used for folding (typically seq_len + pred_len).
                This affects padding/rows computation but the returned sequence length
                remains T.

        Returns:
            BTC:
                Output tensor of shape [B, T, C].

        Raises:
            ValueError:
                If total_len < 1.
        """
        
        if total_len < 1:
            raise ValueError("total_len must be >= 1")
        b, t, c = x.shape

        # Limit k to available non-DC rFFT bins.
        k_eff = min(self.k, max(t // 2, 1))

        periods, weights = fft_top_periods(x, k_eff)  # periods: [k_eff], weights: [b,k_eff]
        weights = weights.to(dtype=x.dtype)
        recon: list[torch.Tensor] = []  # keep internal tensors as torch.Tensor

        for i in range(k_eff):
            p = int(periods[i].item()) or 1
            y, rows = self.fold_time(x, p, total_len)  # [b,c,rows,p]
            y = self.conv(y)  # [b,c,rows,p]
            y = self.unfold_time(y, rows, p, total_len, t)  # [b,t,c] (already BTC via unfold_time)
            recon.append(y)

        if k_eff == 1:
            out = recon[0]
        else:
            stacked = torch.stack(recon, dim=-1)  # [b, t, c, k_eff]
            attn = f.softmax(weights, dim=1).unsqueeze(1).unsqueeze(1)  # [b,1,1,k_eff]
            out = torch.sum(stacked * attn, dim=-1)  # [b, t, c]

        return cast(BTC, out + x)  # <-- cast on return
