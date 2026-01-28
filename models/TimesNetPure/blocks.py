from __future__ import annotations

from typing import List, Tuple, cast

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

# Folded 4D: BCRP = [B, C, Rows, Period]
BCRP_B: int = 0
BCRP_C: int = 1
BCRP_R: int = 2
BCRP_P: int = 3


class TimesBlock(nn.Module):
    """
    A single TimesNet block.

    This block implements the core TimesNet idea:
      1) Find dominant periods via FFT (top-k).
      2) For each period:
         - Fold the 1D time series into a 2D grid [rows x period]
         - Apply multi-scale Conv2D (Inception-style)
         - Unfold back to 1D
      3) Aggregate per-period reconstructions using softmax weights
      4) Add a residual connection.

    Shapes:
        Input : x      -> [b, t, c]
        Output: out    -> [b, t, c]
    """

    @beartype
    def __init__(self, d_model: int, d_ff: int, k: int, num_kernels: int) -> None:
        """
        Args:
            d_model: Embedding/feature dimension (channels for conv2d input).
            d_ff: Hidden channels used inside the conv block.
            k: Number of dominant periods to use (top-k).
            num_kernels: Number of Inception branches (kernel sizes).
        """
        super().__init__()
        from .layers import InceptionBlockV1  # local import to avoid cycles

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

    @staticmethod
    def _effective_k(t: int, k: int) -> int:
        """
        Compute an effective top-k value based on the signal length.

        In rFFT, number of non-DC bins is at most t//2.
        """
        return min(k, max(t // 2, 1))

    @beartype
    def fold_time(self, x: BTC, period: int, total_len: int) -> Tuple[BCRP, int]:
        """
        Fold a 1D sequence into a 2D grid by a given period.

        We reshape [b, t, c] into [b, c, rows, period] where:
          - period is the width
          - rows is ceil(target_len / period)
        Padding (zeros) is added at the end of time dimension if needed.

        Args:
            x: [b, t, c] input tensor.
            period: Period length (>= 1).
            total_len: Target total length (e.g., seq_len + pred_len) used to
                       determine the number of rows.

        Returns:
            y:    [b, c, rows, period] folded tensor for conv2d.
            rows: number of rows used in the folding.
        """
        if total_len < 1:
            raise ValueError("total_len must be >= 1")

        b, t, c = x.shape
        p = max(int(period), 1)

        target_len = max(total_len, t)
        rows = (target_len + p - 1) // p  # ceil(target_len / p)
        pad = rows * p - t

        if pad:
            x = cast(BTC, torch.cat([x, x.new_zeros(b, pad, c)], dim=BTC_T))

        y = (
            x.reshape(b, rows, p, c)  # [b, rows, period, c]
            .permute(BRPC_B, BRPC_C, BRPC_R, BRPC_P)  # -> [b, c, rows, period]
            .contiguous()
        )
        return cast(BCRP, y), rows

    @beartype
    def unfold_time(self, y: BCRP, rows: int, period: int, t_orig: int) -> BTC:
        """
        Unfold a 2D grid back to the original 1D time length.

        Args:
            y: [b, c, rows, period]
            rows: number of rows used during folding.
            period: period used during folding.
            t_orig: original time length to crop to.

        Returns:
            out: [b, t_orig, c]
        """
        if rows < 1 or period < 1 or t_orig < 1:
            raise ValueError("rows, period, and t_orig must be >= 1")

        b, c, _, p = y.shape
        out = (
            y.permute(BCRP_B, BCRP_R, BCRP_P, BCRP_C)  # [b, rows, period, c]
            .contiguous()
            .reshape(b, rows * p, c)  # [b, rows*period, c]
        )
        return cast(BTC, out[:, :t_orig, :])

    @beartype
    def _reconstruct_for_period(self, x: BTC, period: int, total_len: int) -> BTC:
        """
        Run fold -> conv -> unfold for a single period.

        Args:
            x: [b, t, c]
            period: folding period
            total_len: target length to determine folding rows

        Returns:
            recon: [b, t, c] reconstruction for this period
        """
        b, t, _ = x.shape
        y, rows = self.fold_time(x, period, total_len)  # [b,c,rows,p]
        y = self.conv(y)                                # [b,c,rows,p]
        return self.unfold_time(y, rows, period, t)      # [b,t,c]

    @beartype
    def forward(self, x: BTC, total_len: int) -> BTC:
        """
        Forward pass for TimesBlock.

        Args:
            x: [b, t, c] input sequence features.
            total_len: Target length (typically seq_len + pred_len). Used only
                       to decide folding rows and padding length.

        Returns:
            out: [b, t, c] output with residual connection.
        """
        if total_len < 1:
            raise ValueError("total_len must be >= 1")

        _, t, _ = x.shape
        k_eff = self._effective_k(t, self.k)

        periods, weights = fft_top_periods(x, k_eff)  # periods: [k_eff], weights: [b,k_eff]
        weights = weights.to(dtype=x.dtype)

        recon: List[torch.Tensor] = []
        for i in range(k_eff):
            p = int(periods[i].item()) or 1
            recon.append(self._reconstruct_for_period(x, p, total_len))

        if k_eff == 1:
            out = recon[0]
        else:
            stacked = torch.stack(recon, dim=-1)  # [b,t,c,k]
            attn = f.softmax(weights, dim=1).unsqueeze(1).unsqueeze(1)  # [b,1,1,k]
            out = torch.sum(stacked * attn, dim=-1)  # [b,t,c]

        return cast(BTC, out + x)
