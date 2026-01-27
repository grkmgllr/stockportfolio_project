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
    """Fold 1D into 2D by dominant periods, apply conv2d, unfold, and aggregate.

    I/O: x [b, t, c] -> out [b, t, c]
    """

    @beartype
    def __init__(self, d_model: int, d_ff: int, k: int, num_kernels: int):
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
        """Fold [b,t,c] → [b,c,rows,period], padding time to a multiple of period."""
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
        """Unfold [b,c,rows,period] → [b,t_orig,c] (cropped)."""
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
        """Args:
        x: [b, t, c]
        total_len: seq_len + pred_len (padding/reshape target)
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
