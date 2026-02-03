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
    Implements one TimesNet temporal block by period-based 2D folding and convolution.

    This module is the core building unit of TimesNet for long-term forecasting.
    The main idea is to convert a 1D time series into multiple 2D representations
    using dominant periods estimated by FFT, then apply 2D convolutions (Inception-style)
    to capture multi-scale temporal patterns efficiently.

    High-level procedure:
        (1) Period discovery (FFT):
            Estimate top-k dominant periods from the input sequence using rFFT amplitudes.

        (2) Period-conditioned reconstruction:
            For each discovered period p:
                - Fold the time axis into a 2D grid with width p and height rows
                - Apply an Inception-like Conv2D block over the folded representation
                - Unfold the result back to the original time axis length

        (3) Adaptive aggregation:
            Combine per-period reconstructions using softmax-normalized FFT weights.

        (4) Residual connection:
            Add the original input (identity) to preserve information and stabilize training.

    Notes:
        - This block operates on feature sequences that are already embedded into d_model
          channels (i.e., x is expected to be in a latent space, not raw input features).
        - Folding may require zero-padding at the tail of the time dimension to make
          the length divisible by a given period.

    Inputs:
        x (torch.Tensor): Embedded sequence of shape [b, t, c].

    Returns:
        torch.Tensor: Output tensor of shape [b, t, c] after period-based reconstruction
        and residual addition.
    """

    @beartype
    def __init__(self, d_model: int, d_ff: int, k: int, num_kernels: int) -> None:
        """
        Initialize a TimesNet block.

        The block uses an Inception-style 2D convolutional module to process folded
        representations of the time series. Each fold corresponds to a dominant period
        discovered by FFT, and the outputs are aggregated with attention-like weights.

        Args:
            d_model (int): Latent feature dimension (number of channels in the embedded sequence).
            d_ff (int): Hidden channel dimension used inside the Conv2D/Inception block.
            k (int): Number of dominant periods to use (top-k frequencies from FFT).
            num_kernels (int): Number of parallel Conv2D branches in the Inception block.

        Returns:
            None
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
        Compute the effective top-k value for FFT-based period selection.

        In rFFT for a sequence of length t, the number of non-DC usable frequency
        bins is at most floor(t/2). This helper ensures k does not exceed that limit.

        Args:
            t (int): Sequence length.
            k (int): Requested number of top frequency components.

        Returns:
            int: Effective k such that 1 <= k_eff <= max(t//2, 1).
        """
        return min(k, max(t // 2, 1))

    @beartype
    def fold_time(self, x: BTC, period: int, total_len: int) -> Tuple[BCRP, int]:
        """
        Fold a 1D time sequence into a 2D grid given a period.

        TimesNet processes period-aligned patterns by reshaping the time axis into a
        2D grid. For a chosen period p, we reshape time into [rows, p] such that:
            rows = ceil(target_len / p)
        If the current sequence length is not divisible by p, the time axis is
        padded with zeros at the end before reshaping.

        Args:
            x (torch.Tensor): Input tensor of shape [b, t, c].
            period (int): Folding period p (p >= 1).
            total_len (int): Target length (typically seq_len + pred_len) that defines
                the number of rows used for folding.

        Returns:
            Tuple[torch.Tensor, int]:
                - y (torch.Tensor): Folded tensor of shape [b, c, rows, p].
                - rows (int): Number of rows used in the folded representation.
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
        Unfold a 2D folded representation back to the 1D time axis.

        This reverses fold_time(): the tensor [b, c, rows, p] is permuted and reshaped
        back into [b, rows*p, c], then cropped to the original time length.

        Args:
            y (torch.Tensor): Folded tensor of shape [b, c, rows, p].
            rows (int): Number of rows used during folding.
            period (int): Period p used during folding.
            t_orig (int): Original time length to crop the unfolded output to.

        Returns:
            torch.Tensor: Unfolded tensor of shape [b, t_orig, c].
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
        Compute the period-conditioned reconstruction for a single period.

        This helper performs the core transformation used by TimesNet:
            fold_time -> Conv2D processing -> unfold_time

        Args:
            x (torch.Tensor): Input tensor of shape [b, t, c].
            period (int): Period p used for folding.
            total_len (int): Target length (typically seq_len + pred_len) used to decide
                folding/padding behavior.

        Returns:
            torch.Tensor: Reconstruction tensor of shape [b, t, c] for the given period.
        """

        b, t, _ = x.shape
        y, rows = self.fold_time(x, period, total_len)  # [b,c,rows,p]
        y = self.conv(y)                                # [b,c,rows,p]
        return self.unfold_time(y, rows, period, t)      # [b,t,c]

    @beartype
    def forward(self, x: BTC, total_len: int) -> BTC:
        """
        Forward pass of the TimesNet block.

        The block discovers dominant periods via FFT, reconstructs the signal under
        each period using fold-conv-unfold, aggregates reconstructions using softmax
        weights derived from FFT amplitudes, and finally applies a residual addition.

        Args:
            x (torch.Tensor): Embedded input sequence of shape [b, t, c].
            total_len (int): Target length (seq_len + pred_len). This value is used
                to determine padding and folding grid size (rows) for each period.

        Returns:
            torch.Tensor: Output tensor of shape [b, t, c] after aggregation and residual.
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
