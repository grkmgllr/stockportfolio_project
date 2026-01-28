from __future__ import annotations

from typing import Tuple

import torch
from beartype import beartype

from .types import BTC


@beartype
@torch.no_grad()
def fft_top_periods(x: BTC, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Estimate dominant periods via FFT amplitudes.

    The function computes rFFT along the time axis, finds the top-k frequency
    indices by mean amplitude (excluding DC), and converts them to periods.

    Args:
        x: [b, t, c] input time series (batch, time, channels).
        k: number of dominant frequencies/periods to select (k >= 1).

    Returns:
        periods: int64 tensor of shape [k_eff]. Each element is an estimated period length.
        weights: float tensor of shape [b, k_eff]. Per-batch amplitudes for the selected frequencies.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if x.dim() != 3:
        raise ValueError(f"Expected x shape [b,t,c], got {tuple(x.shape)}")

    b, t, c = x.shape
    if t < 2:
        # Not enough length to do meaningful FFT; fall back to period=1
        periods = torch.ones(1, device=x.device, dtype=torch.long)
        weights = torch.ones(b, 1, device=x.device, dtype=x.dtype)
        return periods, weights

    # number of usable non-DC frequency bins in rFFT is at most t//2
    k_eff = min(k, max(t // 2, 1))

    # rFFT along time axis
    xf = torch.fft.rfft(x.to(torch.float32), dim=1)  # [b, t//2+1, c]

    # mean amplitude over batch & channels -> [t//2+1]
    amp_mean = xf.abs().mean(dim=0).mean(dim=-1)
    amp_mean[0] = 0.0  # suppress DC component

    # top-k indices by amplitude
    _, idx = torch.topk(amp_mean, k_eff)  # [k_eff]

    # Convert frequency indices to periods.
    # Ensure idx is at least 1 to avoid division by zero.
    idx = torch.clamp(idx, min=1)

    # periods: t // idx  (>= 1)
    periods = (t // idx).clamp(min=1).to(torch.long)

    # per-batch weights for those indices -> [b, k_eff]
    batch_amp = xf.abs().mean(dim=-1)  # [b, t//2+1]
    weights = batch_amp[:, idx].to(dtype=x.dtype)

    return periods, weights
