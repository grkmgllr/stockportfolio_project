from __future__ import annotations

import torch
from beartype import beartype

from .types import BTC


@beartype
@torch.no_grad()
def fft_top_periods(x: BTC, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate dominant periods via FFT amplitudes.

    Args:
        x: [b, t, c] time series.
        k: number of top frequency indices to select (k >= 1).

    Returns:
        periods: int64 tensor, shape [k_eff]; each is t // freq_idx.
        weights: float tensor, shape [b, k_eff]; per-batch amplitudes.
    """
    if k < 1:
        raise ValueError("k must be >= 1")

    b, t, c = x.shape

    # number of usable non-DC frequency bins in rFFT is at most t//2
    k_eff = min(k, max(t // 2, 1))

    # compute rFFT along time axis
    xf = torch.fft.rfft(x.to(torch.float32), dim=1)  # [b, t//2+1, c]

    # mean amplitude over batch & channels -> [t//2+1]
    amp_mean = xf.abs().mean(dim=0).mean(dim=-1)
    amp_mean[0] = 0.0  # suppress DC

    # top-k indices by amplitude
    _, idx = torch.topk(amp_mean, k_eff)  # [k_eff]

    # periods as t // freq_idx (int64, on same device)
    periods = (t // idx).to(torch.long)

    # per-batch weights for those indices -> [b, k_eff]
    batch_amp = xf.abs().mean(dim=-1)  # [b, t//2+1]
    weights = batch_amp[:, idx].to(dtype=x.dtype)

    return periods, weights
