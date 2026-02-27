"""
Utility functions for frequency-domain analysis in TimesNet.

This module contains helper routines used by TimesNet blocks to extract
dominant temporal patterns from time-series data. The functions defined
here operate purely on tensors and do not introduce trainable parameters.

The utilities are designed to be:

- stateless,
- deterministic,
- independent of model/training logic.

Currently implemented
---------------------
fft_top_periods
    Estimates dominant periods in a batch of time series using FFT-based
    amplitude analysis.
"""

from __future__ import annotations

import torch
from beartype import beartype

from .types import BTC


@beartype
@torch.no_grad()
def fft_top_periods(x: BTC, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Identify dominant temporal periods using Fast Fourier Transform (FFT).

    This function analyzes the frequency-domain representation of an input
    time-series tensor to estimate the most influential periodic components.
    It implements the dominant-period selection strategy used inside TimesNet
    blocks.

    Method
    ------
    (1) Apply real-valued FFT (rFFT) along the time dimension.
    (2) Compute amplitude (magnitude) of frequency components.
    (3) Average amplitudes across batch and channel dimensions.
    (4) Suppress the DC (zero-frequency) component.
    (5) Select the top-k frequency indices with highest amplitudes.
    (6) Convert frequency indices into period lengths via:

            period = T // freq_index

    where T is the sequence length.

    Args
    ----
    x : BTC
        Input tensor of shape [B, T, C], where:
            B = batch size,
            T = sequence length,
            C = number of channels/features.

    k : int
        Number of dominant frequency components to select.
        Must satisfy k ≥ 1.

    Returns
    -------
    periods : torch.Tensor
        Tensor of shape [k_eff] containing estimated period lengths.
        dtype = torch.int64

    weights : torch.Tensor
        Tensor of shape [B, k_eff] containing per-sample amplitude
        weights corresponding to selected frequencies.

    Raises
    ------
    ValueError
        If k < 1.
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
