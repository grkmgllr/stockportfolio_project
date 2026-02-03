"""
Utility functions for TimesNetPure.

This module contains helper routines that are shared across the TimesNetPure
implementation. Currently, it focuses on frequency-domain analysis used by
TimesNet blocks to identify dominant temporal periods in time-series data.

The utilities defined here are:
    - stateless,
    - free of trainable parameters,
    - independent of training logic.

They operate purely on input tensors and return data-dependent outputs without maintaining state.
"""

from __future__ import annotations

from typing import Tuple

import torch
from beartype import beartype

from .types import BTC


@beartype
@torch.no_grad()
def fft_top_periods(x: BTC, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Identify dominant temporal periods in a batch of time series using FFT.

    This function estimates the most influential periodic components of a
    time-series signal by analyzing its frequency-domain representation.
    It follows the core idea used in TimesNet blocks:

    (1) Transform the input sequence from the time domain to the frequency
        domain using the real-valued Fast Fourier Transform (rFFT).
    (2) Compute the magnitude (amplitude) of each frequency component.
    (3) Aggregate amplitudes across the batch and channel dimensions to
        obtain a global importance score per frequency.
    (4) Select the top-k frequency indices with the largest amplitudes,
        excluding the DC component.
    (5) Convert the selected frequency indices into period lengths using
        the relation:
            period ≈ T / f_idx

        where T is the sequence length and f_idx is the selected FFT bin index.
        In this implementation, periods are computed as floor(T / f_idx).

    The resulting periods represent dominant repeating patterns in the
    input sequence and are used by TimesNet blocks to fold the sequence
    into a 2D representation.

    Notes
    -----
    - The DC component (zero frequency) is explicitly suppressed, as it
      corresponds to the mean of the signal rather than a periodic pattern.
    - The effective number of selected frequencies is bounded by T // 2,
      which is the maximum number of non-DC frequency bins in rFFT.
    - For very short sequences (T < 2), the function falls back to a
      trivial period of 1.

    Args
    ----
    x : BTC
        Input time-series tensor of shape [B, T, C], where:
            B = batch size,
            T = sequence length,
            C = number of channels/features.
    k : int
        Number of dominant frequencies (periods) to select.
        Must satisfy k >= 1.

    Returns
    -------
    periods : torch.Tensor
        Integer tensor of shape [k_eff], where each value represents an
        estimated period length (>= 1). Here, k_eff = min(k, max(T // 2, 1)).
    weights : torch.Tensor
        Floating-point tensor of shape [B, k_eff] containing per-sample
        amplitude weights for the selected frequencies. These weights are
        typically used to aggregate reconstructions across periods.

    Raises
    ------
    ValueError
        If k < 1.
    ValueError
        If the input tensor x does not have rank 3.
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