"""
StockMixer — cross-stock forecasting model.

Faithful port of https://github.com/SJTU-DMTai/StockMixer (SJTU-DMTai) with
two minimal changes so it fits this repository's per-sample interface:

1. Adds a leading batch dimension: input ``[B, N, L, F_in]`` /
   output ``[B, N, H, C_out]``. The reference implementation is unbatched
   and produces one scalar per stock; we keep the same architecture and
   only widen the final projection to produce ``pred_len * c_out`` values
   per stock, then reshape.
2. Derives ``scale_dim`` from the lookback length (``L // 2``) instead of
   the hard-coded ``8`` used in the reference (which assumed ``L=16``).

Everything else — the two-scale Mixer2dTriU tower, the cross-stock
``NoGraphMixer`` branch, the summing of the temporal and cross-stock
outputs — is preserved. See :mod:`.blocks` for the primitives.

This model is trained on return targets (``return_targets=True``),
matching the reference paper's loss (MSE + rank loss on returns).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .blocks import MultTime2dMixer, NoGraphMixer


@dataclass(frozen=True)
class StockMixerConfig:
    """Configuration for :class:`StockMixer`.

    Attributes:
        seq_len: Lookback length ``L``. Must be even (the stride-2 conv on
            the time axis halves it into the coarse scale).
        pred_len: Forecast horizon ``H``.
        enc_in: Number of input channels per stock (``F_in``).
        c_out: Number of output channels per stock (``C_out``).
        num_stocks: Number of stocks joined in each sample (``N``).
        market_dim: Hidden bottleneck of the cross-stock mixer.
            Original paper uses 20.
        denorm_indices: Kept for interface parity with TimesNet/TimeMixer;
            not used here because StockMixer is trained on returns.
        return_targets: Must be True — StockMixer is a return-basis model.
            False raises ValueError at construction time.
    """
    seq_len: int
    pred_len: int
    enc_in: int
    c_out: int
    num_stocks: int
    market_dim: int = 20
    denorm_indices: Optional[tuple] = None
    return_targets: bool = True


class StockMixer(nn.Module):
    """Cross-stock return forecaster.

    Combines a two-scale temporal mixer (per stock) with a cross-stock
    ``NoGraphMixer`` branch. The two branches produce ``[B, N, H*C_out]``
    each; their sum is reshaped to ``[B, N, H, C_out]``.
    """

    def __init__(self, config: StockMixerConfig):
        super().__init__()
        if not config.return_targets:
            raise ValueError(
                "StockMixer only supports return_targets=True "
                "(the reference implementation is a return-basis model)."
            )
        if config.seq_len % 2 != 0:
            raise ValueError(
                f"seq_len must be even (got {config.seq_len}); the coarse "
                f"scale halves the time axis via a stride-2 convolution."
            )

        self.config = config
        L = config.seq_len
        H = config.pred_len
        F_in = config.enc_in
        C_out = config.c_out
        N = config.num_stocks

        # Coarse scale length after the stride-2 conv on the time axis.
        scale_dim = L // 2
        # After MultTime2dMixer the time axis becomes
        # [original L] + [mixed L] + [mixed scale_dim] = 2L + scale_dim.
        time_concat = L * 2 + scale_dim
        # Each stock's output slot after the temporal branch.
        out_per_stock = H * C_out

        self.mixer = MultTime2dMixer(L, F_in, scale_dim=scale_dim)
        # Stride-2 conv that produces the coarse-scale input for the mixer.
        # Applied along the time axis, so channels stay unchanged.
        self.conv = nn.Conv1d(
            in_channels=F_in, out_channels=F_in, kernel_size=2, stride=2,
        )
        # Collapse the channel axis (per-stock, per-timestep).
        self.channel_fc = nn.Linear(F_in, 1)
        # Two temporal projections summed at the end — one on the plain
        # temporal features, one after the cross-stock mixer. The reference
        # keeps them as separate linears (`time_fc`, `time_fc_`) with the
        # same shape; we do the same, just widened from `1` to H*C_out.
        self.time_fc = nn.Linear(time_concat, out_per_stock)
        self.time_fc_ = nn.Linear(time_concat, out_per_stock)
        self.stock_mixer = NoGraphMixer(N, hidden_dim=config.market_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: ``[B, N, L, F_in]`` — one batch of joint time-windows.

        Returns:
            ``[B, N, H, C_out]`` — per-stock forecast (returns).
        """
        cfg = self.config
        B, N, L, F_in = x.shape
        if N != cfg.num_stocks:
            raise ValueError(
                f"num_stocks mismatch: model built for {cfg.num_stocks}, "
                f"got {N} in the input."
            )

        # Fuse batch and stock dims for the per-stock temporal ops. All
        # layers up to (and including) `channel_fc` treat each (b, n) pair
        # as an independent sample.
        x_flat = x.reshape(B * N, L, F_in)

        # Coarse scale: [B*N, L, F] -> [B*N, F, L] -> conv (halves L)
        # -> [B*N, F, L/2] -> [B*N, L/2, F]
        y = x_flat.permute(0, 2, 1)
        y = self.conv(y)
        y = y.permute(0, 2, 1)

        # Two-scale temporal mixing -> [B*N, 2L + L/2, F_in]
        feats = self.mixer(x_flat, y)
        # Channel collapse -> [B*N, 2L + L/2, 1] -> squeeze
        feats = self.channel_fc(feats).squeeze(-1)

        # Restore the stock axis for the cross-stock branch.
        feats = feats.reshape(B, N, -1)                    # [B, N, 2L+L/2]

        # Cross-stock mixing: same shape in, same shape out.
        stock_feats = self.stock_mixer(feats)              # [B, N, 2L+L/2]

        # Two temporal projections summed — original architecture.
        temporal_out = self.time_fc(feats)                 # [B, N, H*C_out]
        crossstock_out = self.time_fc_(stock_feats)        # [B, N, H*C_out]
        out = temporal_out + crossstock_out

        return out.reshape(B, N, cfg.pred_len, cfg.c_out)
