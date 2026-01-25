from __future__ import annotations

from typing import List, Literal, Optional

import torch
import torch.nn as nn

from .decomposition import SeriesDecomposition

ActivationName = Literal["gelu"]

def makeActivation(name: ActivationName) -> nn.Module:
    """
    Factory function for activation layers
    """
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")

class TemporalLinearMixer(nn.Module):
    """
    Mix along time with an MLP to align scales.

    I/O: x [B, T_in, D] -> y [B, T_out, D]
    """
    def __init__(
        self,
        input_length: int,
        output_length: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        activation: ActivationName = "gelu",
    ) -> None:
        super().__init__()

        if input_length < 1 or output_length < 1:
            raise ValueError("input_length and output_length must be >= 1")
        if not (0.0 <= dropout < 1.0):
            raise ValueError("dropout must be in [0.0, 1.0).")
        
        self.input_length = input_length
        self.output_length = output_length
        hidden = max(input_length, output_length) if hidden_dim is None else hidden_dim

        self.net = nn.Sequential(
            nn.Linear(input_length, hidden),
            makeActivation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden, output_length),
        )

    def forward(self, time_series: torch.Tensor) -> torch.Tensor:
        
        if time_series.ndim != 3:
            raise ValueError(f"Expected [B, T, D], got {tuple(time_series.shape)}.")

        b, t, d = time_series.shape
        if t != self.input_length:
            raise ValueError(
                f"TemporalLinearMixer expected T_in={self.input_length}, but got T={t}."
            )

        x = time_series.transpose(1, 2).contiguous()  # [B, D, T_in]
        y = self.net(x)                  # [B, D, T_out]
        return y.transpose(1, 2)          # [B, T_out, D]


class MultiscaleSeasonMixing(nn.Module):
    """
    Bottom-up (fine->coarse) residual mixing for seasonal components across scales

    I/O: [s0..sM], s_i [B, T_i, D] -> [out0..outM], out_i [B, T_i, D]
    """
    def __init__(
        self,
        input_lengths: List[int],
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        activation: ActivationName = "gelu",
    ) -> None:
        super().__init__()

        if len(input_lengths) < 2:
            raise ValueError("MultiscaleSeasonMixing requires at least 2 scales.")
        
        self.input_lengths = input_lengths

        mixers: List[nn.Module] = []
        for i in range(len(input_lengths) - 1):
            t_in = input_lengths[i]
            t_out = input_lengths[i + 1]
            mixers.append(
                TemporalLinearMixer(
                    input_length=t_in,
                    output_length=t_out,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    activation=activation,
                )
            )
        self.down_mixers = nn.ModuleList(mixers)
    
    def forward(self, seasonal_components: List[torch.Tensor]) -> List[torch.Tensor]:
        if len(seasonal_components) != len(self.input_lengths):
            raise ValueError(
                f"Expected {len(self.input_lengths)} scales, got {len(seasonal_components)}."
            )

        for i, (x, t_expected) in enumerate(zip(seasonal_components, self.input_lengths)):
            if x.ndim != 3:
                raise ValueError(f"Scale {i} expected [B, T, D], got {tuple(x.shape)}.")
            if x.shape[1] != t_expected:
                raise ValueError(
                    f"Scale {i} expected T={t_expected}, got T={x.shape[1]}."
                )

        out_high = seasonal_components[0]   # [B, T0, D] - fine seasonal
        out_low = seasonal_components[1]    # [B, T1, D] - coarse seasonal
        mixed: List[torch.Tensor] = [out_high]

        for i in range(len(seasonal_components) - 1): # from one scale to the next
            low_residual = self.down_mixers[i](out_high)  # [B, T(i+1), D]
            out_low = out_low + low_residual # residual update

            out_high = out_low # update: new high scale is mixed coarse scale
            if i + 2 <= len(seasonal_components) - 1:
                out_low = seasonal_components[i + 2]

            mixed.append(out_high)

        return mixed


class MultiscaleTrendMixing(nn.Module):
    """
    Top-down (coarse->fine) residual mixing for trend components across scales

    I/O: [t0..tM], t_i [B, T_i, D] -> [out0..outM], out_i [B, T_i, D]
    """

    def __init__(
        self,
        input_lengths: List[int],
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        activation: ActivationName = "gelu",
    ) -> None:
        super().__init__()

        if len(input_lengths) < 2:
            raise ValueError("MultiscaleTrendMixing requires at least 2 scales.")

        self.input_lengths = input_lengths

        # coarse -> fine:
        # for adjacent pair (i, i+1): map T_{i+1} -> T_i
        mixers: List[nn.Module] = []
        for i in reversed(range(len(input_lengths) - 1)):
            t_coarse = input_lengths[i + 1]
            t_fine = input_lengths[i]
            mixers.append(
                TemporalLinearMixer(
                    input_length=t_coarse,
                    output_length=t_fine,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    activation=activation,
                )
            )

        self.up_mixers = nn.ModuleList(mixers)

    def forward(self, trend_components: List[torch.Tensor]) -> List[torch.Tensor]:

        if len(trend_components) != len(self.input_lengths):
            raise ValueError(
                f"Expected {len(self.input_lengths)} scales, got {len(trend_components)}."
            )

        for i, (x, t_expected) in enumerate(zip(trend_components, self.input_lengths)):
            if x.ndim != 3:
                raise ValueError(f"Scale {i} expected [B, T, D], got {tuple(x.shape)}.")
            if x.shape[1] != t_expected:
                raise ValueError(
                    f"Scale {i} expected T={t_expected}, got T={x.shape[1]}."
                )

        # reverse to start mixing from the coarsest trend
        reversed_trends = list(reversed(trend_components))

        out_low = reversed_trends[0]   # coarsest: [B, T_coarse, D]
        out_high = reversed_trends[1]  # next finer: [B, T_next, D]
        mixed_reversed: List[torch.Tensor] = [out_low]

        for i in range(len(reversed_trends) - 1):
            high_residual = self.up_mixers[i](out_low)  # [B, T_next_finer, D]
            out_high = out_high + high_residual

            out_low = out_high
            if i + 2 <= len(reversed_trends) - 1:
                out_high = reversed_trends[i + 2]

            mixed_reversed.append(out_low)

        # convert back to finest->coarsest
        mixed_reversed.reverse()

        return mixed_reversed

class PastDecomposableMixing(nn.Module):
    """
    PDM block: decompose each scale into season/trend,
    apply bottom-up seasonal mixing + top-down trend mixing, then fuse back

    I/O: x_list (len=M+1), x_i [B, T_i, D] -> out_list, out_i [B, T_i, D]   
    """
    # TODO: complete PDM block implementation