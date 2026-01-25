from __future__ import annotations

from typing import List, Literal, Optional

import torch
import torch.nn as nn

from .decomposition import SeriesDecomposition
from .timemixer import TimeMixerConfig

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
        config: "TimeMixerConfig",
        input_length: int,
        output_length: int,
        hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()

        if input_length < 1 or output_length < 1:
            raise ValueError("input_length and output_length must be >= 1")
        
        self.input_length = input_length
        self.output_length = output_length
        hidden = max(input_length, output_length) if hidden_dim is None else hidden_dim

        self.net = nn.Sequential(
            nn.Linear(input_length, hidden),
            makeActivation(config.activation_function_name),
            nn.Dropout(config.dropout_probability),
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
        config: "TimeMixerConfig",
        hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.input_lengths = config.get_multiscale_input_lengths()
        if len(self.input_lengths) < 2:
            raise ValueError("MultiscaleSeasonMixing requires at least 2 scales.")
        
        mixers: List[nn.Module] = []
        for i in range(len(self.input_lengths) - 1):
            t_in = self.input_lengths[i]
            t_out = self.input_lengths[i + 1]
            mixers.append(
                TemporalLinearMixer(
                    config=config,
                    input_length=t_in,
                    output_length=t_out,
                    hidden_dim=hidden_dim,
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
        config: "TimeMixerConfig",
        hidden_dim: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.input_lengths = config.get_multiscale_input_lengths()
        if len(self.input_lengths) < 2:
            raise ValueError("MultiscaleTrendMixing requires at least 2 scales.")

        # coarse -> fine:
        # for adjacent pair (i, i+1): map T_{i+1} -> T_i
        mixers: List[nn.Module] = []
        for i in reversed(range(len(self.input_lengths) - 1)):
            t_coarse = self.input_lengths[i + 1]
            t_fine = self.input_lengths[i]
            mixers.append(
                TemporalLinearMixer(
                    config=config,
                    input_length=t_coarse,
                    output_length=t_fine,
                    hidden_dim=hidden_dim,
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
    def __init__(self, config: TimeMixerConfig) -> None:
        super().__init__()
        self.config = config

        self.input_lengths = config.get_multiscale_input_lengths()
        self.model_embedding_dimension = config.model_embedding_dimension
        self.feedforward_hidden_dimension = config.feedforward_hidden_dimension
        self.dropout_probability = config.dropout_probability
        self.use_channel_independence = config.use_channel_independence
        self.use_output_residual_connection = config.use_output_residual_connection


        if config.decomposition_method_name != "moving_average":
            raise NotImplementedError(
                "our implementation only has moving_average decomposition. "
            )
        
        self.decomposition = SeriesDecomposition(
            kernel_size=config.moving_average_kernel_size,
            padding_mode=config.moving_average_padding_mode,
            center=config.moving_average_centered,
            validate_shapes=True,
        )

        # if channels are not independent allow interaction across feature dimension before mixing
        if not self.use_channel_independence:
            self.pre_mix_channel_ffn = nn.Sequential(
                nn.Linear(self.model_embedding_dimension, self.feedforward_hidden_dimension),
                makeActivation(config.activation_function_name),
                nn.Dropout(self.dropout_probability),
                nn.Linear(self.feedforward_hidden_dimension, self.model_embedding_dimension),
            )
        else:
            self.pre_mix_channel_ffn = None

        # mixing
        self.season_mixer = MultiscaleSeasonMixing(
            config=self.config,
            hidden_dim=None,
        )

        self.trend_mixer = MultiscaleTrendMixing(
            config=self.config,
            hidden_dim=None,
        )

        self.post_mix_ffn = nn.Sequential(
            nn.Linear(self.model_embedding_dimension, self.feedforward_hidden_dimension),
            makeActivation(config.activation_function_name),
            nn.Dropout(self.dropout_probability),
            nn.Linear(self.feedforward_hidden_dimension, self.model_embedding_dimension),
        )

        self.output_layer_norm = nn.LayerNorm(self.model_embedding_dimension)

    def forward(self, x_list: List[torch.Tensor]) -> List[torch.Tensor]:
        if len(x_list) != len(self.input_lengths):
            raise ValueError(
                f"Expected {len(self.input_lengths)} scales, got {len(x_list)}."
            )

        # shape checks
        for i, (x, expected_t) in enumerate(zip(x_list, self.input_lengths)):
            if x.ndim != 3:
                raise ValueError(f"Scale {i} expected [B, T, D], got {tuple(x.shape)}.")
            if x.shape[1] != expected_t:
                raise ValueError(f"Scale {i} expected T={expected_t}, got T={x.shape[1]}.")
            if x.shape[2] != self.model_embedding_dimension:
                raise ValueError(
                    f"Scale {i} expected D={self.model_embedding_dimension}, got D={x.shape[2]}."
                )

        # Decompose each scale: x = season + trend
        seasonal_components: List[torch.Tensor] = []
        trend_components: List[torch.Tensor] = []
        for x in x_list:
            season, trend = self.decomposition(x)

            if self.pre_mix_channel_ffn is not None:
                season = self.pre_mix_channel_ffn(season)
                trend = self.pre_mix_channel_ffn(trend)

            seasonal_components.append(season)
            trend_components.append(trend)

        # Multiscale mixing
        mixed_season = self.season_mixer(seasonal_components)  # fine -> coarse
        mixed_trend = self.trend_mixer(trend_components)       # coarse -> fine

        # Fuse back per scale
        out_list: List[torch.Tensor] = []
        for original_x, season_x, trend_x in zip(x_list, mixed_season, mixed_trend):
            fused = season_x + trend_x  

            if self.use_output_residual_connection:
                fused = original_x + self.post_mix_ffn(fused)

            fused = self.output_layer_norm(fused)
            out_list.append(fused)

        return out_list