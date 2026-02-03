from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as f

from dataclasses import dataclass
from typing import Literal, Sequence, Optional, List
from .blocks import PastDecomposableMixing, makeActivation

ActivationName = Literal["gelu"]
DecompositionMethod = Literal["moving_average", "dft"]
DownsamplingMethod = Literal["avg", "max", "conv"]
PaddingMode = Literal["replicate", "reflect"]

@dataclass(frozen=True)
class TimeMixerConfig:
    """Configuration for the TimeMixer model."""
    historical_lookback_length: int = 96        
    forecast_horizon_length: int = 96            
    number_of_input_features: int = 7             
    number_of_output_features: int = 7            

    model_embedding_dimension: int = 128         
    feedforward_hidden_dimension: int = 512      
    number_of_pdm_blocks: int = 2                
    dropout_probability: float = 0.1
    activation_function_name: ActivationName = "gelu"

    downsampling_window_size: int = 2             
    number_of_downsampling_layers: int = 1        
    downsampling_method_name: DownsamplingMethod = "avg"

    multiscale_input_lengths: Sequence[int] | None = None

    # Decomposition settings
    decomposition_method_name: DecompositionMethod = "moving_average"
    moving_average_kernel_size: int = 25
    dft_top_k_frequencies: int = 5               
    moving_average_centered: bool = True
    moving_average_padding_mode: PaddingMode = "replicate"

    # False => allow cross channel mixing
    use_channel_independence: bool = False

    use_output_residual_connection: bool = True

    # Compatibility properties for unified interface with TimesNetForecastConfig
    @property
    def seq_len(self) -> int:
        """Alias for historical_lookback_length."""
        return self.historical_lookback_length

    @property
    def pred_len(self) -> int:
        """Alias for forecast_horizon_length."""
        return self.forecast_horizon_length

    @property
    def enc_in(self) -> int:
        """Alias for number_of_input_features."""
        return self.number_of_input_features

    @property
    def c_out(self) -> int:
        """Alias for number_of_output_features."""
        return self.number_of_output_features

    def get_multiscale_input_lengths(self) -> list[int]:
        """
        Returns the input length for each scale, from fine -> coarse
        If predefined lengths are provided, they are used
        Otherwise lengths are obtained by downsampling the lookback
        window multiple times
        """
        if self.multiscale_input_lengths is not None:
            return list(self.multiscale_input_lengths)

        lengths = [self.historical_lookback_length]
        current = self.historical_lookback_length

        for _ in range(self.number_of_downsampling_layers):
            if current % self.downsampling_window_size != 0:
                raise ValueError(
                    "historical_lookback_length must be divisible by downsampling_window_size "
                    f"at each scale. Got length={current}, window={self.downsampling_window_size}."
                )
            current = current // self.downsampling_window_size
            lengths.append(current)

        return lengths
    
class TimeMixer(nn.Module):
    """
    TimeMixer forecasting model
    Input : x [B, L, C_in]
    Output: y [B, H, C_out]
    """
    def __init__(self, config: TimeMixerConfig) -> None:
        super().__init__()
        self.config = config

        self.lookback_length = config.historical_lookback_length
        self.horizon_length = config.forecast_horizon_length
        self.input_features = config.number_of_input_features
        self.output_features = config.number_of_output_features
        self.d_model = config.model_embedding_dimension

        self.scale_lengths = config.get_multiscale_input_lengths()

        self.downsampling_window_size = config.downsampling_window_size
        self.downsampling_method_name = config.downsampling_method_name

        if self.downsampling_method_name == "avg": # used method for downsampling in paper
            self.downsampler = nn.AvgPool1d(kernel_size=self.downsampling_window_size)
        elif self.downsampling_method_name == "max":
            self.downsampler = nn.MaxPool1d(kernel_size=self.downsampling_window_size)
        elif self.downsampling_method_name == "conv":
            self.downsampler = nn.Conv1d(
                in_channels=self.input_features,
                out_channels=self.input_features,
                kernel_size=3,
                stride=self.downsampling_window_size,
                padding=1,
                groups=self.input_features,
                bias=False,
            )
        else:
            raise ValueError(f"Unsupported method: {self.downsampling_method_name}")
        
        # embedding [B,T,C_in] -> [B,T,D]
        self.scale_embeddings = nn.ModuleList(
            [nn.Linear(self.input_features, self.d_model) for _ in range(len(self.scale_lengths))]
        )

        # stacked PDM blocks
        self.pdm_blocks = nn.ModuleList(
            [PastDecomposableMixing(config) for _ in range(config.number_of_pdm_blocks)]
        )

        #FMM / one predictor per scale (predict H from each scale)
        self.scale_time_predictors = nn.ModuleList(
            [nn.Linear(t_i, self.horizon_length) for t_i in self.scale_lengths]
        )

        # project from d_model -> c_out
        self.output_projection = nn.Linear(self.d_model, self.output_features)

        self.dropout = nn.Dropout(config.dropout_probability)
        self.activation = makeActivation(config.activation_function_name)


    def multiscale_inputs(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Create multi-scale observations by downsampling along time.
        Returns list ordered finest->coarsest, each [B, T_i, C_in].
        """
        if x.ndim != 3:
            raise ValueError(f"Expected x [B, L, C], got {tuple(x.shape)}.")
        if x.shape[1] != self.lookback_length:
            raise ValueError(f"Expected L={self.lookback_length}, got L={x.shape[1]}.")
        if x.shape[2] != self.input_features:
            raise ValueError(f"Expected C_in={self.input_features}, got C_in={x.shape[2]}.")

        multiscale: List[torch.Tensor] = [x]  # first element is the finest

        # work in [B, C, T] for pool/conv
        x_ct = x.transpose(1, 2).contiguous()  # [B, C_in, L]
        current_ct = x_ct

        # downsample loop
        for _ in range(len(self.scale_lengths) - 1):
            current_ct = self.downsampler(current_ct)          # [B, C_in, T_next]
            multiscale.append(current_ct.transpose(1, 2))      # [B, T_next, C_in]

        # validate lengths
        for i, (xi, ti) in enumerate(zip(multiscale, self.scale_lengths)):
            if xi.shape[1] != ti:
                raise ValueError(f"Scale {i} expected T={ti}, got T={xi.shape[1]}.")
        return multiscale

    def embed_multiscale(self, x_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Apply per scale embedding to get [B, T_i, D]
        """
        embedded: List[torch.Tensor] = []
        for x, emb in zip(x_list, self.scale_embeddings):
            z = emb(x)                  # [B, T, D]
            z = self.dropout(z)
            embedded.append(z)
        return embedded

    def fmulti_predictor_mixing(self, encoded_list: List[torch.Tensor]) -> torch.Tensor:
        """
        FMM: each scale makes a forecast, then ensemble.

        encoded_i: [B, T_i, D]
        predictor maps T_i -> H on each feature channel of D.
        """
        per_scale_forecasts: List[torch.Tensor] = []

        for encoded_i, predictor in zip(encoded_list, self.scale_time_predictors):
            # [B, T_i, D] -> [B, D, T_i]
            x = encoded_i.transpose(1, 2).contiguous()
            # Linear expects last dim = T_i, outputs H: [B, D, H]
            y = predictor(x)
            # back to [B, H, D]
            y = y.transpose(1, 2).contiguous()
            per_scale_forecasts.append(y)

        # ensemble: sum 
        fused = torch.stack(per_scale_forecasts, dim=-1).sum(dim=-1)  # [B, H, D]
        return fused

    def forward(self, x: torch.Tensor, x_mark: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for TimeMixer.
        
        Args:
            x: Input tensor [B, L, C_in]
            x_mark: Optional time features (unused, for interface compatibility)
            
        Returns:
            y: Output tensor [B, H, C_out]
        """
        # multiscale observations
        x_scales = self.multiscale_inputs(x)  # [B,T_i,C_in]
        # embed
        z_scales = self.embed_multiscale(x_scales)  # [B,T_i,D]
        # PDM encoder stack
        for pdm in self.pdm_blocks:
            z_scales = pdm(z_scales)
        # FMM forecasting
        future_latent = self.fmulti_predictor_mixing(z_scales)  # [B,H,D]
        # output projection
        y = self.output_projection(future_latent)  # [B,H,C_out]
        return y
    