from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as f

from dataclasses import dataclass
from typing import Literal, Sequence, Optional, List
from .blocks import PastDecomposableMixing

ActivationName = Literal["gelu"]
DecompositionMethod = Literal["moving_average", "dft"]
DownsamplingMethod = Literal["avg", "max", "conv"]
PaddingMode = Literal["replicate", "reflect"]

@dataclass(frozen=True)
class TimeMixerConfig:
    """Configuration for the TimeMixer model."""
    historical_lookback_length: int = 96          # input sequence length 
    forecast_horizon_length: int = 96             # prediction length 
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

        return 
    
class TimeMixer(nn.Module):
    """
    TimeMixer forecasting model
    Input : x [B, L, C_in]
    Output: y [B, H, C_out]
    """

    # TODO: implement TimeMixer class