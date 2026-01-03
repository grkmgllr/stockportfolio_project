"""
Convolutional layers for ModernTCN.

ModernTCN uses large-kernel depthwise separable convolutions
inspired by modern ConvNet designs (ConvNeXt, RepLKNet).
"""
import torch
import torch.nn as nn


class DWConv(nn.Module):
    """
    Depthwise Separable Convolution with large kernel.
    
    ModernTCN uses large kernels (e.g., 51) to capture long-range
    temporal dependencies efficiently via depthwise convolution.
    
    Structure:
    1. Depthwise Conv (per-channel conv with large kernel)
    2. Pointwise Conv (1x1 conv for channel mixing)
    """
    
    def __init__(
        self,
        d_model: int,
        kernel_size: int = 51,
        dilation: int = 1,
        groups: int = None
    ):
        super().__init__()
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.dilation = dilation
        
        # Depthwise convolution: each channel has its own filter
        # groups=d_model means depthwise
        groups = groups if groups is not None else d_model
        
        # Calculate padding for 'same' output size
        effective_kernel = kernel_size + (kernel_size - 1) * (dilation - 1)
        padding = effective_kernel // 2
        
        self.dw_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False
        )
        
        # Pointwise convolution: 1x1 conv for channel mixing
        self.pw_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=1,
            bias=True
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, d_model, L]
        
        Returns:
            Output tensor [B, d_model, L]
        """
        # Depthwise conv
        x = self.dw_conv(x)
        
        # Pointwise conv
        x = self.pw_conv(x)
        
        return x


class ConvFFN(nn.Module):
    """
    Convolutional Feed-Forward Network.
    
    Replaces traditional FFN (Linear->GELU->Linear) with:
    1x1 Conv -> GELU -> 1x1 Conv
    
    This maintains the convolutional nature of ModernTCN
    while providing channel-wise non-linearity.
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Expand channels
        self.fc1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        
        # Activation
        self.act = nn.GELU()
        
        # Project back
        self.fc2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, d_model, L]
        
        Returns:
            Output tensor [B, d_model, L]
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        
        return x


class MultiScaleDWConv(nn.Module):
    """
    Multi-scale Depthwise Convolution.
    
    Uses multiple kernel sizes to capture patterns at different
    temporal scales, similar to Inception-style multi-scale processing.
    """
    
    def __init__(
        self,
        d_model: int,
        kernel_sizes: list = [7, 15, 31, 51],
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.kernel_sizes = kernel_sizes
        self.num_scales = len(kernel_sizes)
        
        # Split channels across scales
        assert d_model % self.num_scales == 0, \
            f"d_model ({d_model}) must be divisible by num_scales ({self.num_scales})"
        
        self.channels_per_scale = d_model // self.num_scales
        
        # Create DWConv for each scale
        self.dwconvs = nn.ModuleList([
            DWConv(
                d_model=self.channels_per_scale,
                kernel_size=k
            )
            for k in kernel_sizes
        ])
        
        # Final pointwise to mix scales
        self.pw_final = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, d_model, L]
        
        Returns:
            Output tensor [B, d_model, L]
        """
        B, C, L = x.shape
        
        # Split channels for each scale
        x_splits = x.chunk(self.num_scales, dim=1)
        
        # Apply DWConv at each scale
        outputs = []
        for dwconv, x_scale in zip(self.dwconvs, x_splits):
            outputs.append(dwconv(x_scale))
        
        # Concatenate scale outputs
        x = torch.cat(outputs, dim=1)
        
        # Final pointwise mixing
        x = self.pw_final(x)
        x = self.dropout(x)
        
        return x

