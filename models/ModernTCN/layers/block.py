"""
ModernTCN Block - Core building block of ModernTCN.

Each block consists of:
1. Large-kernel depthwise convolution
2. Layer normalization
3. Convolutional FFN
4. Residual connections
"""
import torch
import torch.nn as nn

from .conv import DWConv, ConvFFN, MultiScaleDWConv
from .norm import LayerNorm, BatchNorm


class ModernTCNBlock(nn.Module):
    """
    Single ModernTCN block.
    
    Architecture (ConvNeXt-style for temporal data):
    
        Input
          │
          ├───────────────────┐
          │                   │
          ▼                   │
       DWConv (large kernel)  │
          │                   │
          ▼                   │
       LayerNorm              │
          │                   │
          ▼                   │
       ConvFFN                │
          │                   │
          ▼                   │
       Dropout                │
          │                   │
          └───────── + ◄──────┘
                     │
                  Output
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        kernel_size: int = 51,
        dropout: float = 0.1,
        layer_scale_init: float = 1e-6,
        use_multi_scale: bool = False,
        norm_type: str = 'layer'
    ):
        super().__init__()
        self.d_model = d_model
        self.use_layer_scale = layer_scale_init is not None
        
        # Depthwise convolution (large kernel)
        if use_multi_scale:
            self.dwconv = MultiScaleDWConv(d_model, dropout=dropout)
        else:
            self.dwconv = DWConv(d_model, kernel_size=kernel_size)
        
        # Normalization
        if norm_type == 'layer':
            self.norm = LayerNorm(d_model)
        elif norm_type == 'batch':
            self.norm = BatchNorm(d_model)
        else:
            raise ValueError(f"Unknown norm type: {norm_type}")
        
        # Convolutional FFN
        self.ffn = ConvFFN(d_model, d_ff, dropout=dropout)
        
        # Optional layer scale (from ConvNeXt)
        if self.use_layer_scale:
            self.layer_scale = nn.Parameter(
                layer_scale_init * torch.ones(d_model),
                requires_grad=True
            )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, d_model, L]
        
        Returns:
            Output tensor [B, d_model, L]
        """
        residual = x
        
        # DWConv
        x = self.dwconv(x)
        
        # Norm
        x = self.norm(x)
        
        # FFN
        x = self.ffn(x)
        
        # Layer scale
        if self.use_layer_scale:
            x = x * self.layer_scale.view(1, -1, 1)
        
        # Dropout + Residual
        x = self.dropout(x) + residual
        
        return x


class ModernTCNStage(nn.Module):
    """
    A stage of multiple ModernTCN blocks.
    
    Optionally includes downsampling between stages for
    hierarchical feature extraction.
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_blocks: int = 2,
        kernel_size: int = 51,
        dropout: float = 0.1,
        downsample: bool = False,
        downsample_factor: int = 2,
        use_multi_scale: bool = False
    ):
        super().__init__()
        self.downsample = downsample
        
        # Stack of blocks
        self.blocks = nn.ModuleList([
            ModernTCNBlock(
                d_model=d_model,
                d_ff=d_ff,
                kernel_size=kernel_size,
                dropout=dropout,
                use_multi_scale=use_multi_scale
            )
            for _ in range(num_blocks)
        ])
        
        # Optional downsampling
        if downsample:
            self.down_conv = nn.Conv1d(
                d_model, d_model,
                kernel_size=downsample_factor,
                stride=downsample_factor
            )
            self.down_norm = LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, d_model, L]
        
        Returns:
            Output tensor [B, d_model, L'] where L' = L or L/downsample_factor
        """
        # Apply blocks
        for block in self.blocks:
            x = block(x)
        
        # Optional downsampling
        if self.downsample:
            x = self.down_conv(x)
            x = self.down_norm(x)
        
        return x

