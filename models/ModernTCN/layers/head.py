"""
Prediction head for ModernTCN.

Projects encoded features to prediction horizon.
"""
import torch
import torch.nn as nn


class ForecastHead(nn.Module):
    """
    Forecasting head for time series prediction.
    
    Takes encoded patch representations and projects to prediction horizon.
    
    Architecture:
    1. Flatten patches: [B, d_model, num_patches] -> [B, d_model * num_patches]
    2. Project to prediction: [B, d_model * num_patches] -> [B, pred_len * c_out]
    3. Reshape: [B, pred_len * c_out] -> [B, pred_len, c_out]
    """
    
    def __init__(
        self,
        d_model: int,
        num_patches: int,
        pred_len: int,
        c_out: int,
        dropout: float = 0.1,
        use_norm: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.num_patches = num_patches
        self.pred_len = pred_len
        self.c_out = c_out
        
        flatten_dim = d_model * num_patches
        output_dim = pred_len * c_out
        
        # Optional normalization before projection
        self.norm = nn.LayerNorm(d_model) if use_norm else nn.Identity()
        
        # Projection layers
        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(flatten_dim, flatten_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(flatten_dim // 2, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Encoded features [B, d_model, num_patches]
        
        Returns:
            Predictions [B, pred_len, c_out]
        """
        B = x.shape[0]
        
        # Transpose for LayerNorm: [B, d_model, L] -> [B, L, d_model]
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2)  # Back to [B, d_model, L]
        
        # Project to predictions
        x = self.head(x)  # [B, pred_len * c_out]
        
        # Reshape to [B, pred_len, c_out]
        x = x.view(B, self.pred_len, self.c_out)
        
        return x


class LinearHead(nn.Module):
    """
    Simple linear projection head.
    
    More lightweight alternative that projects each patch independently
    and combines for prediction.
    """
    
    def __init__(
        self,
        d_model: int,
        num_patches: int,
        pred_len: int,
        c_out: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.num_patches = num_patches
        self.pred_len = pred_len
        self.c_out = c_out
        
        # Project d_model to c_out for each time step
        self.channel_proj = nn.Linear(d_model, c_out)
        
        # Project num_patches to pred_len
        self.temporal_proj = nn.Linear(num_patches, pred_len)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Encoded features [B, d_model, num_patches]
        
        Returns:
            Predictions [B, pred_len, c_out]
        """
        # [B, d_model, num_patches] -> [B, num_patches, d_model]
        x = x.transpose(1, 2)
        
        # Project channels: [B, num_patches, d_model] -> [B, num_patches, c_out]
        x = self.channel_proj(x)
        x = self.dropout(x)
        
        # Project time: [B, num_patches, c_out] -> [B, c_out, num_patches]
        x = x.transpose(1, 2)
        
        # [B, c_out, num_patches] -> [B, c_out, pred_len]
        x = self.temporal_proj(x)
        
        # [B, c_out, pred_len] -> [B, pred_len, c_out]
        x = x.transpose(1, 2)
        
        return x


class ChannelIndependentHead(nn.Module):
    """
    Channel-independent prediction head.
    
    Treats each output channel independently, projecting from
    patch representations. Useful when output channels have
    different characteristics.
    """
    
    def __init__(
        self,
        d_model: int,
        num_patches: int,
        pred_len: int,
        c_out: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.num_patches = num_patches
        self.pred_len = pred_len
        self.c_out = c_out
        
        # Independent projection for each output channel
        self.channel_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model * num_patches, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, pred_len)
            )
            for _ in range(c_out)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Encoded features [B, d_model, num_patches]
        
        Returns:
            Predictions [B, pred_len, c_out]
        """
        B = x.shape[0]
        
        # Flatten: [B, d_model, num_patches] -> [B, d_model * num_patches]
        x_flat = x.flatten(start_dim=1)
        
        # Apply each channel head
        outputs = []
        for head in self.channel_heads:
            out = head(x_flat)  # [B, pred_len]
            outputs.append(out)
        
        # Stack: [B, pred_len, c_out]
        x = torch.stack(outputs, dim=-1)
        
        return x

