"""
Normalization layers for ModernTCN.

Provides various normalization strategies suitable for
temporal convolutional networks.
"""
import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    Layer Normalization for 1D convolutional features.
    
    Normalizes across the channel dimension for each time step.
    Adapted from ConvNeXt-style LayerNorm for temporal data.
    """
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, d_model, L]
        
        Returns:
            Normalized tensor [B, d_model, L]
        """
        # Normalize across channel dimension
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        
        # Apply learnable parameters
        x = self.weight.view(1, -1, 1) * x + self.bias.view(1, -1, 1)
        
        return x


class BatchNorm(nn.Module):
    """
    Batch Normalization wrapper for 1D temporal features.
    
    Uses standard BatchNorm1d but provides consistent interface.
    """
    
    def __init__(self, d_model: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.bn = nn.BatchNorm1d(d_model, eps=eps, momentum=momentum)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, d_model, L]
        
        Returns:
            Normalized tensor [B, d_model, L]
        """
        return self.bn(x)


class RevIN(nn.Module):
    """
    Reversible Instance Normalization.
    
    Instance normalization that can be reversed during inference.
    Particularly useful for time series forecasting to handle
    distribution shift between train and test.
    """
    
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        
        if self.affine:
            self.weight = nn.Parameter(torch.ones(1, num_features, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_features, 1))
        
        # Store statistics for denormalization
        self.mean = None
        self.std = None
    
    def forward(self, x: torch.Tensor, mode: str = 'norm') -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, C, L] or [B, L, C] depending on context
            mode: 'norm' for normalization, 'denorm' for denormalization
        
        Returns:
            Normalized or denormalized tensor
        """
        if mode == 'norm':
            return self._normalize(x)
        elif mode == 'denorm':
            return self._denormalize(x)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        # Compute and store statistics
        self.mean = x.mean(dim=-1, keepdim=True)
        self.std = torch.sqrt(x.var(dim=-1, keepdim=True, unbiased=False) + self.eps)
        
        # Normalize
        x = (x - self.mean) / self.std
        
        # Apply affine transformation
        if self.affine:
            x = x * self.weight + self.bias
        
        return x
    
    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise RuntimeError("Must call normalize before denormalize")
        
        # Reverse affine transformation
        if self.affine:
            x = (x - self.bias) / (self.weight + self.eps)
        
        # Denormalize
        x = x * self.std + self.mean
        
        return x

