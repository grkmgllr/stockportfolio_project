"""
Patch Embedding layer for ModernTCN.

ModernTCN uses patching to group consecutive time steps together,
similar to Vision Transformers. This reduces sequence length while
capturing local temporal patterns.
"""
import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    Patchify time series and embed to model dimension.
    
    Converts input [B, T, C] to patches [B, num_patches, d_model]
    where num_patches = T // patch_size
    """
    
    def __init__(
        self,
        seq_len: int,
        enc_in: int,
        d_model: int,
        patch_size: int = 8,
        stride: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.seq_len = seq_len
        self.enc_in = enc_in
        self.d_model = d_model
        self.patch_size = patch_size
        self.stride = stride
        
        # Calculate number of patches
        self.num_patches = (seq_len - patch_size) // stride + 1
        
        # Patch embedding: project each patch to d_model
        # Input: [B, C, T] -> patches of size [B, C, num_patches, patch_size]
        # We use Conv1d for efficient patching
        self.patch_proj = nn.Conv1d(
            in_channels=enc_in,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=stride
        )
        
        # Positional embedding for patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, d_model, self.num_patches)
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize positional embedding
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, T, C] (batch, time, channels)
        
        Returns:
            Patch embeddings [B, d_model, num_patches]
        """
        B, T, C = x.shape
        
        # Transpose for Conv1d: [B, T, C] -> [B, C, T]
        x = x.transpose(1, 2)
        
        # Apply patch projection: [B, C, T] -> [B, d_model, num_patches]
        x = self.patch_proj(x)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        x = self.dropout(x)
        
        return x


class TimeFeatureEmbedding(nn.Module):
    """
    Optional time feature embedding for temporal context.
    
    Embeds time features (hour, day, month, etc.) and adds to patch embeddings.
    """
    
    def __init__(
        self,
        d_model: int,
        time_feat_dim: int = 4,
        patch_size: int = 8,
        stride: int = 8
    ):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        
        # Project time features to model dimension
        self.time_proj = nn.Linear(time_feat_dim, d_model)
    
    def forward(self, x_mark: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_mark: Time features [B, T, time_feat_dim]
        
        Returns:
            Time embeddings aligned with patches [B, d_model, num_patches]
        """
        if x_mark is None:
            return None
        
        B, T, D = x_mark.shape
        
        # Sample time features at patch centers
        # Take the middle time step of each patch
        indices = torch.arange(
            self.patch_size // 2,
            T - self.patch_size // 2 + 1,
            self.stride,
            device=x_mark.device
        )
        
        # Gather time features at patch positions
        x_mark_patches = x_mark[:, indices, :]  # [B, num_patches, time_feat_dim]
        
        # Project to model dimension: [B, num_patches, d_model]
        time_embed = self.time_proj(x_mark_patches)
        
        # Transpose for consistency: [B, d_model, num_patches]
        return time_embed.transpose(1, 2)

