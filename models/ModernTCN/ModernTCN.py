"""
ModernTCN - Modern Temporal Convolutional Network

A pure convolutional model for time series forecasting that modernizes
traditional TCNs by incorporating design principles from modern ConvNets.

Key innovations:
1. Patch embedding for efficient sequence processing
2. Large-kernel depthwise separable convolutions for long-range dependencies
3. ConvNeXt-style block design with layer normalization
4. Multi-scale temporal feature extraction

Reference: ModernTCN (ICLR 2024)
"""
import torch
import torch.nn as nn

from .layers.embedding import PatchEmbedding, TimeFeatureEmbedding
from .layers.block import ModernTCNBlock, ModernTCNStage
from .layers.head import ForecastHead, LinearHead
from .layers.norm import RevIN


class ModernTCN(nn.Module):
    """
    ModernTCN main model for time series forecasting.
    
    Architecture overview:
    1. Instance normalization (RevIN)
    2. Patch embedding
    3. Stack of ModernTCN blocks
    4. Forecast head
    5. Instance denormalization
    """
    
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        c_out: int,
        d_model: int = 64,
        d_ff: int = 128,
        e_layers: int = 2,
        dropout: float = 0.1,
        patch_size: int = 8,
        stride: int = 8,
        kernel_size: int = 51,
        use_multi_scale: bool = False,
        use_revin: bool = True,
        time_feat_dim: int = 0,
        head_type: str = 'linear',
        **kwargs  # Accept extra kwargs for compatibility
    ):
        super().__init__()
        
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.c_out = c_out
        self.use_revin = use_revin
        
        # Calculate number of patches
        self.num_patches = (seq_len - patch_size) // stride + 1
        
        # 1. Reversible Instance Normalization
        if use_revin:
            self.revin = RevIN(enc_in, affine=True)
        
        # 2. Patch Embedding
        self.patch_embed = PatchEmbedding(
            seq_len=seq_len,
            enc_in=enc_in,
            d_model=d_model,
            patch_size=patch_size,
            stride=stride,
            dropout=dropout
        )
        
        # Optional time feature embedding
        self.use_time_feat = time_feat_dim > 0
        if self.use_time_feat:
            self.time_embed = TimeFeatureEmbedding(
                d_model=d_model,
                time_feat_dim=time_feat_dim,
                patch_size=patch_size,
                stride=stride
            )
        
        # 3. ModernTCN Blocks
        self.blocks = nn.ModuleList([
            ModernTCNBlock(
                d_model=d_model,
                d_ff=d_ff,
                kernel_size=kernel_size,
                dropout=dropout,
                use_multi_scale=use_multi_scale
            )
            for _ in range(e_layers)
        ])
        
        # 4. Forecast Head
        if head_type == 'linear':
            self.head = LinearHead(
                d_model=d_model,
                num_patches=self.num_patches,
                pred_len=pred_len,
                c_out=c_out,
                dropout=dropout
            )
        else:
            self.head = ForecastHead(
                d_model=d_model,
                num_patches=self.num_patches,
                pred_len=pred_len,
                c_out=c_out,
                dropout=dropout
            )
    
    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor = None,
        x_dec: torch.Tensor = None,
        x_mark_dec: torch.Tensor = None,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass for time series forecasting.
        
        Args:
            x_enc: Input sequence [B, seq_len, enc_in]
            x_mark_enc: Optional time features [B, seq_len, time_feat_dim]
            x_dec: Decoder input (unused, for interface compatibility)
            x_mark_dec: Decoder time features (unused)
            mask: Optional attention mask (unused)
        
        Returns:
            Predictions [B, pred_len, c_out]
        """
        B, T, C = x_enc.shape
        
        # 1. Instance normalization
        if self.use_revin:
            # RevIN expects [B, C, T] format
            x = x_enc.transpose(1, 2)  # [B, C, T]
            x = self.revin(x, mode='norm')
            x = x.transpose(1, 2)  # [B, T, C]
        else:
            x = x_enc
        
        # 2. Patch embedding: [B, T, C] -> [B, d_model, num_patches]
        x = self.patch_embed(x)
        
        # Add time feature embedding if available
        if self.use_time_feat and x_mark_enc is not None:
            time_emb = self.time_embed(x_mark_enc)
            x = x + time_emb
        
        # 3. ModernTCN blocks
        for block in self.blocks:
            x = block(x)
        
        # 4. Forecast head: [B, d_model, num_patches] -> [B, pred_len, c_out]
        out = self.head(x)
        
        # 5. Instance denormalization
        if self.use_revin:
            # RevIN expects [B, C, T] format
            out = out.transpose(1, 2)  # [B, c_out, pred_len]
            out = self.revin(out, mode='denorm')
            out = out.transpose(1, 2)  # [B, pred_len, c_out]
        
        return out


class Model(nn.Module):
    """
    Wrapper class for compatibility with existing training infrastructure.
    
    This provides the same interface as TimesNet and TimeMixer models.
    """
    
    def __init__(self, configs):
        super().__init__()
        
        # Extract parameters from configs object
        self.model = ModernTCN(
            seq_len=configs.seq_len,
            pred_len=configs.pred_len,
            enc_in=configs.enc_in,
            c_out=configs.c_out,
            d_model=configs.d_model,
            d_ff=configs.d_ff,
            e_layers=configs.e_layers,
            dropout=configs.dropout,
            patch_size=getattr(configs, 'patch_size', 8),
            stride=getattr(configs, 'stride', 8),
            kernel_size=getattr(configs, 'kernel_size', 51),
            use_multi_scale=getattr(configs, 'use_multi_scale', False),
            use_revin=getattr(configs, 'use_revin', True),
            time_feat_dim=getattr(configs, 'time_feat_dim', 0),
            head_type=getattr(configs, 'head_type', 'linear')
        )
        
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
    
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        """
        Forward pass with interface matching other models.
        """
        return self.model(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)

