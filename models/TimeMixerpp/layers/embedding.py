import torch
import torch.nn as nn

class DataEmbeddingWoPos(nn.Module):
    """
    Minimal embedding: (x + optional time features) -> d_model
    No positional embedding required for TimeMixer.
    """
    def __init__(self, c_in: int, d_model: int, dropout: float = 0.1, time_feat_dim: int = 0):
        super().__init__()
        self.time_feat_dim = time_feat_dim
        self.proj = nn.Linear(c_in + time_feat_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, x_mark: torch.Tensor = None):
        # x: [B,T,C]
        if x_mark is not None and self.time_feat_dim > 0:
            x = torch.cat([x, x_mark[..., : self.time_feat_dim]], dim=-1)
        out = self.proj(x)
        return self.dropout(out)