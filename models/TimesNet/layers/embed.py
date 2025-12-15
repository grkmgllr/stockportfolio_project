import torch
import torch.nn as nn


class DataEmbedding(nn.Module):
    """
    Combines:
    - value embedding
    - time feature embedding
    """

    def __init__(self, enc_in, d_model, embed_type, freq, dropout):
        super().__init__()
        self.value_embedding = nn.Linear(enc_in, d_model)
        self.time_embedding = nn.Linear(4, d_model)  # month, day, weekday, hour
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mark):
        """
        x: [B, T, enc_in]
        x_mark: [B, T, time_features]
        """
        if x_mark is None:
            out = self.value_embedding(x)
        else:
            out = self.value_embedding(x) + self.time_embedding(x_mark)
        return self.dropout(out)