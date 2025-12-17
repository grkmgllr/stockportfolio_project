import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers.embed import DataEmbedding
from .layers.conv_blocks import Inception_Block_V1
from .layers.utils import FFT_for_Period


class TimesBlock(nn.Module):
    """
    Core building block of TimesNet.

    What it does:
    1) Uses FFT to discover dominant periods in the time series
    2) Reshapes the series into a 2D representation based on those periods
    3) Applies 2D convolution to capture intra- and inter-period patterns
    4) Aggregates multiple periods using adaptive weights
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.top_k = configs.top_k

        # Lightweight Inception-style convolution
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff, configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model, configs.num_kernels)
        )

    def forward(self, x):
        """
        x: [Batch, Time, Channels]
        """
        B, T, C = x.shape

        # Find dominant periods using FFT
        period_list, period_weight = FFT_for_Period(x, self.top_k)

        outputs = []

        for i in range(self.top_k):
            period = period_list[i]

            # Pad sequence so it can be reshaped cleanly
            total_len = self.seq_len + self.pred_len
            if total_len % period != 0:
                padded_len = ((total_len // period) + 1) * period
                padding = torch.zeros(
                    B, padded_len - total_len, C, device=x.device
                )
                x_pad = torch.cat([x, padding], dim=1)
            else:
                x_pad = x

            # Reshape to 2D: [B, C, num_blocks, period]
            x_2d = (
                x_pad
                .reshape(B, -1, period, C)
                .permute(0, 3, 1, 2)
                .contiguous()
            )

            # Apply 2D convolution
            y = self.conv(x_2d)

            # Back to 1D time series
            y = (
                y
                .permute(0, 2, 3, 1)
                .reshape(B, -1, C)
            )

            outputs.append(y[:, :total_len])

        # Stack results from different periods
        outputs = torch.stack(outputs, dim=-1)

        # Adaptive aggregation using FFT amplitudes
        weights = F.softmax(period_weight, dim=1)
        weights = weights.unsqueeze(1).unsqueeze(1)
        outputs = torch.sum(outputs * weights, dim=-1)

        # Residual connection
        return outputs + x


class Model(nn.Module):

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        # Embedding layer (value + time features)
        self.embedding = DataEmbedding(
            configs.enc_in,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout
        )

        # Stack TimesBlocks
        self.blocks = nn.ModuleList([
            TimesBlock(configs) for _ in range(configs.e_layers)
        ])

        self.layer_norm = nn.LayerNorm(configs.d_model)

        # Projection head
        self.predict_linear = nn.Linear(
            self.seq_len,
            self.seq_len + self.pred_len
        )
        self.projection = nn.Linear(configs.d_model, configs.c_out)

    def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None, mask=None):
        """
        x_enc: [B, seq_len, enc_in]
        x_mark_enc: time features
        """

        # Instance-wise normalization (important for finance data)
        mean = x_enc.mean(dim=1, keepdim=True)
        std = torch.sqrt(x_enc.var(dim=1, keepdim=True) + 1e-5)
        x_enc = (x_enc - mean) / std

        # Embedding
        x = self.embedding(x_enc, x_mark_enc)

        # Expand temporal dimension for prediction
        x = self.predict_linear(x.permute(0, 2, 1)).permute(0, 2, 1)

        # TimesNet blocks
        for block in self.blocks:
            x = self.layer_norm(block(x))

        # Project to output dimension
        out = self.projection(x)

        # De-normalization
        out = out * std + mean

        # Return only prediction horizon
        return out[:, -self.pred_len:]