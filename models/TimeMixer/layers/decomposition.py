import torch
import torch.nn as nn
import torch.nn.functional as F


class MovingAvgDecomp(nn.Module):
    """
    Moving average decomposition:
      x = season + trend
    Here trend is moving average, season is residual.
    """
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor):
        # x: [B, T, C]
        B, T, C = x.shape
        pad = (self.kernel_size - 1) // 2

        # pad on time dimension
        x_t = x.permute(0, 2, 1)  # [B, C, T]
        x_pad = F.pad(x_t, (pad, pad), mode="replicate")
        trend = F.avg_pool1d(x_pad, kernel_size=self.kernel_size, stride=1)
        trend = trend.permute(0, 2, 1)  # [B, T, C]

        season = x - trend
        return season, trend


class DFTSeriesDecomp(nn.Module):
    """
    DFT-based decomposition:
    Keep only top_k frequencies as season, remainder as trend.
    """
    def __init__(self, top_k: int = 5):
        super().__init__()
        self.top_k = top_k

    def forward(self, x: torch.Tensor):
        # x: [B, T, C]
        xf = torch.fft.rfft(x, dim=1)  # [B, F, C]
        freq = torch.abs(xf)
        freq[:, 0, :] = 0  # remove DC

        # pick top_k per (B,C) over frequency dimension
        top_k_val, _ = torch.topk(freq, k=min(self.top_k, freq.shape[1]), dim=1)
        thr = top_k_val[:, -1:, :]  # [B,1,C]
        xf = xf.masked_fill(freq < thr, 0)

        season = torch.fft.irfft(xf, n=x.shape[1], dim=1)
        trend = x - season
        return season, trend