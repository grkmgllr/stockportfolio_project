import torch
import torch.nn as nn
import torch.fft
import warnings

# Suppress PyTorch internal FFT deprecation warning on MPS
warnings.filterwarnings("ignore", message="An output with one or more elements was resized")

class MultiResolutionTimeImaging(nn.Module):
    """
    MRTI: Transforms 1D Time Series into 2D Time Images.
    It dynamically finds the dominant period using FFT and folds the series.
    
    Improved version with better period detection and fallback handling.
    """
    def __init__(self, seq_len, k_periods=3, min_period=4, max_period=None):
        super().__init__()
        self.seq_len = seq_len
        self.k_periods = k_periods
        self.min_period = min_period
        self.max_period = max_period if max_period else seq_len // 2

    def forward(self, x):
        # x: [Batch, Length, Channel]
        B, L, C = x.shape
        
        # 1. Find Dominant Period using FFT (Frequency Domain)
        # We average over Batch and Channel to find a global stable period for the batch
        xf = torch.fft.rfft(x, dim=1)  # [B, F, C]
        freqs = xf.abs().mean(dim=-1).mean(dim=0)  # Average magnitude across channels and batch
        freqs[0] = 0  # Ignore DC component (trend)
        
        # Get top-k dominant frequencies
        k = min(self.k_periods, len(freqs) - 1)
        top_k_indices = torch.topk(freqs, k).indices
        
        # Calculate period from best valid frequency
        period = None
        for idx in top_k_indices:
            freq_idx = idx.item()
            if freq_idx > 0:
                candidate_period = int(L / freq_idx)
                if self.min_period <= candidate_period <= self.max_period:
                    period = candidate_period
                    break
        
        # Fallback: use a reasonable default based on sequence length
        if period is None:
            # Common periods: hourly data often has daily (24) or weekly (168) patterns
            # For ETTh1 with seq_len=96: try 24 (daily pattern) or 12 (half-day)
            for fallback in [24, 12, 8, 6, 4]:
                if fallback <= L // 2:
                    period = fallback
                    break
            else:
                period = max(self.min_period, L // 8)
        
        period = max(period, self.min_period)
        
        # 2. Reshape to 2D Image
        # We need shape [B, Num_Periods, Period, C]
        num_periods = (L + period - 1) // period
        pad_len = num_periods * period - L
        
        if pad_len > 0:
            x_pad = torch.nn.functional.pad(x, (0, 0, 0, pad_len)) # Pad Time dim
        else:
            x_pad = x
            
        # Reshape: [B, H(Num_Periods), W(Period), C] -> Permute to [B, C, H, W] for Conv2d
        x_2d = x_pad.view(B, num_periods, period, C).permute(0, 3, 1, 2)
        
        return x_2d, period, pad_len
    
    def inverse(self, x_2d, pad_len):
        # Flattens back to 1D
        # x_2d: [B, C, H, W]
        B, C, H, W = x_2d.shape
        
        # Permute back to [B, H, W, C] -> Flatten H*W -> [B, L_pad, C]
        x_flat = x_2d.permute(0, 2, 3, 1).reshape(B, H * W, C)
        
        if pad_len > 0:
            x_flat = x_flat[:, :-pad_len, :]
            
        return x_flat