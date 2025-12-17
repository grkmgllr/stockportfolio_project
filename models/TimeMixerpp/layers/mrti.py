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
    """
    def __init__(self, seq_len, k_periods=1):
        super().__init__()
        self.seq_len = seq_len
        self.k_periods = k_periods

    def forward(self, x):
        # x: [Batch, Length, Channel]
        B, L, C = x.shape
        
        # 1. Find Dominant Period using FFT (Frequency Domain)
        # We average over Batch and Channel to find a global stable period for the batch
        xf = torch.fft.rfft(x, dim=1)  # [B, F, C]
        freqs = xf.abs().mean(dim=-1).mean(dim=0)  # Average magnitude across channels and batch
        freqs[0] = 0  # Ignore DC component (trend)
        
        # Get top-1 dominant period
        # frequency index 'k' corresponds to period L/k
        top_k_indices = torch.topk(freqs, self.k_periods).indices
        
        # Calculate period (avoid div by zero)
        if len(top_k_indices) > 0 and top_k_indices[0].item() > 0:
            period = int(L // (top_k_indices[0].item()))
        else:
            period = 2 # Fallback
            
        period = max(period, 2)
        
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