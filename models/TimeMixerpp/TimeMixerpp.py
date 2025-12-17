import torch
import torch.nn as nn

# Import layers (Ensure these files exist in src/models/layers/)
from .layers.embedding import DataEmbeddingWoPos
from .layers.downsampling import MultiScaleDownsampler
from .layers.mixing import MultiScaleSeasonMixing, MultiScaleTrendMixing
from .layers.norm import Normalize
from .layers.head import ForecastHead
from .layers.mrti import MultiResolutionTimeImaging
from .layers.tid import TimeImageDecomposition

class PastImageDecomposableMixing(nn.Module):
    """
    TimeMixer++ Block: Uses MRTI (Imaging) + TID (2D Decomp)
    """
    def __init__(self, seq_len, pred_len, d_model, d_ff, dropout,
                 down_window, down_layers):
        super().__init__()
        self.seq_len = seq_len
        
        # 1. Image Components (The "++" Upgrade)
        self.mrti = MultiResolutionTimeImaging(seq_len)
        self.tid = TimeImageDecomposition(kernel_size=(3, 3))
        
        # 2. Standard Mixing
        self.mix_season = MultiScaleSeasonMixing(seq_len, down_window, down_layers)
        self.mix_trend = MultiScaleTrendMixing(seq_len, down_window, down_layers)

        # 3. Output Projection
        self.out_mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.cross = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x_list):
        # x_list: list of [B, T_i, d_model]
        length_list = [x.size(1) for x in x_list]
        season_list, trend_list = [], []
        
        # Step A: 2D Decomposition per scale
        for x in x_list:
            # 1D -> 2D Image
            x_img, period, pad_len = self.mrti(x)
            # 2D Decomp
            season_img, trend_img = self.tid(x_img)
            # 2D -> 1D Flatten
            season_1d = self.mrti.inverse(season_img, pad_len)
            trend_1d = self.mrti.inverse(trend_img, pad_len)
            # Projection
            season_1d = self.cross(season_1d)
            trend_1d = self.cross(trend_1d)
            # Append (Mixing expects [B, d_model, T] usually, our mixing.py expects [B, C, T] format)
            # Based on your mixing.py: expects [B, C, T]. Our data is [B, T, C=d_model].
            season_list.append(season_1d.permute(0, 2, 1))
            trend_list.append(trend_1d.permute(0, 2, 1))

        # Step B: Multi-Scale Mixing
        out_season_list = self.mix_season(season_list)
        out_trend_list = self.mix_trend(trend_list)

        # Step C: Re-composition
        out_list = []
        for ori, s, t, L in zip(x_list, out_season_list, out_trend_list, length_list):
            # Sum Season + Trend (already in [B, T, d_model] from mixing layers)
            mixed = s + t
            out = ori + self.out_mlp(mixed)
            out_list.append(out[:, :L, :])
            
        return out_list


class TimeMixerPlusPlus(nn.Module):
    """
    TimeMixer++ (ICLR 2025) Main Model Class
    """
    def __init__(self, *,
                 seq_len: int,
                 pred_len: int,
                 enc_in: int,
                 c_out: int,
                 d_model: int = 128,
                 d_ff: int = 256,
                 e_layers: int = 2,
                 dropout: float = 0.1,
                 down_sampling_window: int = 2,
                 down_sampling_layers: int = 2,
                 down_sampling_method: str = "avg",
                 time_feat_dim: int = 0):
        super().__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # 1. Multi-Scale Downsampling
        self.downsampler = MultiScaleDownsampler(
            enc_in=enc_in, window=down_sampling_window, layers=down_sampling_layers, method=down_sampling_method
        )

        # 2. Norms
        self.norms = nn.ModuleList([
            Normalize(enc_in, affine=True)
            for _ in range(down_sampling_layers + 1)
        ])

        # 3. Embedding
        self.embedding = DataEmbeddingWoPos(enc_in, d_model, dropout=dropout, time_feat_dim=time_feat_dim)

        # 4. Backbone: TimeMixer++ Image Mixing Blocks
        self.pdm_blocks = nn.ModuleList([
            PastImageDecomposableMixing(
                seq_len=seq_len,
                pred_len=pred_len,
                d_model=d_model,
                d_ff=d_ff,
                dropout=dropout,
                down_window=down_sampling_window,
                down_layers=down_sampling_layers,
            )
            for _ in range(e_layers)
        ])

        # 5. Forecast Head
        self.head = ForecastHead(
            seq_len=seq_len,
            pred_len=pred_len,
            d_model=d_model,
            c_out=c_out,
            down_window=down_sampling_window,
            down_layers=down_sampling_layers
        )

    def forward(self, x_enc, x_mark_enc=None):
        # 1) Multi-scale generation
        x_list, mark_list = self.downsampler(x_enc, x_mark_enc)

        # 2) Norm + Embed
        enc_out_list = []
        for i, x in enumerate(x_list):
            x = self.norms[i](x, "norm")
            m = mark_list[i] if mark_list is not None else None
            enc_out_list.append(self.embedding(x, m))

        # 3) TimeMixer++ Mixing
        for blk in self.pdm_blocks:
            enc_out_list = blk(enc_out_list)

        # 4) Head -> Prediction
        out = self.head(enc_out_list)
        
        # 5) Denormalize
        out = self.norms[0](out, "denorm")
        
        return out