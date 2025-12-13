import torch
import torch.nn as nn

from .layers.decomposition import MovingAvgDecomp, DFTSeriesDecomp
from .layers.embedding import DataEmbeddingWoPos
from .layers.downsampling import MultiScaleDownsampler
from .layers.mixing import MultiScaleSeasonMixing, MultiScaleTrendMixing
from .layers.norm import Normalize
from .layers.head import ForecastHead


class PastDecomposableMixing(nn.Module):
    def __init__(self, seq_len, pred_len, d_model, d_ff, dropout,
                 down_window, down_layers,
                 decomp_method="moving_avg", moving_avg=25, top_k=5,
                 channel_independence=False):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.down_sampling_window = down_window
        self.channel_independence = channel_independence

        self.dropout = nn.Dropout(dropout)

        if decomp_method == "moving_avg":
            self.decomp = MovingAvgDecomp(moving_avg)
        elif decomp_method == "dft_decomp":
            self.decomp = DFTSeriesDecomp(top_k)
        else:
            raise ValueError("decomp_method must be 'moving_avg' or 'dft_decomp'")

        # optional cross-channel MLP (useful for OHLCV)
        self.cross = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

        self.mix_season = MultiScaleSeasonMixing(seq_len, down_window, down_layers)
        self.mix_trend = MultiScaleTrendMixing(seq_len, down_window, down_layers)

        self.out_mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x_list):
        # x_list: list of [B,T_i,d_model]
        length_list = [x.size(1) for x in x_list]

        season_list, trend_list = [], []
        for x in x_list:
            season, trend = self.decomp(x)   # [B,T,d_model]
            season = self.cross(season)
            trend = self.cross(trend)
            season_list.append(season.permute(0, 2, 1))  # [B,d_model,T]
            trend_list.append(trend.permute(0, 2, 1))    # [B,d_model,T]

        out_season_list = self.mix_season(season_list)  # list of [B,T_i,d_model]
        out_trend_list = self.mix_trend(trend_list)

        out_list = []
        for ori, s, t, L in zip(x_list, out_season_list, out_trend_list, length_list):
            out = s + t
            out = ori + self.out_mlp(out)   # residual
            out_list.append(out[:, :L, :])
        return out_list


class TimeMixer(nn.Module):
    """
    Minimal TimeMixer for forecasting (finance-friendly).
    Input:
      x_enc: [B, seq_len, enc_in]
      x_mark_enc: optional time features [B, seq_len, time_feat_dim]
    Output:
      y: [B, pred_len, c_out]
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
                 decomp_method: str = "moving_avg",
                 moving_avg: int = 25,
                 top_k: int = 5,
                 time_feat_dim: int = 0):
        super().__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.c_out = c_out
        self.d_model = d_model

        self.downsampler = MultiScaleDownsampler(
            enc_in=enc_in, window=down_sampling_window, layers=down_sampling_layers, method=down_sampling_method
        )

        # one normalizer per scale
        self.norms = nn.ModuleList([
            Normalize(enc_in, affine=True)
            for _ in range(down_sampling_layers + 1)
        ])

        self.embedding = DataEmbeddingWoPos(enc_in, d_model, dropout=dropout, time_feat_dim=time_feat_dim)

        self.pdm_blocks = nn.ModuleList([
            PastDecomposableMixing(
                seq_len=seq_len,
                pred_len=pred_len,
                d_model=d_model,
                d_ff=d_ff,
                dropout=dropout,
                down_window=down_sampling_window,
                down_layers=down_sampling_layers,
                decomp_method=decomp_method,
                moving_avg=moving_avg,
                top_k=top_k,
            )
            for _ in range(e_layers)
        ])

        self.head = ForecastHead(
            seq_len=seq_len,
            pred_len=pred_len,
            d_model=d_model,
            c_out=c_out,
            down_window=down_sampling_window,
            down_layers=down_sampling_layers
        )

    def forward(self, x_enc, x_mark_enc=None):
        # 1) multi-scale
        x_list, mark_list = self.downsampler(x_enc, x_mark_enc)

        # 2) norm per scale + embed
        enc_out_list = []
        for i, x in enumerate(x_list):
            x = self.norms[i](x, "norm")
            m = mark_list[i] if mark_list is not None else None
            enc_out_list.append(self.embedding(x, m))  # [B,T_i,d_model]

        # 3) Past Decomposable Mixing blocks
        for blk in self.pdm_blocks:
            enc_out_list = blk(enc_out_list)

        # 4) head -> pred
        out = self.head(enc_out_list)  # [B,pred_len,c_out]
        # denorm using scale0 stats (common practice)
        out = self.norms[0](out, "denorm")
        return out