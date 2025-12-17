import torch
import torch.nn as nn

class MultiScaleSeasonMixing(nn.Module):
    """
    Bottom-up mixing: high -> low for season
    """
    def __init__(self, seq_len: int, down_window: int, down_layers: int):
        super().__init__()
        self.down_sampling_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(seq_len // (down_window ** i), seq_len // (down_window ** (i + 1))),
                nn.GELU(),
                nn.Linear(seq_len // (down_window ** (i + 1)), seq_len // (down_window ** (i + 1))),
            )
            for i in range(down_layers)
        ])

    def forward(self, season_list):
        out_high = season_list[0]
        out_low = season_list[1]
        out_season_list = [out_high.permute(0, 2, 1)]

        for i in range(len(season_list) - 1):
            out_low_res = self.down_sampling_layers[i](out_high)
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 1))

        return out_season_list

class MultiScaleTrendMixing(nn.Module):
    """
    Top-down mixing: low -> high for trend
    """
    def __init__(self, seq_len: int, down_window: int, down_layers: int):
        super().__init__()
        self.up_sampling_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(seq_len // (down_window ** (i + 1)), seq_len // (down_window ** i)),
                nn.GELU(),
                nn.Linear(seq_len // (down_window ** i), seq_len // (down_window ** i)),
            )
            for i in reversed(range(down_layers))
        ])

    def forward(self, trend_list):
        rev = trend_list.copy()
        rev.reverse()

        out_low = rev[0]
        out_high = rev[1]
        out_trend_list = [out_low.permute(0, 2, 1)]

        for i in range(len(rev) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(rev) - 1:
                out_high = rev[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))

        out_trend_list.reverse()
        return out_trend_list