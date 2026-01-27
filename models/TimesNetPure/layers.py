from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn as nn
from beartype import beartype

from .types import BTC, BTD

# TODO only have the parts that actually being used by timesnet but consider this while adding new models (timemixer and itransformer)


class PositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding (fixed, no gradients)."""

    pe: torch.Tensor

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float)
        pe.requires_grad_(False)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    @beartype
    def forward(self, x: BTC) -> BTD:
        t = x.size(1)
        if t > self.pe.size(1):
            raise ValueError(f"PositionalEmbedding max_len={self.pe.size(1)} < seq_len={t}")
        out: torch.Tensor = self.pe[:, :t]
        return cast(BTD, out)


class TokenEmbedding(nn.Module):
    """Value embedding with 1D conv over time (circular padding), like the original.

    Input  : [b, t, c_in]
    Output : [b, t, d_model]
    """

    def __init__(self, c_in: int, d_model: int) -> None:
        super().__init__()
        # Original used padding_mode='circular' with kernel_size=3.
        self.token_conv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
            bias=False,
        )
        # Kaiming init for convs (as in original)
        nn.init.kaiming_normal_(self.token_conv.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [b, t, c] -> [b, c, t] -> conv -> [b, d, t] -> [b, t, d]
        return self.token_conv(x.transpose(1, 2)).transpose(1, 2)


class FixedEmbedding(nn.Module):
    """Fixed (sin/cos) embedding over an index space [0..c_in-1]."""

    def __init__(self, c_in: int, d_model: int) -> None:
        super().__init__()
        w = torch.zeros(c_in, d_model, dtype=torch.float)
        w.requires_grad_(False)

        position = torch.arange(0, c_in, dtype=torch.float).unsqueeze(1)  # [c_in, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model)
        )
        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: Long indices -> [*, d_model]
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    """Calendar/time embedding with (hour, weekday, day, month[, minute]) components.

    If embed_type='fixed', uses FixedEmbedding; else learnable nn.Embedding.
    freq:
      - 't' (minute) enables minute embedding
      - 'h' (hourly), 'd' (daily), etc. match the original behavior
    """

    def __init__(self, d_model: int, embed_type: str = "fixed", freq: str = "h") -> None:
        super().__init__()
        minute_size, hour_size, weekday_size, day_size, month_size = 4, 24, 7, 32, 13
        embed_cls = FixedEmbedding if embed_type == "fixed" else nn.Embedding

        if freq == "t":
            self.minute_embed = embed_cls(minute_size, d_model)
        self.hour_embed = embed_cls(hour_size, d_model)
        self.weekday_embed = embed_cls(weekday_size, d_model)
        self.day_embed = embed_cls(day_size, d_model)
        self.month_embed = embed_cls(month_size, d_model)

    def forward(self, x_mark: torch.Tensor) -> torch.Tensor:
        # x_mark: [b, t, k] integer time features in columns:
        #   [:, :, 0]=month, [:, :, 1]=day, [:, :, 2]=weekday, [:, :, 3]=hour, [:, :, 4]=minute (if present)
        x_mark = x_mark.long()
        if hasattr(self, "minute_embed"):
            minute_x = self.minute_embed(x_mark[:, :, 4])
        else:
            minute_x = torch.zeros_like(self.hour_embed(x_mark[:, :, 3]))
        hour_x = self.hour_embed(x_mark[:, :, 3])
        weekday_x = self.weekday_embed(x_mark[:, :, 2])
        day_x = self.day_embed(x_mark[:, :, 1])
        month_x = self.month_embed(x_mark[:, :, 0])
        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    """Linear projection of continuous time features (timeF mode)."""

    def __init__(self, d_model: int, embed_type: str = "timeF", freq: str = "h") -> None:
        super().__init__()
        # Matches the original freq_map
        freq_map: dict[str, int] = {
            "h": 4,
            "t": 5,
            "s": 6,
            "m": 1,
            "a": 1,
            "w": 2,
            "d": 3,
            "b": 3,
        }
        d_inp = freq_map.get(freq)
        if d_inp is None:
            raise ValueError(f"Unsupported freq='{freq}'. Expected one of {sorted(freq_map)}.")
        self.proj = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x_mark: torch.Tensor) -> torch.Tensor:
        return self.proj(x_mark)


class DataEmbedding(nn.Module):
    """Value + positional (+ optional time) embedding, mirroring the original DataEmbedding.

    If x_mark is None: value + position.
    Else: value + temporal(x_mark) + position.
    """

    def __init__(
        self,
        c_in: int,
        d_model: int,
        embed: str = "fixed",
        freq: str = "h",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        if embed == "timeF":
            self.temporal_embedding: nn.Module = TimeFeatureEmbedding(
                d_model=d_model, embed_type=embed, freq=freq
            )
        else:
            self.temporal_embedding = TemporalEmbedding(
                d_model=d_model, embed_type=embed, freq=freq
            )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, x_mark: torch.Tensor | None) -> torch.Tensor:
        # x: [b, t, c] ; returns [b, t, d_model]
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = (
                self.value_embedding(x)
                + self.temporal_embedding(x_mark)
                + self.position_embedding(x)
            )
        return self.dropout(x)


# (Optional helpers from the original file; include if you later need them)
class DataEmbeddingInverted(nn.Module):
    """Inverted variant used by some repos (not required for TimesNet, kept for parity)."""

    def __init__(
        self,
        c_in: int,
        d_model: int,
        embed: str = "fixed",
        freq: str = "h",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, x_mark: torch.Tensor | None) -> torch.Tensor:
        # Original expects [b, c, t]; we accept [b, t, c]
        xv = x.transpose(1, 2)  # [b, c, t]
        if x_mark is None:
            out = self.value_embedding(xv)
        else:
            out = self.value_embedding(torch.cat([xv, x_mark.transpose(1, 2)], dim=1))
        return self.dropout(out)


class DataEmbeddingWithoutPos(nn.Module):
    """Value (+ optional time) embedding without positional term."""

    def __init__(
        self,
        c_in: int,
        d_model: int,
        embed: str = "fixed",
        freq: str = "h",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        if embed == "timeF":
            self.temporal_embedding: nn.Module = TimeFeatureEmbedding(
                d_model=d_model, embed_type=embed, freq=freq
            )
        else:
            self.temporal_embedding = TemporalEmbedding(
                d_model=d_model, embed_type=embed, freq=freq
            )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, x_mark: torch.Tensor | None) -> torch.Tensor:
        if x_mark is None:
            out = self.value_embedding(x)
        else:
            out = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(out)


# -----------------------------
# Inception-style 2D conv block
# -----------------------------


class InceptionBlockV1(nn.Module):
    """TimesNet Inception block (V1).

    Builds num_kernels parallel Conv2d branches with odd kernel sizes
    (1, 3, 5, ...), concatenates results via a stack-and-mean reduction.

    Input : [b, in_ch, h, w]
    Output: [b, out_ch, h, w]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_kernels: int = 6,
        init_weight: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels

        kernels: list[nn.Conv2d] = []
        for i in range(self.num_kernels):
            # kernel size = 2*i + 1  (1,3,5,...) with symmetric padding
            ksize = 2 * i + 1
            kernels.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=ksize, padding=i, bias=True)
            )
        self.kernels = nn.ModuleList(kernels)

        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Parallel convs, then mean over the branch dimension
        res_list = [k(x) for k in self.kernels]  # list of [b, out_ch, h, w]
        res = torch.stack(res_list, dim=-1).mean(
            dim=-1
        )  # [b, out_ch, h, w, k] -> mean -> [b, out_ch, h, w]
        return res


class InceptionBlockV2(nn.Module):
    """TimesNet Inception block (V2).

    Uses separable rectangular kernels: [1, k] and [k, 1] pairs, plus a 1x1 conv.
    Total branches = (num_kernels // 2) * 2 + 1.

    Input : [b, in_ch, h, w]
    Output: [b, out_ch, h, w]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_kernels: int = 6,
        init_weight: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels

        kernels: list[nn.Conv2d] = []
        # paired [1, k] and [k, 1] convs
        for i in range(self.num_kernels // 2):
            k = 2 * i + 3
            kernels.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=(1, k),
                    padding=(0, i + 1),
                    bias=True,
                )
            )
            kernels.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=(k, 1),
                    padding=(i + 1, 0),
                    bias=True,
                )
            )
        # final 1x1 conv
        kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True))
        self.kernels = nn.ModuleList(kernels)

        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res_list = [k(x) for k in self.kernels]
        res = torch.stack(res_list, dim=-1).mean(dim=-1)
        return res


__all__ = [
    "PositionalEmbedding",
    "TokenEmbedding",
    "FixedEmbedding",
    "TemporalEmbedding",
    "TimeFeatureEmbedding",
    "DataEmbedding",
    "DataEmbeddingWithoutPos",
    "InceptionBlockV1",
    "InceptionBlockV2",
]
