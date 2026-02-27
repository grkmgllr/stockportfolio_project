"""
Embedding and convolutional building blocks for TimesNet.

This module provides the layers required by TimesNet-style forecasting models:
    - Fixed sinusoidal positional embeddings
    - Token/value embeddings via 1D convolution over the time axis
    - Temporal embeddings for discrete calendar fields (month/day/weekday/hour[/minute])
    - Time-feature embeddings for continuous time features ("timeF" mode)
    - DataEmbedding wrappers that combine value + position (+ optional time)
    - Inception-style 2D convolution blocks used inside TimesNet blocks

All layers operate on batched time-series tensors with shapes [B, T, C] or
intermediate embeddings of shape [B, T, D].
"""

from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn as nn
from beartype import beartype

from .types import BTC, BTD

# TODO only have the parts that actually being used by timesnet but consider this while adding new models (timemixer and itransformer)


class PositionalEmbedding(nn.Module):
    """
    Fixed sinusoidal positional embedding (non-trainable).

    This layer implements the standard Transformer sinusoidal positional encoding.
    The full table up to `max_len` is precomputed once and stored as a buffer.
    At runtime, the encoding is sliced to match the input sequence length.

    Notes
    -----
    - The returned tensor is deterministic and does not carry gradients.
    - The input tensor `x` is used only to read the time dimension T.

    Input / Output
    --------------
    Input:
        x : [B, T, C]  (only T is used)
    Returns:
        pe : [B, T, D] where D = d_model

    Raises
    ------
    ValueError
        If T exceeds the maximum length used to precompute the buffer.
    """

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
        """
        Retrieve positional encodings for the input sequence length.

        Args:
            x (BTC):
                Input tensor of shape [B, T, C]. Only the time length T is used.

        Returns:
            BTD:
                Positional embedding tensor of shape [B, T, D].

        Raises:
            ValueError:
                If T is larger than the precomputed maximum length.
        """

        t = x.size(1)
        if t > self.pe.size(1):
            raise ValueError(f"PositionalEmbedding max_len={self.pe.size(1)} < seq_len={t}")
        out: torch.Tensor = self.pe[:, :t]
        return cast(BTD, out)


class TokenEmbedding(nn.Module):
    """
    Token/value embedding via 1D convolution over the time axis.

    This layer maps raw input features (C_in) into the model embedding space (D)
    using a 1D convolution applied along time. Circular padding is used to
    reduce boundary artifacts, following common TimesNet implementations.

    Args:
        c_in (int):
            Number of input channels/features.
        d_model (int):
            Output embedding dimension.

    Input / Output
    --------------
    Input:
        x : [B, T, C_in]
    Returns:
        z : [B, T, D]
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
        """
        Embed input values into the model dimension.

        Args:
            x (torch.Tensor):
                Input tensor of shape [B, T, C_in].

        Returns:
            torch.Tensor:
                Embedded tensor of shape [B, T, D].
        """
        # [b, t, c] -> [b, c, t] -> conv -> [b, d, t] -> [b, t, d]
        return self.token_conv(x.transpose(1, 2)).transpose(1, 2)


class FixedEmbedding(nn.Module):
    """
    Fixed (non-trainable) sinusoidal embedding for discrete indices.

    This layer constructs a sinusoidal table for indices in the range
    [0, c_in - 1] and stores it as a frozen nn.Embedding weight.

    It is typically used to embed discrete calendar fields (e.g., month, weekday)
    in a deterministic manner.

    Args:
        c_in (int):
            Vocabulary size (number of discrete indices).
        d_model (int):
            Embedding dimension.

    Input / Output
    --------------
    Input:
        x : integer tensor of indices with arbitrary shape [*]
    Returns:
        emb : tensor of shape [*, D]
    """

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
        """
        Look up fixed sinusoidal embeddings for discrete indices.

        Args:
            x (torch.Tensor):
                Integer tensor of indices with shape [*].

        Returns:
            torch.Tensor:
                Embedded tensor of shape [*, D].
        """
        # x: Long indices -> [*, d_model]
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    """
    Embedding for discrete calendar/time features.

    This layer embeds discrete calendar components and sums them to obtain a
    per-timestep time embedding. It supports:
        - fixed sinusoidal embeddings ("fixed")
        - learnable embeddings (any embed_type != "fixed")

    The expected discrete fields follow the common TimesNet convention:
        x_mark[..., 0] = month
        x_mark[..., 1] = day
        x_mark[..., 2] = weekday
        x_mark[..., 3] = hour
        x_mark[..., 4] = minute  (only if freq == "t")

    Args:
        d_model (int):
            Embedding dimension.
        embed_type (str):
            "fixed" for sinusoidal embeddings, otherwise uses learnable embeddings.
        freq (str):
            Frequency string. If "t", includes minute embeddings.

    Input / Output
    --------------
    Input:
        x_mark : [B, T, K] integer calendar features
    Returns:
        t_emb : [B, T, D]
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
        """
        Compute summed calendar embeddings for each time step.

        Args:
            x_mark (torch.Tensor):
                Integer calendar feature tensor of shape [B, T, K].

        Returns:
            torch.Tensor:
                Time embedding of shape [B, T, D].
        """
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
    """
    Linear projection for continuous time features ("timeF" mode).

    Some data pipelines represent time using continuous features rather than
    discrete calendar indices. This layer projects those features to the model
    embedding dimension.

    The input feature dimension depends on `freq` and follows the common mapping
    used in TimesNet-style implementations.

    Args:
        d_model (int):
            Embedding dimension.
        embed_type (str):
            Kept for API compatibility (typically "timeF").
        freq (str):
            Frequency identifier used to select the expected input dimension.

    Input / Output
    --------------
    Input:
        x_mark : [B, T, d_in]
    Returns:
        t_emb : [B, T, D]

    Raises
    ------
    ValueError
        If `freq` is not in the supported frequency map.
    """

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
        """
        Project continuous time features to the model dimension.

        Args:
            x_mark (torch.Tensor):
                Continuous time features of shape [B, T, d_in].

        Returns:
            torch.Tensor:
                Projected features of shape [B, T, D].
        """
        return self.proj(x_mark)


class DataEmbedding(nn.Module):
    """
    Combined embedding layer used by TimesNet.

    This layer constructs the encoder input representation by summing:
        (1) token/value embedding of the raw input sequence `x`,
        (2) fixed sinusoidal positional embedding,
        (3) optional temporal embedding derived from `x_mark`.

    A dropout is applied to the summed embedding.

    Args:
        c_in (int):
            Number of input channels/features.
        d_model (int):
            Embedding dimension.
        embed (str):
            Temporal embedding mode. If "timeF", uses `TimeFeatureEmbedding`
            (continuous time features). Otherwise uses `TemporalEmbedding`
            (discrete calendar features).
        freq (str):
            Frequency identifier passed to the temporal embedding.
        dropout (float):
            Dropout probability.

    Input / Output
    --------------
    Input:
        x      : [B, T, C_in]
        x_mark : Optional [B, T, K]
    Returns:
        z      : [B, T, D]
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
        """
        Compute the combined embedding for an input sequence.

        Args:
            x (torch.Tensor):
                Input values of shape [B, T, C_in].
            x_mark (torch.Tensor | None):
                Optional time features aligned with x. If None, only value and
                positional embeddings are used.

        Returns:
            torch.Tensor:
                Embedded representation of shape [B, T, D].
        """
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
    """
    Inverted embedding variant (kept for compatibility with other repositories).

    Some implementations treat the channel dimension as the "sequence" dimension
    and embed [B, C, T] rather than [B, T, C]. This project does not require this
    layer for the default TimesNet path, but it is kept for parity and potential
    future integrations.

    Input / Output
    --------------
    Input:
        x      : [B, T, C]
        x_mark : Optional [B, T, K]
    Returns:
        z      : Tensor with embedding dimension D (shape depends on concatenation).
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
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, x_mark: torch.Tensor | None) -> torch.Tensor:
        """
        Compute inverted embeddings.

        Args:
            x (torch.Tensor):
                Input sequence of shape [B, T, C].
            x_mark (torch.Tensor | None):
                Optional time features of shape [B, T, K].

        Returns:
            torch.Tensor:
                Embedded tensor after inverting dimensions (see implementation notes).
        """
        # Original expects [b, c, t]; we accept [b, t, c]
        xv = x.transpose(1, 2)  # [b, c, t]
        if x_mark is None:
            out = self.value_embedding(xv)
        else:
            out = self.value_embedding(torch.cat([xv, x_mark.transpose(1, 2)], dim=1))
        return self.dropout(out)


class DataEmbeddingWithoutPos(nn.Module):
    """
    Embedding layer without positional encoding.

    This layer is equivalent to `DataEmbedding` but does not add positional
    encodings. It can be useful for ablation studies or models that handle
    positional information differently.

    Input / Output
    --------------
    Input:
        x      : [B, T, C_in]
        x_mark : Optional [B, T, K]
    Returns:
        z      : [B, T, D]
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
        """
        Compute embeddings without positional encodings.

        Args:
            x (torch.Tensor):
                Input values of shape [B, T, C_in].
            x_mark (torch.Tensor | None):
                Optional time features of shape [B, T, K].

        Returns:
            torch.Tensor:
                Embedded representation of shape [B, T, D].
        """
        if x_mark is None:
            out = self.value_embedding(x)
        else:
            out = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(out)


# -----------------------------
# Inception-style 2D conv block
# -----------------------------


class InceptionBlockV1(nn.Module):
    """
    Inception-style 2D convolution block (V1) used in TimesNet.

    This module builds `num_kernels` parallel Conv2D branches with odd kernel
    sizes (1, 3, 5, ...). Each branch preserves spatial resolution via symmetric
    padding. The branch outputs are aggregated by taking the mean across the
    branch dimension.

    Args:
        in_channels (int):
            Number of input channels.
        out_channels (int):
            Number of output channels per branch.
        num_kernels (int):
            Number of parallel convolution branches.
        init_weight (bool):
            If True, applies Kaiming initialization to Conv2D weights.

    Input / Output
    --------------
    Input:
        x : [B, C_in, H, W]
    Returns:
        y : [B, C_out, H, W]
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
        """
        Apply parallel Conv2D branches and aggregate their outputs by mean.

        Args:
            x (torch.Tensor):
                Input feature map of shape [B, C_in, H, W].

        Returns:
            torch.Tensor:
                Output feature map of shape [B, C_out, H, W].
        """
        # Parallel convs, then mean over the branch dimension
        res_list = [k(x) for k in self.kernels]  # list of [b, out_ch, h, w]
        res = torch.stack(res_list, dim=-1).mean(
            dim=-1
        )  # [b, out_ch, h, w, k] -> mean -> [b, out_ch, h, w]
        return res


class InceptionBlockV2(nn.Module):
    """
    Inception-style 2D convolution block (V2) with separable rectangular kernels.

    This variant constructs branches using paired rectangular convolutions:
        - (1, k) and (k, 1) for multiple k values
    plus an additional 1x1 convolution branch. The outputs are aggregated by
    averaging across branches.

    Args:
        in_channels (int):
            Number of input channels.
        out_channels (int):
            Number of output channels per branch.
        num_kernels (int):
            Controls the number of rectangular branches (pairs are built from it).
        init_weight (bool):
            If True, applies Kaiming initialization to Conv2D weights.

    Input / Output
    --------------
    Input:
        x : [B, C_in, H, W]
    Returns:
        y : [B, C_out, H, W]
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
        """
        Apply separable/rectangular Conv2D branches and aggregate outputs by mean.

        Args:
            x (torch.Tensor):
                Input feature map of shape [B, C_in, H, W].

        Returns:
            torch.Tensor:
                Output feature map of shape [B, C_out, H, W].
        """
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
