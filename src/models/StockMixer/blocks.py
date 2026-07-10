"""
Building blocks for StockMixer, ported from
https://github.com/SJTU-DMTai/StockMixer/blob/master/src/model.py

The original implementation is unbatched and operates on ``[stocks, T, C]``
tensors directly. Everything here is kept faithful to that reference except
:class:`NoGraphMixer`, which is made batch-aware (accepts ``[B, N, D]``) so
the wrapper in :mod:`.StockMixer` can process a full mini-batch of
time-windows in one forward pass.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


ACT = nn.GELU()


class MixerBlock(nn.Module):
    """Two-layer MLP with a GELU activation (feature-independent mixer).

    Same as the original ``MixerBlock``: Linear -> GELU -> (dropout) ->
    Linear -> (dropout). Operates on the last dim of the input.
    """

    def __init__(self, mlp_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = dropout
        self.dense_1 = nn.Linear(mlp_dim, hidden_dim)
        self.act = ACT
        self.dense_2 = nn.Linear(hidden_dim, mlp_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense_1(x)
        x = self.act(x)
        if self.dropout != 0.0:
            x = F.dropout(x, p=self.dropout)
        x = self.dense_2(x)
        if self.dropout != 0.0:
            x = F.dropout(x, p=self.dropout)
        return x


class TriU(nn.Module):
    """Causal (upper-triangular) time-mixer used by the original StockMixer.

    Time step ``t`` is a learned linear combination of steps ``[0..t]`` only,
    i.e. the future never leaks into the past. Implemented as a list of
    ``nn.Linear(t+1, 1)`` layers whose outputs are concatenated along the
    time axis. Input/output shape is ``[..., channels, time_steps]``.
    """

    def __init__(self, time_step: int):
        super().__init__()
        self.time_step = time_step
        self.triU = nn.ParameterList(
            [nn.Linear(i + 1, 1) for i in range(time_step)]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.triU[0](inputs[..., 0:1])
        for i in range(1, self.time_step):
            x = torch.cat([x, self.triU[i](inputs[..., : i + 1])], dim=-1)
        return x


class Mixer2dTriU(nn.Module):
    """One time-then-channel mixing block used inside :class:`MultTime2dMixer`.

    Applied per stock. Input/output: ``[batch, time_steps, channels]``.
      1) LayerNorm over (T, C)
      2) Causal time-mixing along T (via :class:`TriU`)
      3) Residual + LayerNorm
      4) Channel-mixing MLP along C
      5) Residual
    """

    def __init__(self, time_steps: int, channels: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm([time_steps, channels])
        self.ln_2 = nn.LayerNorm([time_steps, channels])
        self.time_mixer = TriU(time_steps)
        self.channel_mixer = MixerBlock(channels, channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.ln_1(inputs)
        x = x.permute(0, 2, 1)       # [B, C, T] for TriU
        x = self.time_mixer(x)
        x = x.permute(0, 2, 1)       # back to [B, T, C]

        x = self.ln_2(x + inputs)
        y = self.channel_mixer(x)
        return x + y


class MultTime2dMixer(nn.Module):
    """Two-scale temporal mixer: original resolution + a downsampled branch.

    Takes the raw sequence ``inputs`` at length ``T`` and its downsampled
    version ``y`` at length ``scale_dim`` (produced by a stride-2 conv
    outside this module). Applies :class:`Mixer2dTriU` on each scale and
    concatenates ``[inputs, mixed_full, mixed_scale]`` along the time axis,
    giving ``[B, 2*T + scale_dim, C]``.
    """

    def __init__(self, time_step: int, channels: int, scale_dim: int):
        super().__init__()
        self.mix_layer = Mixer2dTriU(time_step, channels)
        self.scale_mix_layer = Mixer2dTriU(scale_dim, channels)

    def forward(self, inputs: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y = self.scale_mix_layer(y)
        x = self.mix_layer(inputs)
        return torch.cat([inputs, x, y], dim=1)


class NoGraphMixer(nn.Module):
    """Cross-stock mixer — low-rank learnable "market" projection.

    Takes stock representations ``[B, N_stocks, D]`` and mixes across the
    stock axis with LayerNorm(N) -> Linear(N→hidden) -> Hardswish ->
    Linear(hidden→N). No explicit graph is used — the hidden bottleneck
    forces the model to route information through a small "market" space.

    Compared to the original (which was unbatched with shape ``[N, D]``),
    this version accepts a leading batch dimension so a mini-batch of
    time-windows can be processed in one pass.
    """

    def __init__(self, num_stocks: int, hidden_dim: int = 20):
        super().__init__()
        self.dense1 = nn.Linear(num_stocks, hidden_dim)
        self.activation = nn.Hardswish()
        self.dense2 = nn.Linear(hidden_dim, num_stocks)
        self.layer_norm_stock = nn.LayerNorm(num_stocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: [B, N, D] — move stocks to the last dim so the linear
        # layers and LayerNorm operate along it, mirroring the original
        # `permute(1, 0)` on the [N, D] tensor.
        x = inputs.transpose(-1, -2)      # [B, D, N]
        x = self.layer_norm_stock(x)
        x = self.dense1(x)                # [B, D, hidden]
        x = self.activation(x)
        x = self.dense2(x)                # [B, D, N]
        return x.transpose(-1, -2)        # [B, N, D]
