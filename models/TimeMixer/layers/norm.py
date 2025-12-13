import torch
import torch.nn as nn


class Normalize(nn.Module):
    """
    Simple per-sample normalization over time dimension.
    For finance series, this stabilizes scale changes.
    """
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.gamma = nn.Parameter(torch.ones(1, 1, num_features))
            self.beta = nn.Parameter(torch.zeros(1, 1, num_features))
        else:
            self.register_parameter("gamma", None)
            self.register_parameter("beta", None)

        self._cache = {}

    def forward(self, x: torch.Tensor, mode: str):
        # x: [B, T, C]
        if mode == "norm":
            mean = x.mean(dim=1, keepdim=True)   # [B,1,C]
            std = x.std(dim=1, keepdim=True)     # [B,1,C]
            std = std.clamp_min(self.eps)

            self._cache["mean"] = mean
            self._cache["std"] = std

            x_n = (x - mean) / std
            if self.affine:
                x_n = x_n * self.gamma + self.beta
            return x_n

        if mode == "denorm":
            mean = self._cache.get("mean", None)
            std = self._cache.get("std", None)
            if mean is None or std is None:
                return x

            x_d = x
            if self.affine:
                x_d = (x_d - self.beta) / (self.gamma.clamp_min(self.eps))
            x_d = x_d * std + mean
            return x_d

        raise ValueError("mode must be 'norm' or 'denorm'")