import torch
import torch.nn as nn

class Normalize(nn.Module):
    """
    Simple per-sample normalization.
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
        if mode == "norm":
            mean = x.mean(dim=1, keepdim=True)
            std = x.std(dim=1, keepdim=True).clamp_min(self.eps)
            self._cache["mean"], self._cache["std"] = mean, std
            x_n = (x - mean) / std
            if self.affine: x_n = x_n * self.gamma + self.beta
            return x_n

        if mode == "denorm":
            mean, std = self._cache.get("mean"), self._cache.get("std")
            if mean is None: return x
            x_d = x
            if self.affine: x_d = (x_d - self.beta) / (self.gamma.clamp_min(self.eps))
            return x_d * std + mean