from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as f

from ..base import ForecastModel
from .blocks import TimesBlock
from .layers import DataEmbedding


@dataclass
class TimesNetConfig:
    """Minimal config for this TimesNet port."""

    task_name: str = (
        "long_term_forecast"  # {"long_term_forecast","short_term_forecast","imputation","anomaly_detection","classification"}
    )
    seq_len: int = 96
    label_len: int = 0
    pred_len: int = 96
    enc_in: int = 1
    c_out: int = 1
    d_model: int = 64
    d_ff: int = 256
    e_layers: int = 3
    top_k: int = 2
    num_kernels: int = 6
    embed: str = "fixed"
    freq: str = "h"
    dropout: float = 0.1
    num_class: int = 2


class TimesNetModel(ForecastModel):
    """TimesNet wrapper with unified tasks; inputs/outputs use [b, t, c]."""

    def __init__(self, cfg: TimesNetConfig):
        super().__init__()
        self.cfg = cfg
        self.seq_len = cfg.seq_len
        self.pred_len = cfg.pred_len
        self.total_len = cfg.seq_len + cfg.pred_len

        self.blocks = nn.ModuleList(
            [
                TimesBlock(
                    d_model=cfg.d_model,
                    d_ff=cfg.d_ff,
                    k=cfg.top_k,
                    num_kernels=cfg.num_kernels,
                )
                for _ in range(cfg.e_layers)
            ]
        )
        self.enc_embedding = DataEmbedding(
            cfg.enc_in, cfg.d_model, cfg.embed, cfg.freq, cfg.dropout
        )
        self.norm = nn.LayerNorm(cfg.d_model)

        if cfg.task_name in {"long_term_forecast", "short_term_forecast"}:
            self.align_time = nn.Linear(cfg.seq_len, self.total_len)
            self.proj = nn.Linear(cfg.d_model, cfg.c_out, bias=True)
        elif cfg.task_name in {"imputation", "anomaly_detection"}:
            self.proj = nn.Linear(cfg.d_model, cfg.c_out, bias=True)
        elif cfg.task_name == "classification":
            self.act = f.gelu
            self.drop = nn.Dropout(cfg.dropout)
            self.proj = nn.Linear(cfg.d_model * cfg.seq_len, cfg.num_class)
        else:
            raise ValueError(f"Unknown task_name={cfg.task_name}")

    # ---- helpers ----

    def _norm_ns_transformer(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-batch normalization as used in the original implementation."""
        means = x.mean(1, keepdim=True).detach()  # [b,1,c]
        x_norm = x - means
        stdev = torch.sqrt(x_norm.var(dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_norm = x_norm / stdev
        return x_norm, means, stdev

    def _denorm_ns_transformer(
        self, y: torch.Tensor, means: torch.Tensor, stdev: torch.Tensor
    ) -> torch.Tensor:
        return y * stdev[:, 0, :].unsqueeze(1) + means[:, 0, :].unsqueeze(1)

    # ---- tasks ----

    def forecast(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None,
        x_dec: torch.Tensor | None,
        x_mark_dec: torch.Tensor | None,
    ) -> torch.Tensor:
        x_norm, means, stdev = self._norm_ns_transformer(x_enc)
        enc = self.enc_embedding(x_norm, x_mark_enc)  # [b,t,d]
        enc = self.align_time(enc.transpose(1, 2)).transpose(1, 2)  # [b,t_total,d]
        for blk in self.blocks:
            enc = self.norm(blk(enc, self.total_len))
        y = self.proj(enc)  # [b,t_total,c]
        return self._denorm_ns_transformer(y, means, stdev)

    def anomaly_detection(self, x_enc: torch.Tensor) -> torch.Tensor:
        x_norm, means, stdev = self._norm_ns_transformer(x_enc)
        enc = self.enc_embedding(x_norm, None)
        for blk in self.blocks:
            enc = self.norm(blk(enc, self.total_len))
        y = self.proj(enc)
        return self._denorm_ns_transformer(y, means, stdev)

    def classification(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None,
    ) -> torch.Tensor:
        enc = self.enc_embedding(x_enc, None)
        for blk in self.blocks:
            enc = self.norm(blk(enc, self.total_len))
        out = self.drop(self.act(enc))  # [b,t,d]

        # padding mask: [b,t] or [b,t,1] → [b,t,1]; default all-ones
        if x_mark_enc is None:
            pad_mask = torch.ones(out.size(0), out.size(1), 1, device=out.device, dtype=out.dtype)
        else:
            if x_mark_enc.dim() == 2:
                pad_mask = x_mark_enc.unsqueeze(-1).to(dtype=out.dtype)
            elif x_mark_enc.dim() == 3 and x_mark_enc.size(-1) == 1:
                pad_mask = x_mark_enc.to(dtype=out.dtype)
            else:
                raise ValueError(
                    f"Expected x_mark_enc shape [b,t] or [b,t,1], got {tuple(x_mark_enc.shape)}"
                )

        out = (out * pad_mask).reshape(out.size(0), -1)  # [b,t*d]
        return self.proj(out)  # [b,num_class]

    def imputation(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None,
        x_dec: torch.Tensor | None,
        x_mark_dec: torch.Tensor | None,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if mask is None:
            mask = torch.ones_like(x_enc)  # [b,t,c]
        elif mask.shape != x_enc.shape:
            raise ValueError(f"Expected mask shape {tuple(x_enc.shape)}, got {tuple(mask.shape)}")

        means = (x_enc.sum(dim=1) / (mask == 1).sum(dim=1)).unsqueeze(1).detach()  # [b,1,c]
        x0 = (x_enc - means).masked_fill(mask == 0, 0.0)
        stdev = (
            torch.sqrt((x0 * x0).sum(dim=1) / (mask == 1).sum(dim=1) + 1e-5).unsqueeze(1).detach()
        )
        x = x0 / stdev

        enc = self.enc_embedding(x, x_mark_enc)
        for blk in self.blocks:
            enc = self.norm(blk(enc, self.total_len))
        y = self.proj(enc)
        return y * stdev[:, 0, :].unsqueeze(1) + means[:, 0, :].unsqueeze(1)

    # ---- required by ForecastModel ----
    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None = None,
        x_dec: torch.Tensor | None = None,
        x_mark_dec: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        t = self.cfg.task_name
        if t in {"long_term_forecast", "short_term_forecast"}:
            y = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return y[:, -self.cfg.pred_len :, :]  # [b,pred_len,c]
        if t == "imputation":
            return self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        if t == "anomaly_detection":
            return self.anomaly_detection(x_enc)
        if t == "classification":
            return self.classification(x_enc, x_mark_enc)
        raise RuntimeError(f"Unhandled task_name={t}")
