"""
TimesNet model definition.

This module implements the TimesNet architecture and exposes:
    - TimesNetConfig: a hyperparameter container for configuring TimesNet
    - TimesNetModel: a PyTorch module supporting multiple time-series tasks

The implementation follows the common TimesNet pipeline:
    (1) (Optional) normalize inputs per sample across time for stability
    (2) embed inputs using DataEmbedding (value + positional + optional temporal)
    (3) process representations through stacked TimesBlock modules
    (4) project latent features to task-specific outputs

Supported tasks are selected via `cfg.task_name`:
    - long_term_forecast / short_term_forecast
    - imputation
    - anomaly_detection
    - classification
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as f

from ..base import ForecastModel
from .blocks import TimesBlock
from .layers import DataEmbedding


@dataclass(frozen=True)
class TimesNetConfig:
    """
    Hyperparameter container for the TimesNet model.

    This configuration collects architectural and task-related hyperparameters
    used by `TimesNetModel`. The same TimesNet backbone can be used for multiple
    time-series tasks. The task is selected via `task_name` and affects which
    heads/projections are instantiated and which forward path is used.

    Task selection
    --------------
    The model behavior is controlled by `task_name` (string):
        - "long_term_forecast" / "short_term_forecast":
            Forecast `pred_len` future steps from a historical window of length `seq_len`.
        - "imputation":
            Reconstruct missing values given an explicit mask.
        - "anomaly_detection":
            Reconstruct the input sequence to support anomaly scoring.
        - "classification":
            Produce class logits from the encoded sequence.

    Notes
    -----
    - The current dataclass definition in this file lists the core forecasting
      hyperparameters, but the model implementation also expects additional fields
      such as `task_name` and (for classification) `num_class`.
      Ensure these attributes exist on the cfg object passed to `TimesNetModel`.

    Attributes
    ----------
    task_name : str
        Task identifier controlling model behavior. Determines which prediction
        head and forward path are used. Supported tasks include:
        {"long_term_forecast", "short_term_forecast", "imputation",
         "anomaly_detection", "classification"}.
    seq_len : int
        Historical lookback length (number of past time steps).
    pred_len : int
        Forecast horizon length (number of future time steps to predict).
    enc_in : int
        Number of input features/channels in the input time series.
    c_out : int
        Number of output features/channels for reconstruction/forecast outputs.
    d_model : int
        Model embedding dimension.
    d_ff : int
        Hidden channel dimension inside the TimesNet 2D-convolution block.
    e_layers : int
        Number of stacked TimesNet blocks.
    top_k : int
        Number of dominant periods selected via FFT in each block.
    num_kernels : int
        Number of parallel convolution branches in the Inception-style block.
    embed : str
        Temporal embedding mode passed to `DataEmbedding` (e.g., "fixed", "timeF").
    freq : str
        Frequency label passed to `DataEmbedding` for time feature handling.
    dropout : float
        Dropout probability used in embeddings and classification head dropout.
    num_class : int
        Number of classes used when task_name="classification". Ignored for
        forecasting and reconstruction tasks.
    """

    task_name: str = (
        "long_term_forecast"  # {"long_term_forecast","short_term_forecast","imputation","anomaly_detection","classification"}
    )

    # Task lengths
    seq_len: int = 96
    pred_len: int = 96

    # Input / output channels
    enc_in: int = 7
    c_out: int = 7

    # Model dimensions
    d_model: int = 64
    d_ff: int = 256
    e_layers: int = 2

    # TimesNet-specific
    top_k: int = 2
    num_kernels: int = 6

    # Embedding
    embed: str = "fixed"   # "fixed" or "timeF" etc.
    freq: str = "h"

    # Regularization
    dropout: float = 0.1

    num_class: int = 2



class TimesNetModel(ForecastModel):
    """
    Forecasting-only TimesNet implementation.

    This model implements the core TimesNet idea for time-series forecasting:
    (1) Normalize the input per-sample across time to stabilize training.
    (2) Embed the input using value + positional (+ optional temporal) embeddings.
    (3) Expand the time axis from `seq_len` to `seq_len + pred_len` using a linear
        projection (time alignment).
    (4) Apply a stack of TimesNet blocks. Each block:
        - identifies dominant periods using FFT,
        - folds the sequence into a 2D representation by each period,
        - applies Inception-style Conv2D processing,
        - unfolds back to 1D and aggregates reconstructions.
    (5) Project latent features back to `c_out` channels.
    (6) Denormalize outputs back to the original data scale.
    (7) Return only the last `pred_len` time steps as the forecast horizon.

    Shape contract
    --------------
    Inputs
    - x: Tensor of shape [B, seq_len, enc_in]
        Historical window of the time-series.
    - x_mark: Optional Tensor of shape [B, seq_len, K]
        Optional time features aligned to `x`. Can be None.

    Outputs
    - y_pred: Tensor of shape [B, pred_len, c_out]
        Forecast horizon in the *original* scale (after denormalization).

    Raises
    ------
    ValueError
        If the input tensor does not have rank 3 or if its time length does not
        match cfg.seq_len passed at initialization.
    """

    def __init__(self, cfg: TimesNetConfig):
        """
        Initialize a TimesNet model and instantiate task-specific heads.

        This constructor builds:
            - a stack of `TimesBlock` modules (TimesNet backbone),
            - an input embedding layer (`DataEmbedding`),
            - a LayerNorm over the latent dimension,
            - task-dependent projection layers based on `cfg.task_name`.

        Task-dependent initialization
        -----------------------------
        - Forecasting ("long_term_forecast", "short_term_forecast"):
            Creates `align_time` to expand time length from `seq_len` to
            `seq_len + pred_len`, and `proj` to map latent features to `c_out`.
        - Imputation / anomaly_detection:
            Creates `proj` for reconstruction to `c_out`.
        - Classification:
            Creates activation + dropout and a linear classifier head projecting
            flattened features of size `d_model * seq_len` to `num_class`.

        Args:
            cfg (TimesNetConfig):
                Model hyperparameters and task selection. The implementation
                expects `cfg.task_name` to exist. For classification, it also
                expects `cfg.num_class`.

        Raises:
            ValueError:
                If `cfg.task_name` is not one of the supported task strings.
        """
        
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
        """
        Normalize each sample across the time dimension (NS-Transformer style).

        The normalization is computed per sample and per channel:
            means[b, 1, c] = average over time
            stdev[b, 1, c] = sqrt(var over time + eps)

        This stabilizes optimization by reducing scale differences across samples.

        Args:
            x (torch.Tensor):
                Input tensor of shape [B, T, C].

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - x_norm: Normalized tensor of shape [B, T, C].
                - means:  Per-sample mean of shape [B, 1, C] (detached).
                - stdev:  Per-sample standard deviation of shape [B, 1, C].

        Notes
        -----
        The mean is detached to prevent gradients flowing through normalization
        statistics, following common practice in forecasting implementations.
        """

        means = x.mean(1, keepdim=True).detach()  # [b,1,c]
        x_norm = x - means
        stdev = torch.sqrt(x_norm.var(dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_norm = x_norm / stdev
        return x_norm, means, stdev

    def _denorm_ns_transformer(
        self, y: torch.Tensor, means: torch.Tensor, stdev: torch.Tensor
    ) -> torch.Tensor:
        """
        Invert NS-Transformer normalization to restore the original scale.

        This applies:
            y_orig = y * stdev + means
        using the per-sample statistics computed by `_norm_ns_transformer`.

        Args:
            y (torch.Tensor):
                Normalized tensor of shape [B, T, C].
            means (torch.Tensor):
                Per-sample mean of shape [B, 1, C].
            stdev (torch.Tensor):
                Per-sample standard deviation of shape [B, 1, C].

        Returns:
            torch.Tensor:
                Denormalized tensor of shape [B, T, C].
        """

        return y * stdev[:, 0, :].unsqueeze(1) + means[:, 0, :].unsqueeze(1)

    # ---- tasks ----

    def forecast(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None,
        x_dec: torch.Tensor | None,
        x_mark_dec: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Forecasting forward path for long/short-term forecasting tasks.

        This method:
            (1) normalizes the encoder input,
            (2) embeds it using `DataEmbedding`,
            (3) expands the time axis from `seq_len` to `seq_len + pred_len`
                via `align_time`,
            (4) applies stacked TimesNet blocks,
            (5) projects to `c_out` channels,
            (6) denormalizes outputs back to the original scale.

        Args:
            x_enc (torch.Tensor):
                Encoder input of shape [B, seq_len, enc_in].
            x_mark_enc (torch.Tensor | None):
                Optional time features aligned to x_enc, shape [B, seq_len, K].
            x_dec (torch.Tensor | None):
                Decoder input (kept for API compatibility; not required here).
            x_mark_dec (torch.Tensor | None):
                Decoder time features (kept for API compatibility; not required here).

        Returns:
            torch.Tensor:
                Full-length prediction tensor of shape [B, seq_len + pred_len, c_out]
                in the original scale.
        """

        x_norm, means, stdev = self._norm_ns_transformer(x_enc)
        enc = self.enc_embedding(x_norm, x_mark_enc)  # [b,t,d]
        enc = self.align_time(enc.transpose(1, 2)).transpose(1, 2)  # [b,t_total,d]
        for blk in self.blocks:
            enc = self.norm(blk(enc, self.total_len))
        y = self.proj(enc)  # [b,t_total,c]
        return self._denorm_ns_transformer(y, means, stdev)

    def anomaly_detection(self, x_enc: torch.Tensor) -> torch.Tensor:
        """
        Anomaly detection forward path via sequence reconstruction.

        The model normalizes the input sequence, embeds it, processes it through
        TimesNet blocks, projects back to `c_out`, and denormalizes to the original
        scale. The reconstruction can be used to compute anomaly scores.

        Args:
            x_enc (torch.Tensor):
                Input sequence of shape [B, T, enc_in].

        Returns:
            torch.Tensor:
                Reconstructed sequence of shape [B, T, c_out] in the original scale.
        """

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
        """
        Classification forward path producing class logits.

        The model embeds the input sequence, processes it through TimesNet blocks,
        applies activation + dropout, masks padding if a padding mask is provided,
        flattens the sequence representation, and projects to class logits.

        Padding mask handling
        ---------------------
        - If x_mark_enc is None: all time steps are treated as valid.
        - If x_mark_enc has shape [B, T]: it is treated as a padding mask.
        - If x_mark_enc has shape [B, T, 1]: it is treated as a padding mask.

        Args:
            x_enc (torch.Tensor):
                Input sequence of shape [B, T, enc_in].
            x_mark_enc (torch.Tensor | None):
                Optional padding mask of shape [B, T] or [B, T, 1].

        Returns:
            torch.Tensor:
                Class logits of shape [B, num_class].

        Raises:
            ValueError:
                If x_mark_enc is provided but does not have shape [B, T] or [B, T, 1].
        """

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
        """
        Imputation forward path for missing-value reconstruction.

        This method performs masked normalization:
            - computes per-sample mean and variance using only observed entries,
            - normalizes the sequence while zeroing out missing entries,
            - runs TimesNet blocks and projects to `c_out`,
            - rescales the output back to the original scale.

        Args:
            x_enc (torch.Tensor):
                Input sequence of shape [B, T, C].
            x_mark_enc (torch.Tensor | None):
                Optional time features aligned to x_enc, shape [B, T, K].
            x_dec (torch.Tensor | None):
                Decoder input (kept for API compatibility).
            x_mark_dec (torch.Tensor | None):
                Decoder time features (kept for API compatibility).
            mask (torch.Tensor | None):
                Observation mask of shape [B, T, C] with 1 for observed entries
                and 0 for missing entries. If None, all entries are treated as observed.

        Returns:
            torch.Tensor:
                Imputed/reconstructed sequence of shape [B, T, c_out].

        Raises:
            ValueError:
                If `mask` is provided but does not match `x_enc.shape`.
        """

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
        """
        Dispatch to the task-specific forward path based on `cfg.task_name`.

        This method selects the appropriate computation graph:
            - forecasting: `forecast(...)` and return last `pred_len` steps
            - imputation: `imputation(...)`
            - anomaly_detection: `anomaly_detection(...)`
            - classification: `classification(...)`

        Args:
            x_enc (torch.Tensor):
                Primary input tensor. For forecasting, shape [B, seq_len, enc_in].
            x_mark_enc (torch.Tensor | None):
                Optional encoder time features or padding mask (task-dependent).
            x_dec (torch.Tensor | None):
                Optional decoder input (kept for API compatibility).
            x_mark_dec (torch.Tensor | None):
                Optional decoder time features (kept for API compatibility).
            mask (torch.Tensor | None):
                Optional observation mask for imputation tasks.

        Returns:
            torch.Tensor:
                Task-dependent output tensor:
                    - forecasting: [B, pred_len, c_out]
                    - imputation: [B, T, c_out]
                    - anomaly_detection: [B, T, c_out]
                    - classification: [B, num_class]

        Raises:
            RuntimeError:
                If `cfg.task_name` is not handled by the dispatcher.
        """

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
