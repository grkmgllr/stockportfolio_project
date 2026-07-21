"""
Cross-stock trainer and evaluator for StockMixer.

Unlike :mod:`.pytorch_runner`, which loops over per-ticker
:class:`StockDataset` samples, this module drives a single
:class:`CrossStockDataset` where every sample already contains all tickers
jointly. The trained model sees all stocks in one forward pass, which is
what the ``NoGraphMixer`` cross-stock branch needs.

Loss follows the reference StockMixer implementation:
    total = MSE(pred, target) + alpha * rank_loss
where ``rank_loss`` is the mean ReLU of pairwise-return-difference products
across the stock axis. Because our targets already are returns (not raw
future prices), we skip the reference's price-to-return conversion.
"""
from __future__ import annotations

import os
import random
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def _set_seed(seed: int) -> None:
    """Fix all sources of stochasticity so runs are bit-for-bit reproducible.

    Covers Python's random module, NumPy, PyTorch (CPU + CUDA + MPS), and the
    per-worker DataLoader seed. cuDNN determinism is enabled at a small
    throughput cost — worth it for ablation runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

from dataset_crossstock import CrossStockDataset
from models.registry import get_model, get_model_config
from paths import CHECKPOINTS_ROOT
from utils import EarlyStopping, calculate_metrics, calculate_return_metrics, \
    get_scheduler, load_checkpoint

from forecasting.config import TrainingConfig
from reporting import print_epoch, print_training_config


# ─────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────
def crossstock_loss(
    pred: torch.Tensor, target: torch.Tensor, alpha: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Regression + pairwise-rank loss from the StockMixer paper.

    Both tensors are returns of shape ``[B, N, H, C]``. The rank term
    penalises pairs of stocks whose predicted return-difference has the
    opposite sign of the ground-truth return-difference, computed
    independently per (batch, horizon, channel) triple and averaged.
    Uses ``ReLU(p_diff * (-t_diff))`` so the loss activates exactly when
    the predicted ordering disagrees with the true ordering.

    Returns (total_loss, reg_loss, rank_loss).
    """
    reg = torch.nn.functional.mse_loss(pred, target)

    # Pairwise differences along the stock axis (dim=1).
    #   p_diff[..., i, j, ...] = pred_i - pred_j
    #   t_diff[..., i, j, ...] = target_i - target_j
    p_diff = pred.unsqueeze(2) - pred.unsqueeze(1)         # [B, N, N, H, C]
    t_diff = target.unsqueeze(2) - target.unsqueeze(1)     # [B, N, N, H, C]
    # Rank loss activates when signs disagree, i.e. p_diff * t_diff < 0.
    rank = torch.nn.functional.relu(-p_diff * t_diff).mean()

    return reg + alpha * rank, reg.detach(), rank.detach()


# ─────────────────────────────────────────────────────────────────────
# Per-epoch loops
# ─────────────────────────────────────────────────────────────────────
def _train_epoch(model, loader, optimizer, device, alpha, grad_clip):
    model.train()
    losses, regs, ranks = [], [], []
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)             # StockMixer: single-arg forward
        loss, reg, rank = crossstock_loss(outputs, batch_y, alpha)

        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        losses.append(loss.item())
        regs.append(reg.item())
        ranks.append(rank.item())
    return float(np.mean(losses)), float(np.mean(regs)), float(np.mean(ranks))


def _validate_epoch(model, loader, device, alpha):
    model.eval()
    losses, regs, ranks = [], [], []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            outputs = model(batch_x)
            loss, reg, rank = crossstock_loss(outputs, batch_y, alpha)
            losses.append(loss.item())
            regs.append(reg.item())
            ranks.append(rank.item())
    return float(np.mean(losses)), float(np.mean(regs)), float(np.mean(ranks))


# ─────────────────────────────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────────────────────────────
def train(tickers: List[str], model_name: str, ma_targets: List[str],
          *, seq_len: int, pred_len: int,
          epochs: int, batch_size: int, lr: float,
          patience: int, data_root: str,
          alpha: float = 0.1,
          market_dim: int = 2,
          seed: int = 42,
          start_date: str | None = "2022-01-01",
          device: str | None = None,
          checkpoint_dir: str = CHECKPOINTS_ROOT) -> str:
    """Train a cross-stock model on a joint CrossStockDataset.

    Only StockMixer is a cross-stock model at the moment; the checkpoint
    is stored under ``crossstock_{model_name}_best.pt``. Returns the
    checkpoint path.
    """
    if len(tickers) < 2:
        raise ValueError(
            f"Cross-stock training requires at least 2 tickers, got {tickers}."
        )

    _set_seed(seed)

    train_cfg = TrainingConfig(
        model_name=model_name,
        ticker="crossstock",
        data_root=data_root,
        seq_len=seq_len,
        pred_len=pred_len,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=lr,
        patience=patience,
        checkpoint_dir=checkpoint_dir,
    )
    if device:
        train_cfg.device = device
    os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)

    print("Loading Data...")
    train_ds = CrossStockDataset(
        tickers=tickers, root_path=data_root, flag="train",
        seq_len=seq_len, pred_len=pred_len,
        ma_targets=ma_targets, return_targets=True, start_date=start_date,
    )
    val_ds = CrossStockDataset(
        tickers=tickers, root_path=data_root, flag="val",
        seq_len=seq_len, pred_len=pred_len,
        ma_targets=ma_targets, return_targets=True, start_date=start_date,
    )

    model_cfg = get_model_config(
        model_name, seq_len, pred_len,
        enc_in=train_ds.enc_in,
        c_out=train_ds.c_out,
        denorm_indices=train_ds.denorm_indices,
        return_targets=True,
        num_stocks=train_ds.num_stocks,
        market_dim=market_dim,
    )
    print_training_config(train_cfg, model_cfg, train_ds.target_features, tickers)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, drop_last=False)
    print(f"\nTrain windows: {len(train_ds)}")
    print(f"Val   windows: {len(val_ds)}")
    print(f"Stocks per sample: {train_ds.num_stocks}  ({train_ds.tickers})")

    model = get_model(model_name, model_cfg).to(train_cfg.device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Rank-loss alpha: {alpha}   |   market_dim: {market_dim}   |   seed: {seed}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=train_cfg.weight_decay,
    )
    scheduler = get_scheduler(
        optimizer,
        scheduler_type=train_cfg.scheduler,
        epochs=epochs,
        step_size=train_cfg.scheduler_step_size,
        gamma=train_cfg.scheduler_gamma,
    )

    checkpoint_path = os.path.join(
        train_cfg.checkpoint_dir, f"crossstock_{model_name}_best.pt",
    )
    early = EarlyStopping(patience=patience, checkpoint_path=checkpoint_path, verbose=True)

    print("\nStarting Training...")
    print("-" * 60)
    for epoch in range(epochs):
        t0 = time.time()
        tr_loss, tr_reg, tr_rank = _train_epoch(
            model, train_loader, optimizer,
            train_cfg.device, alpha, train_cfg.grad_clip,
        )
        val_loss, val_reg, val_rank = _validate_epoch(
            model, val_loader, train_cfg.device, alpha,
        )

        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = lr

        print_epoch(epoch + 1, epochs, tr_loss, val_loss, current_lr, time.time() - t0)
        print(f"    Train: reg={tr_reg:.4e}  rank={tr_rank:.4e}   "
              f"Val: reg={val_reg:.4e}  rank={val_rank:.4e}")

        if early(val_loss, model, epoch + 1):
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    print("-" * 60)
    print(f"Training complete! Best epoch: {early.best_epoch}")
    print(f"Best validation loss: {early.best_loss:.6f}")
    print(f"Checkpoint: {checkpoint_path}")
    return checkpoint_path


# ─────────────────────────────────────────────────────────────────────
# Evaluate
# ─────────────────────────────────────────────────────────────────────
def evaluate(tickers: List[str], model_name: str, ma_targets: List[str],
             *, seq_len: int, pred_len: int,
             batch_size: int, data_root: str,
             market_dim: int = 2,
             checkpoint_override: str | None = None,
             start_date: str | None = "2022-01-01",
             device: str | None = None,
             checkpoint_dir: str = CHECKPOINTS_ROOT,
             ) -> Dict[str, Tuple[np.ndarray, np.ndarray, List[str], dict]]:
    """Run a trained cross-stock model on the joint test split.

    Returns a dict ``{ticker: (preds, trues, target_names, eval_results)}``
    where ``preds`` and ``trues`` are per-ticker arrays back on the
    dollar-price scale, shape ``[n_windows, pred_len, c_out]`` — matching
    what :func:`forecasting.pytorch_runner.evaluate` returns per call, so
    downstream code (main.cmd_test) can consume both identically.
    """
    test_ds = CrossStockDataset(
        tickers=tickers, root_path=data_root, flag="test",
        seq_len=seq_len, pred_len=pred_len,
        ma_targets=ma_targets, return_targets=True, start_date=start_date,
    )
    loader = DataLoader(test_ds, batch_size=batch_size,
                        shuffle=False, drop_last=False)

    model_cfg = get_model_config(
        model_name, seq_len, pred_len,
        enc_in=test_ds.enc_in, c_out=test_ds.c_out,
        denorm_indices=test_ds.denorm_indices, return_targets=True,
        num_stocks=test_ds.num_stocks,
        market_dim=market_dim,
    )
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model(model_name, model_cfg).to(device)

    ckpt = checkpoint_override or os.path.join(
        checkpoint_dir, f"crossstock_{model_name}_best.pt",
    )
    load_checkpoint(model, ckpt, device)

    all_preds, all_trues, losses = [], [], []
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            outputs = model(batch_x)
            loss, _, _ = crossstock_loss(outputs, batch_y, alpha=0.0)  # reg only for logging
            losses.append(loss.item())
            all_preds.append(outputs.cpu().numpy())
            all_trues.append(batch_y.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)   # [n_windows, N, H, C]
    all_trues = np.concatenate(all_trues, axis=0)
    test_loss = float(np.mean(losses))

    anchors = test_ds.get_anchors()                 # [n_windows, N]
    target_names = test_ds.target_features

    # Split per ticker and convert returns back to prices.
    out: Dict[str, Tuple[np.ndarray, np.ndarray, List[str], dict]] = {}
    for i, ticker in enumerate(test_ds.tickers):
        preds_ret = all_preds[:, i, :, :]           # [n_windows, H, C]
        trues_ret = all_trues[:, i, :, :]
        a = anchors[:, i][:, None, None]            # [n_windows, 1, 1]

        preds_price = a * (1.0 + preds_ret)
        trues_price = a * (1.0 + trues_ret)

        results = {
            "overall": calculate_metrics(preds_price, trues_price),
            "test_loss": test_loss,
            "overall_returns": calculate_return_metrics(preds_ret, trues_ret),
        }
        for j, name in enumerate(target_names):
            results[name] = calculate_metrics(
                preds_price[:, :, j], trues_price[:, :, j],
            )
            results[f"{name}_returns"] = calculate_return_metrics(
                preds_ret[:, :, j], trues_ret[:, :, j],
            )

        out[ticker] = (preds_price, trues_price, target_names, results)

    return out
