"""
Train StockMixer on the canonical NDX-100 panel.

Compact standalone trainer — bypasses the legacy `crossstock_runner`
so we can iterate on the new pipeline without touching the 5-ticker
baseline. Uses:

    Dataset : data.dataset.CrossStockPanelDataset  (view='cs')
    Model   : models.StockMixer.StockMixer
    Loss    : masked MSE + alpha * masked pairwise rank loss
    Optim   : Adam + Cosine LR
    Stop    : val_loss patience

Outputs
-------
    checkpoints/canonical/stockmixer_best.pt
    checkpoints/canonical/stockmixer_history.json
    checkpoints/canonical/stockmixer_metrics.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.dataset import CrossStockPanelDataset
from models.StockMixer.StockMixer import StockMixer, StockMixerConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cpu"


def masked_loss(pred: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor, alpha: float):
    """MSE + alpha * pairwise rank loss, both masked over unavailable cells.

    pred, target: [B, N, H, 1]
    mask:         [B, N, H]        (True where target is trustworthy)
    """
    m = mask.unsqueeze(-1).float()
    denom = m.sum().clamp(min=1)
    reg = ((pred - target).pow(2) * m).sum() / denom

    # Rank loss: pairwise stock differences, only for pairs where BOTH
    # stocks are valid at that (batch, horizon, channel) triple.
    p_diff = pred.unsqueeze(2) - pred.unsqueeze(1)              # [B, N, N, H, 1]
    t_diff = target.unsqueeze(2) - target.unsqueeze(1)
    pair_mask = (mask.unsqueeze(2) & mask.unsqueeze(1)).unsqueeze(-1).float()
    rank_terms = torch.nn.functional.relu(-p_diff * t_diff) * pair_mask
    rank = rank_terms.sum() / pair_mask.sum().clamp(min=1)
    return reg + alpha * rank, reg.detach(), rank.detach()


def epoch_pass(model, loader, device, alpha, optimizer=None, grad_clip=1.0):
    train = optimizer is not None
    model.train(mode=train)
    losses, regs, ranks = [], [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y, _mx, my in loader:
            x = x.to(device); y = y.to(device); my = my.to(device)
            if train:
                optimizer.zero_grad()
            pred = model(x)                    # [B, N, H, 1]
            target = y.unsqueeze(-1)           # [B, N, H, 1]
            loss, reg, rank = masked_loss(pred, target, my, alpha)
            if train:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            losses.append(loss.item())
            regs.append(reg.item())
            ranks.append(rank.item())
    return float(np.mean(losses)), float(np.mean(regs)), float(np.mean(ranks))


def rank_ic(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    """Mean per-date rank correlation between pred and target across tickers.

    pred, target: [n_samples, N, H, 1]  →  averaged over first horizon step
    mask:         [n_samples, N, H]
    """
    from scipy.stats import spearmanr
    p = pred[:, :, 0, 0]
    t = target[:, :, 0, 0]
    m = mask[:, :, 0]
    ics = []
    for i in range(p.shape[0]):
        idx = np.where(m[i])[0]
        if idx.size < 5:
            continue
        rho, _ = spearmanr(p[i, idx], t[i, idx])
        if np.isfinite(rho):
            ics.append(rho)
    return float(np.mean(ics)) if ics else float("nan")


def evaluate_test(model, loader, device):
    model.eval()
    preds, targets, masks = [], [], []
    with torch.no_grad():
        for x, y, _mx, my in loader:
            x = x.to(device); y = y.to(device); my = my.to(device)
            pred = model(x)
            preds.append(pred.cpu().numpy())
            targets.append(y.unsqueeze(-1).cpu().numpy())
            masks.append(my.cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)
    masks = np.concatenate(masks, axis=0)

    # Overall masked MSE
    m = masks[..., None].astype(np.float32)
    mse = float((((preds - targets) ** 2) * m).sum() / max(m.sum(), 1))
    ric = rank_ic(preds, targets, masks)
    # Directional accuracy (sign match at horizon step 0)
    p0 = preds[:, :, 0, 0]; t0 = targets[:, :, 0, 0]; m0 = masks[:, :, 0]
    da = float(((np.sign(p0) == np.sign(t0)) & m0).sum() / max(m0.sum(), 1))
    return {"test_mse": mse, "rank_ic_h1": ric, "directional_accuracy_h1": da}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--panel", default="data/processed/views_panel.parquet")
    p.add_argument("--seq-len", type=int, default=30)
    p.add_argument("--pred-len", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--market-dim", type=int, default=2)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--ckpt-dir", default="checkpoints/canonical")
    p.add_argument("--tag", default="stockmixer")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    device = pick_device()
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(0)}")
    elif device == "mps":
        print("GPU:    Apple Silicon (MPS)")

    train_ds = CrossStockPanelDataset(
        panel_path=args.panel, split="train",
        seq_len=args.seq_len, pred_len=args.pred_len, view="cs",
    )
    val_ds = CrossStockPanelDataset(
        panel_path=args.panel, split="val",
        seq_len=args.seq_len, pred_len=args.pred_len, view="cs",
    )
    test_ds = CrossStockPanelDataset(
        panel_path=args.panel, split="test",
        seq_len=args.seq_len, pred_len=args.pred_len, view="cs",
    )
    print(f"Windows — train: {len(train_ds)}  val: {len(val_ds)}  test: {len(test_ds)}")
    print(f"Stocks: {len(train_ds.tickers)}   Features: {train_ds.features}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, drop_last=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers,
                             pin_memory=True)

    cfg = StockMixerConfig(
        seq_len=args.seq_len, pred_len=args.pred_len,
        enc_in=len(train_ds.features), c_out=1,
        num_stocks=len(train_ds.tickers),
        market_dim=args.market_dim,
    )
    model = StockMixer(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: StockMixer  |  params: {n_params:,}")
    print(f"Hyperparams: lr={args.lr} batch={args.batch_size} "
          f"alpha={args.alpha} market_dim={args.market_dim} "
          f"patience={args.patience} seed={args.seed}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs,
    )

    ckpt_path = Path(args.ckpt_dir) / f"{args.tag}_best.pt"
    hist_path = Path(args.ckpt_dir) / f"{args.tag}_history.json"
    metrics_path = Path(args.ckpt_dir) / f"{args.tag}_metrics.json"

    history = {"train_loss": [], "val_loss": [], "train_reg": [], "val_reg": [],
               "train_rank": [], "val_rank": [], "lr": [], "epoch_time_s": []}
    best_val = float("inf"); best_epoch = -1; patience_left = args.patience
    t_all = time.time()

    print("\n" + "=" * 70)
    print("Training …")
    print("=" * 70)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_reg, tr_rank = epoch_pass(
            model, train_loader, device, args.alpha,
            optimizer=optimizer, grad_clip=args.grad_clip,
        )
        va_loss, va_reg, va_rank = epoch_pass(
            model, val_loader, device, args.alpha,
        )
        scheduler.step()
        cur_lr = scheduler.get_last_lr()[0]
        dt = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_reg"].append(tr_reg)
        history["val_reg"].append(va_reg)
        history["train_rank"].append(tr_rank)
        history["val_rank"].append(va_rank)
        history["lr"].append(cur_lr)
        history["epoch_time_s"].append(dt)

        marker = ""
        if va_loss < best_val - 1e-6:
            best_val = va_loss; best_epoch = epoch; patience_left = args.patience
            torch.save(model.state_dict(), ckpt_path)
            marker = " ← best"
        else:
            patience_left -= 1

        print(f"epoch {epoch:3d}/{args.epochs}  "
              f"train={tr_loss:.6f} (reg={tr_reg:.4e} rank={tr_rank:.4e})  "
              f"val={va_loss:.6f} (reg={va_reg:.4e} rank={va_rank:.4e})  "
              f"lr={cur_lr:.2e}  {dt:.1f}s{marker}")

        if patience_left <= 0:
            print(f"\nEarly stopping (patience {args.patience} exhausted).")
            break

    total_time = time.time() - t_all
    print("=" * 70)
    print(f"Best epoch: {best_epoch}  |  Best val loss: {best_val:.6f}  "
          f"|  Total: {total_time / 60:.1f} min")

    print("\nLoading best checkpoint and evaluating on test …")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    metrics = evaluate_test(model, test_loader, device)
    metrics.update({
        "best_val_loss": best_val, "best_epoch": best_epoch,
        "n_params": n_params, "total_time_s": total_time,
    })
    print(f"Test metrics: {metrics}")

    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved:\n  {ckpt_path}\n  {hist_path}\n  {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
