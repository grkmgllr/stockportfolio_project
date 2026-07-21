"""PyTorch forecaster (TimeMixer / TimesNet): full train and evaluate flows."""
from __future__ import annotations

import os
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader

from dataset import StockDataset
from models.registry import get_model, get_model_config
from paths import CHECKPOINTS_ROOT, forecaster_checkpoint
from utils import EarlyStopping, get_scheduler, load_checkpoint

from forecasting.config import TrainingConfig
from forecasting.loop import evaluate as evaluate_loop, train_epoch, validate_epoch
from reporting import print_epoch, print_training_config


def train(tickers: List[str], model_name: str, ma_targets: List[str],
          *, seq_len: int, pred_len: int,
          epochs: int, batch_size: int, lr: float,
          patience: int, data_root: str,
          start_date: str | None = "2022-01-01",
          device: str | None = None,
          checkpoint_dir: str = CHECKPOINTS_ROOT) -> str:
    """Train a PyTorch forecaster (optionally on pooled tickers).

    Returns the checkpoint path so callers can load it back.
    """
    label = "pooled" if len(tickers) > 1 else tickers[0]

    train_cfg = TrainingConfig(
        model_name=model_name,
        ticker=label,
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
    train_datasets, val_datasets = [], []
    for t in tickers:
        train_datasets.append(StockDataset(
            ticker=t, root_path=train_cfg.data_root, flag="train",
            seq_len=train_cfg.seq_len, pred_len=train_cfg.pred_len,
            ma_targets=ma_targets, return_targets=True, start_date=start_date,
        ))
        val_datasets.append(StockDataset(
            ticker=t, root_path=train_cfg.data_root, flag="val",
            seq_len=train_cfg.seq_len, pred_len=train_cfg.pred_len,
            ma_targets=ma_targets, return_targets=True, start_date=start_date,
        ))

    if len(tickers) > 1:
        train_dataset = ConcatDataset(train_datasets)
        val_dataset = ConcatDataset(val_datasets)
    else:
        train_dataset = train_datasets[0]
        val_dataset = val_datasets[0]

    # all tickers share the same features; first dataset gives metadata
    ref_dataset = train_datasets[0]

    model_cfg = get_model_config(
        train_cfg.model_name, train_cfg.seq_len, train_cfg.pred_len,
        enc_in=ref_dataset.enc_in,
        c_out=ref_dataset.c_out,
        denorm_indices=ref_dataset.denorm_indices,
        return_targets=True,
    )

    print_training_config(train_cfg, model_cfg, ref_dataset.target_features, tickers)

    train_loader = DataLoader(train_dataset, batch_size=train_cfg.batch_size,
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=train_cfg.batch_size,
                            shuffle=False, drop_last=False)

    print(f"\nTrain samples: {len(train_dataset)}")
    if len(tickers) > 1:
        for ds in train_datasets:
            print(f"  {ds.ticker}: {len(ds)}")
    print(f"Val samples: {len(val_dataset)}")

    model = get_model(train_cfg.model_name, model_cfg).to(train_cfg.device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )
    scheduler = get_scheduler(
        optimizer,
        scheduler_type=train_cfg.scheduler,
        epochs=train_cfg.epochs,
        step_size=train_cfg.scheduler_step_size,
        gamma=train_cfg.scheduler_gamma,
    )

    checkpoint_path = os.path.join(
        train_cfg.checkpoint_dir,
        f"{train_cfg.ticker}_{train_cfg.model_name}_best.pt",
    )
    early_stopping = EarlyStopping(
        patience=train_cfg.patience,
        checkpoint_path=checkpoint_path,
        verbose=True,
    )

    print("\nStarting Training...")
    print("-" * 60)

    for epoch in range(train_cfg.epochs):
        start_time = time.time()
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer,
            train_cfg.device, train_cfg.grad_clip,
        )
        val_loss = validate_epoch(model, val_loader, criterion, train_cfg.device)

        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = train_cfg.learning_rate

        print_epoch(epoch + 1, train_cfg.epochs, train_loss, val_loss,
                    current_lr, time.time() - start_time)

        if early_stopping(val_loss, model, epoch + 1):
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    print("-" * 60)
    print(f"Training complete! Best epoch: {early_stopping.best_epoch}")
    print(f"Best validation loss: {early_stopping.best_loss:.6f}")
    print(f"Checkpoint: {checkpoint_path}")

    return checkpoint_path


def evaluate(ticker: str, model_name: str, ma_targets: List[str],
             *, seq_len: int, pred_len: int,
             batch_size: int, data_root: str,
             checkpoint_override: str | None = None,
             start_date: str | None = "2022-01-01",
             device: str | None = None,
             checkpoint_dir: str = CHECKPOINTS_ROOT
             ) -> Tuple[np.ndarray, np.ndarray, List[str], dict]:
    """Load a saved PyTorch model and evaluate on one ticker's test split."""
    test_dataset = StockDataset(
        ticker=ticker, root_path=data_root, flag="test",
        seq_len=seq_len, pred_len=pred_len,
        ma_targets=ma_targets, return_targets=True, start_date=start_date,
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, drop_last=False)

    model_cfg = get_model_config(
        model_name, seq_len, pred_len,
        enc_in=test_dataset.enc_in, c_out=test_dataset.c_out,
        denorm_indices=test_dataset.denorm_indices, return_targets=True,
    )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model(model_name, model_cfg).to(device)

    checkpoint_path = checkpoint_override or forecaster_checkpoint(ticker, model_name)
    load_checkpoint(model, checkpoint_path, device)

    criterion = nn.MSELoss()
    eval_results, preds, trues = evaluate_loop(
        model, test_loader, criterion, device, test_dataset,
    )
    return preds, trues, test_dataset.target_features, eval_results
