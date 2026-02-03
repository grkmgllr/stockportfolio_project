"""
Training script for stock price forecasting.
Predicts High/Low from OHLCV data using TimeMixer or TimesNetPure.

Usage:
    python train.py --ticker AAPL
    python train.py --ticker AAPL --model TimeMixer --epochs 100
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import time
import os
import argparse
from dataclasses import dataclass, field
from typing import Literal

from dataset import YahooDataset
from utils import EarlyStopping, get_scheduler, calculate_metrics

# Import models
from models import (
    TimeMixer, TimeMixerConfig,
    TimesNetForecastModel, TimesNetForecastConfig,
)


@dataclass
class TrainingConfig:
    """
    Training configuration for stock price forecasting.
    
    Contains all hyperparameters and settings for training a model
    to predict High/Low prices from OHLCV input data.
    
    Attributes:
        model_name: Model architecture ('TimeMixer' or 'TimesNetPure')
        ticker: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        data_root: Directory containing {ticker}.csv files
        seq_len: Number of historical days to use as input (lookback window)
        pred_len: Number of future days to predict (forecast horizon)
        batch_size: Training batch size
        epochs: Maximum number of training epochs
        learning_rate: Initial learning rate for Adam optimizer
        weight_decay: L2 regularization coefficient
        patience: Early stopping patience (epochs without improvement)
        grad_clip: Maximum gradient norm for clipping
        scheduler: Learning rate scheduler type
        device: Compute device (auto-detected if not specified)
        checkpoint_dir: Directory to save model checkpoints
    """
    # Model selection
    model_name: Literal["TimeMixer", "TimesNetPure"] = "TimesNetPure"
    
    # Data configuration
    ticker: str = "AAPL"
    data_root: str = "data/raw"
    seq_len: int = 30       # 30 days lookback
    pred_len: int = 5       # 5 days forecast
    
    # Training hyperparameters
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    patience: int = 10
    grad_clip: float = 1.0
    scheduler: Literal["cosine", "step", "none"] = "cosine"
    scheduler_step_size: int = 10
    scheduler_gamma: float = 0.5
    
    # Runtime configuration
    device: str = field(default_factory=lambda: (
        "cuda" if torch.cuda.is_available() 
        else "mps" if torch.backends.mps.is_available() 
        else "cpu"
    ))
    checkpoint_dir: str = "checkpoints"


def get_model_config(model_name: str, seq_len: int, pred_len: int):
    """
    Get model config for Yahoo stock prediction.
    
    Input: OHLCV (5 features)
    Output: High, Low (2 features)
    """
    if model_name == "TimesNetPure":
        return TimesNetForecastConfig(
            seq_len=seq_len,
            pred_len=pred_len,
            enc_in=5,       # OHLCV
            c_out=2,        # High, Low
            d_model=32,
            d_ff=64,
            e_layers=2,
            top_k=3,
            num_kernels=6,
            dropout=0.1,
        )
    elif model_name == "TimeMixer":
        # Adjust downsampling layers based on seq_len divisibility
        # seq_len must be divisible by 2^number_of_downsampling_layers
        n_layers = 1 if seq_len % 4 != 0 else 2
        return TimeMixerConfig(
            historical_lookback_length=seq_len,
            forecast_horizon_length=pred_len,
            number_of_input_features=5,     # OHLCV
            number_of_output_features=2,    # High, Low
            model_embedding_dimension=64,
            feedforward_hidden_dimension=128,
            number_of_pdm_blocks=2,
            dropout_probability=0.1,
            downsampling_window_size=2,
            number_of_downsampling_layers=n_layers,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_model(model_name: str, config):
    """Create model instance from config."""
    if model_name == "TimesNetPure":
        return TimesNetForecastModel(config)
    elif model_name == "TimeMixer":
        return TimeMixer(config)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def print_config(train_cfg: TrainingConfig, model_cfg) -> None:
    """Print configuration summary."""
    print("\n" + "=" * 60)
    print("Stock Price Forecasting - Training Configuration")
    print("=" * 60)
    print(f"Ticker: {train_cfg.ticker}")
    print(f"Model: {train_cfg.model_name}")
    print(f"Device: {train_cfg.device}")
    print(f"\nTask: Predict High/Low from OHLCV")
    print(f"  Input:  OHLCV ({model_cfg.enc_in} features)")
    print(f"  Output: High, Low ({model_cfg.c_out} features)")
    print(f"  Lookback: {model_cfg.seq_len} days")
    print(f"  Forecast: {model_cfg.pred_len} days")
    print(f"\nTraining:")
    print(f"  Batch size: {train_cfg.batch_size}")
    print(f"  Epochs: {train_cfg.epochs}")
    print(f"  Learning rate: {train_cfg.learning_rate}")
    print(f"  Patience: {train_cfg.patience}")
    print("=" * 60 + "\n")


def train_epoch(model, train_loader, criterion, optimizer, device, grad_clip):
    """Run one training epoch."""
    model.train()
    train_loss = []
    
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(batch_x, None)
        
        # outputs: [B, pred_len, c_out] - already correct shape
        loss = criterion(outputs, batch_y)
        train_loss.append(loss.item())
        
        # Backward pass
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
    
    return np.average(train_loss)


def validate_epoch(model, val_loader, criterion, device):
    """Run validation epoch."""
    model.eval()
    val_loss = []
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_x, None)
            loss = criterion(outputs, batch_y)
            val_loss.append(loss.item())
    
    return np.average(val_loss)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train stock price forecasting model (predict High/Low from OHLCV)"
    )

    # Required
    parser.add_argument(
        "--ticker",
        type=str,
        default="AAPL",
        help="Stock ticker symbol (default: AAPL)",
    )
    
    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="TimesNetPure",
        choices=["TimeMixer", "TimesNetPure"],
        help="Model to train (default: TimesNetPure)",
    )
    
    # Sequence lengths
    parser.add_argument("--seq_len", type=int, default=30, help="Lookback window in days (default: 30)")
    parser.add_argument("--pred_len", type=int, default=5, help="Prediction horizon in days (default: 5)")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "step", "none"])

    # Data
    parser.add_argument("--data_root", type=str, default="data/raw", help="Data directory")

    # Runtime
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda", "mps"])
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")

    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create training config
    train_cfg = TrainingConfig(
        model_name=args.model,
        ticker=args.ticker,
        data_root=args.data_root,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        grad_clip=args.grad_clip,
        scheduler=args.scheduler,
        checkpoint_dir=args.checkpoint_dir,
    )
    
    if args.device:
        train_cfg.device = args.device
    
    # Get model config
    model_cfg = get_model_config(train_cfg.model_name, train_cfg.seq_len, train_cfg.pred_len)
    
    # Print configuration
    print_config(train_cfg, model_cfg)
    
    # Create checkpoint directory
    os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)
    
    # Load datasets
    print("Loading Data...")
    train_dataset = YahooDataset(
        ticker=train_cfg.ticker,
        root_path=train_cfg.data_root,
        flag='train',
        seq_len=train_cfg.seq_len,
        pred_len=train_cfg.pred_len,
    )
    val_dataset = YahooDataset(
        ticker=train_cfg.ticker,
        root_path=train_cfg.data_root,
        flag='val',
        seq_len=train_cfg.seq_len,
        pred_len=train_cfg.pred_len,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        drop_last=True,
    )
    
    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create model
    model = get_model(train_cfg.model_name, model_cfg).to(train_cfg.device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss, optimizer, scheduler
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
    
    # Early stopping
    checkpoint_path = os.path.join(
        train_cfg.checkpoint_dir,
        f"{train_cfg.ticker}_{train_cfg.model_name}_best.pt"
    )
    early_stopping = EarlyStopping(
        patience=train_cfg.patience,
        checkpoint_path=checkpoint_path,
        verbose=True,
    )
    
    # Training loop
    print("\nStarting Training...")
    print("-" * 60)
    
    for epoch in range(train_cfg.epochs):
        start_time = time.time()
        
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer,
            train_cfg.device, train_cfg.grad_clip
        )
        val_loss = validate_epoch(model, val_loader, criterion, train_cfg.device)
        
        # Scheduler step
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = train_cfg.learning_rate
        
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1:3d}/{train_cfg.epochs} | "
              f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
              f"LR: {current_lr:.2e} | Time: {elapsed:.1f}s")
        
        # Early stopping
        if early_stopping(val_loss, model, epoch + 1):
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break
    
    print("-" * 60)
    print(f"Training complete! Best epoch: {early_stopping.best_epoch}")
    print(f"Best validation loss: {early_stopping.best_loss:.6f}")
    print(f"Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
