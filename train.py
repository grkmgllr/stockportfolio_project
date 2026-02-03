"""
Training script for TimeMixer/TimesNetPure models.
Usage: python train.py [--model TimeMixer|TimesNetPure] [--epochs N] ...
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import time
import os
import argparse
from dataclasses import dataclass, field
from typing import Literal, Union

from dataset import ETTh1Dataset
from utils import EarlyStopping, get_scheduler, calculate_metrics

# Import model configs
from models import (
    TimeMixer, TimeMixerConfig,
    TimesNetForecastModel, TimesNetForecastConfig,
)


@dataclass
class TrainingConfig:
    """Training and runtime configuration."""
    # Training hyperparameters
    batch_size: int = 64
    epochs: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    patience: int = 5
    grad_clip: float = 1.0
    scheduler: Literal["cosine", "step", "none"] = "cosine"
    scheduler_step_size: int = 10
    scheduler_gamma: float = 0.5
    
    # Data configuration
    data_root: str = "data/processed"
    data_path: str = "ETTh1_processed.csv"
    features: Literal["M", "S", "MS"] = "M"
    
    # Runtime configuration
    device: str = field(default_factory=lambda: (
        "cuda" if torch.cuda.is_available() 
        else "mps" if torch.backends.mps.is_available() 
        else "cpu"
    ))
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    # Model selection
    model_name: Literal["TimeMixer", "TimesNetPure"] = "TimesNetPure"


# Type alias for model configs
ModelConfig = Union[TimeMixerConfig, TimesNetForecastConfig]


def get_model_and_config(model_name: str) -> tuple:
    """
    Get model class and default config based on model name.
    
    Returns:
        Tuple of (model_class, config_instance)
    """
    if model_name == "TimeMixer":
        return TimeMixer, TimeMixerConfig()
    elif model_name == "TimesNetPure":
        return TimesNetForecastModel, TimesNetForecastConfig()
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose 'TimeMixer' or 'TimesNetPure'.")


def print_config(model_name: str, model_cfg: ModelConfig, train_cfg: TrainingConfig) -> None:
    """Print configuration summary."""
    print("\n" + "=" * 50)
    print("Configuration Summary")
    print("=" * 50)
    print(f"Model: {model_name}")
    print(f"Device: {train_cfg.device}")
    print(f"\nModel Parameters:")
    print(f"  seq_len: {model_cfg.seq_len}")
    print(f"  pred_len: {model_cfg.pred_len}")
    print(f"  enc_in: {model_cfg.enc_in}")
    print(f"  c_out: {model_cfg.c_out}")
    if hasattr(model_cfg, 'd_model'):
        print(f"  d_model: {model_cfg.d_model}")
    if hasattr(model_cfg, 'e_layers'):
        print(f"  e_layers: {model_cfg.e_layers}")
    print(f"\nTraining Parameters:")
    print(f"  batch_size: {train_cfg.batch_size}")
    print(f"  epochs: {train_cfg.epochs}")
    print(f"  learning_rate: {train_cfg.learning_rate}")
    print(f"  patience: {train_cfg.patience}")
    print("=" * 50 + "\n")


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_cfg: TrainingConfig,
    pred_len: int,
) -> float:
    """Run one training epoch."""
    model.train()
    train_loss = []
    
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(train_cfg.device)
        batch_y = batch_y.to(train_cfg.device)
        
        optimizer.zero_grad()
        
        # Forward pass (both models accept x, x_mark)
        outputs = model(batch_x, None)
        
        # Get predictions for loss calculation
        pred = outputs[:, -pred_len:, :]
        true = batch_y[:, -pred_len:, :]
        
        loss = criterion(pred, true)
        train_loss.append(loss.item())
        
        # Backward pass with gradient clipping
        loss.backward()
        if train_cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                max_norm=train_cfg.grad_clip
            )
        optimizer.step()
    
    return np.average(train_loss)


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    train_cfg: TrainingConfig,
    pred_len: int,
) -> float:
    """Run validation epoch."""
    model.eval()
    val_loss = []
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(train_cfg.device)
            batch_y = batch_y.to(train_cfg.device)
            
            outputs = model(batch_x, None)
            pred = outputs[:, -pred_len:, :]
            true = batch_y[:, -pred_len:, :]
            
            loss = criterion(pred, true)
            val_loss.append(loss.item())
    
    return np.average(val_loss)

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for training-time configuration overrides.
    
    All arguments are optional. Any argument not provided will be None
    and won't override the default TrainingConfig values.
    """
    parser = argparse.ArgumentParser(description="Train forecasting model")

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["TimeMixer", "TimesNetPure"],
        help="Model to train (default: TimesNetPure).",
    )

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=None, help="Weight decay.")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience.")
    parser.add_argument("--grad_clip", type=float, default=None, help="Gradient clipping max norm.")

    parser.add_argument(
        "--scheduler",
        type=str,
        default=None,
        choices=["cosine", "step", "none"],
        help="LR scheduler type.",
    )
    parser.add_argument("--scheduler_step_size", type=int, default=None, help="Step scheduler step size.")
    parser.add_argument("--scheduler_gamma", type=float, default=None, help="Step scheduler gamma.")

    # Data configuration
    parser.add_argument("--data_root", type=str, default=None, help="Dataset root path.")
    parser.add_argument("--data_path", type=str, default=None, help="Dataset CSV filename.")
    parser.add_argument(
        "--features",
        type=str,
        default=None,
        choices=["M", "S", "MS"],
        help="Features mode (M=multivariate, S=univariate, MS=multivariate-to-single).",
    )

    # Runtime configuration
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Device to use for training.",
    )
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Checkpoint directory.")
    parser.add_argument("--log_dir", type=str, default=None, help="Log directory.")

    return parser.parse_args()


def apply_overrides(train_cfg: TrainingConfig, args: argparse.Namespace) -> None:
    """
    Apply command-line argument overrides to training configuration.
    
    Only arguments explicitly set by the user (not None) are applied.
    """
    # Model selection
    if args.model is not None:
        train_cfg.model_name = args.model

    # Device
    if args.device is not None:
        train_cfg.device = args.device

    # Training hyperparameters
    if args.epochs is not None:
        train_cfg.epochs = args.epochs
    if args.batch_size is not None:
        train_cfg.batch_size = args.batch_size
    if args.lr is not None:
        train_cfg.learning_rate = args.lr
    if args.weight_decay is not None:
        train_cfg.weight_decay = args.weight_decay
    if args.patience is not None:
        train_cfg.patience = args.patience
    if args.grad_clip is not None:
        train_cfg.grad_clip = args.grad_clip
    if args.scheduler is not None:
        train_cfg.scheduler = args.scheduler
    if args.scheduler_step_size is not None:
        train_cfg.scheduler_step_size = args.scheduler_step_size
    if args.scheduler_gamma is not None:
        train_cfg.scheduler_gamma = args.scheduler_gamma

    # Data configuration
    if args.data_root is not None:
        train_cfg.data_root = args.data_root
    if args.data_path is not None:
        train_cfg.data_path = args.data_path
    if args.features is not None:
        train_cfg.features = args.features

    # Paths
    if args.checkpoint_dir is not None:
        train_cfg.checkpoint_dir = args.checkpoint_dir
    if args.log_dir is not None:
        train_cfg.log_dir = args.log_dir

def main():
    # Parse arguments and create training config
    args = parse_args()
    train_cfg = TrainingConfig()
    apply_overrides(train_cfg, args)
    
    # Get model class and config based on model name
    model_class, model_cfg = get_model_and_config(train_cfg.model_name)
    
    # Print configuration
    print_config(train_cfg.model_name, model_cfg, train_cfg)
    
    # Create checkpoint directory
    os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)
    
    # Data loading
    print("Loading Data...")
    size = [model_cfg.seq_len, 0, model_cfg.pred_len]
    
    train_dataset = ETTh1Dataset(
        root_path=train_cfg.data_root, 
        flag='train', 
        size=size, 
        features=train_cfg.features
    )
    val_dataset = ETTh1Dataset(
        root_path=train_cfg.data_root, 
        flag='val', 
        size=size, 
        features=train_cfg.features
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=train_cfg.batch_size, 
        shuffle=True, 
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=train_cfg.batch_size, 
        shuffle=False, 
        drop_last=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Model initialization
    model = model_class(model_cfg).to(train_cfg.device)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss, optimizer, scheduler
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay
    )
    scheduler = get_scheduler(
        optimizer,
        scheduler_type=train_cfg.scheduler,
        epochs=train_cfg.epochs,
        step_size=train_cfg.scheduler_step_size,
        gamma=train_cfg.scheduler_gamma,
    )
    
    # Early stopping
    checkpoint_path = os.path.join(train_cfg.checkpoint_dir, f"{train_cfg.model_name}_best.pt")
    early_stopping = EarlyStopping(
        patience=train_cfg.patience,
        checkpoint_path=checkpoint_path,
        verbose=True
    )
    
    # Training loop
    print("\nStarting Training...")
    print("-" * 60)
    
    for epoch in range(train_cfg.epochs):
        start_time = time.time()
        
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, train_cfg, model_cfg.pred_len
        )
        
        # Validate
        val_loss = validate_epoch(
            model, val_loader, criterion, train_cfg, model_cfg.pred_len
        )
        
        # Scheduler step
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = train_cfg.learning_rate
        
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1:3d}/{train_cfg.epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"LR: {current_lr:.2e} | "
              f"Time: {elapsed:.1f}s")
        
        # Early stopping check
        if early_stopping(val_loss, model, epoch + 1):
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            break
    
    print("-" * 60)
    print(f"Training complete! Best model saved at epoch {early_stopping.best_epoch}")
    print(f"Best validation loss: {early_stopping.best_loss:.4f}")
    print(f"Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
