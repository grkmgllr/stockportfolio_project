"""
Training script for TimeMixer/TimeMixer++ models.
Usage: python train.py
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import time
import os
import argparse

from configs.config import get_config
from dataset import ETTh1Dataset
from utils import (
    EarlyStopping, 
    get_model, 
    get_scheduler, 
    print_config,
    calculate_metrics
)


def train_epoch(model, train_loader, criterion, optimizer, config):
    """Run one training epoch."""
    model.train()
    train_loss = []
    
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(config.device)
        batch_y = batch_y.to(config.device)
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(batch_x, None)
        
        # Get predictions for loss calculation
        pred = outputs[:, -config.model.pred_len:, :]
        true = batch_y[:, -config.model.pred_len:, :]
        
        loss = criterion(pred, true)
        train_loss.append(loss.item())
        
        # Backward pass with gradient clipping
        loss.backward()
        if config.training.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                max_norm=config.training.grad_clip
            )
        optimizer.step()
    
    return np.average(train_loss)


def validate_epoch(model, val_loader, criterion, config):
    """Run validation epoch."""
    model.eval()
    val_loss = []
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(config.device)
            batch_y = batch_y.to(config.device)
            
            outputs = model(batch_x, None)
            pred = outputs[:, -config.model.pred_len:, :]
            true = batch_y[:, -config.model.pred_len:, :]
            
            loss = criterion(pred, true)
            val_loss.append(loss.item())
    
    return np.average(val_loss)

def parse_args():
    """
    Parse command-line arguments for training-time configuration overrides.

    This function defines and parses a set of optional command-line arguments
    that allow users to override selected fields of the static configuration
    file at runtime. It is designed to support lightweight experimentation
    without modifying configuration files or source code.

    The parsed arguments are intentionally limited to:
        - model selection,
        - training hyperparameters,
        - data paths and feature modes,
        - runtime environment options (device, output paths).

    All arguments are optional. Any argument that is not explicitly provided
    by the user will be returned as None and must not override the default
    configuration values.

    The parsed arguments are expected to be applied to the configuration
    object via a separate post-processing function (e.g., `apply_overrides`).

    Returns
    -------
    argparse.Namespace
        A namespace object containing parsed command-line arguments.
        Each attribute corresponds to a CLI flag and is either:
            - a user-provided value, or
            - None if the flag was not specified.
    """

    parser = argparse.ArgumentParser(description="Train forecasting model")

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["TimeMixer", "TimeMixer++", "TimesNetPure", "ModernTCN"],
        help="Override model_name in config.",
    )

    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs.")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--weight_decay", type=float, default=None, help="Override weight decay.")
    parser.add_argument("--patience", type=int, default=None, help="Override early stopping patience.")
    parser.add_argument("--grad_clip", type=float, default=None, help="Override grad clip max norm.")

    parser.add_argument(
        "--scheduler",
        type=str,
        default=None,
        choices=["cosine", "step", "none"],
        help="Override scheduler type.",
    )
    parser.add_argument("--scheduler_step_size", type=int, default=None, help="Override step scheduler step size.")
    parser.add_argument("--scheduler_gamma", type=float, default=None, help="Override step scheduler gamma.")

    parser.add_argument("--data_root", type=str, default=None, help="Override dataset root_path.")
    parser.add_argument("--data_path", type=str, default=None, help="Override dataset csv filename.")
    parser.add_argument(
        "--features",
        type=str,
        default=None,
        choices=["M", "S", "MS"],
        help="Override features mode.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Override device.",
    )

    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Override checkpoint directory.")
    parser.add_argument("--log_dir", type=str, default=None, help="Override log directory.")

    return parser.parse_args()

def apply_overrides(config, args):
    """
    Apply command-line argument overrides to a configuration object.

    This function updates a configuration instance in-place using values
    provided via command-line arguments. Only arguments that are explicitly
    set by the user (i.e., not None) are applied; all other configuration
    fields remain unchanged.

    The override logic is intentionally explicit and conservative:
        - No new configuration fields are introduced.
        - Nested configuration objects (e.g., training, data) are updated
          field-by-field to preserve structure.
        - Device-specific constraints (e.g., disabling AMP on non-CUDA
          backends) are enforced after overrides are applied.

    This separation of concerns ensures that:
        - configuration parsing,
        - configuration definition,
        - and configuration mutation

    remain logically decoupled and easy to reason about.

    Args
    ----
    config : Config
        Configuration object returned by `get_config()`. This object is
        modified in-place.
    args : argparse.Namespace
        Parsed command-line arguments returned by `parse_args()`.

    Returns
    -------
    None
        This function has no return value. All updates are applied directly
        to the provided configuration object.

    Notes
    -----
    - If a device override is provided, device-dependent flags such as AMP
      and torch.compile are adjusted accordingly.
    - This function does not validate semantic consistency between arguments
      (e.g., model compatibility); it assumes valid input.
    """
    
    # model
    if args.model is not None:
        config.model_name = args.model

    # device
    if args.device is not None:
        config.device = args.device
        # mirror config device rules
        if config.device != "cuda":
            config.use_amp = False
            config.compile_model = False
        if config.device == "mps":
            config.training.num_workers = 0

    # training
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.lr is not None:
        config.training.learning_rate = args.lr
    if args.weight_decay is not None:
        config.training.weight_decay = args.weight_decay
    if args.patience is not None:
        config.training.patience = args.patience
    if args.grad_clip is not None:
        config.training.grad_clip = args.grad_clip
    if args.scheduler is not None:
        config.training.scheduler = args.scheduler
    if args.scheduler_step_size is not None:
        config.training.scheduler_step_size = args.scheduler_step_size
    if args.scheduler_gamma is not None:
        config.training.scheduler_gamma = args.scheduler_gamma

    # data
    if args.data_root is not None:
        config.data.root_path = args.data_root
    if args.data_path is not None:
        config.data.data_path = args.data_path
    if args.features is not None:
        config.data.features = args.features

    # paths
    if args.checkpoint_dir is not None:
        config.checkpoint_dir = args.checkpoint_dir
    if args.log_dir is not None:
        config.log_dir = args.log_dir

def main():
    # Load configuration
    args = parse_args()

    # Load configuration
    config = get_config()
    apply_overrides(config, args)
    print_config(config)
    
    # Create checkpoint directory
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # Data loading
    print("Loading Data...")
    size = [config.model.seq_len, 0, config.model.pred_len]
    
    train_dataset = ETTh1Dataset(
        root_path=config.data.root_path, 
        flag='train', 
        size=size, 
        features=config.data.features
    )
    val_dataset = ETTh1Dataset(
        root_path=config.data.root_path, 
        flag='val', 
        size=size, 
        features=config.data.features
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.training.batch_size, 
        shuffle=True, 
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.training.batch_size, 
        shuffle=False, 
        drop_last=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Model initialization
    model = get_model(config.model_name, config.get_model_params(), config.device)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss, optimizer, scheduler
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    scheduler = get_scheduler(optimizer, config)
    
    # Early stopping
    checkpoint_path = os.path.join(config.checkpoint_dir, f"{config.model_name.replace('+', 'p')}_best.pt")
    early_stopping = EarlyStopping(
        patience=config.training.patience,
        checkpoint_path=checkpoint_path,
        verbose=True
    )
    
    # Training loop
    print("\nStarting Training...")
    print("-" * 60)
    
    for epoch in range(config.training.epochs):
        start_time = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, config)
        
        # Validate
        val_loss = validate_epoch(model, val_loader, criterion, config)
        
        # Scheduler step
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = config.training.learning_rate
        
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1:3d}/{config.training.epochs} | "
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
