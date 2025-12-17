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


def main():
    # Load configuration
    config = get_config()
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
