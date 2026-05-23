"""
Utility functions and classes for training and evaluation.

This module provides:
- EarlyStopping: Stop training when validation loss plateaus
- calculate_metrics: Compute MSE, MAE, RMSE for predictions
- load_checkpoint: Load model weights from saved checkpoint
- get_scheduler: Create learning rate scheduler
"""
import torch
import numpy as np
from scipy import stats
import os
from typing import Optional, Dict


class EarlyStopping:
    """
    Early stopping callback to stop training when validation loss doesn't improve.
    
    Monitors validation loss and saves the best model checkpoint. Training stops
    if no improvement is seen for `patience` consecutive epochs.
    
    Attributes:
        best_loss: Best validation loss seen so far
        best_epoch: Epoch number of the best model
        early_stop: Flag indicating whether to stop training
    """
    
    def __init__(
        self,
        patience: int = 5,
        delta: float = 0.0, 
        checkpoint_path: str = "checkpoints/best_model.pt",
        verbose: bool = True
    ):
        """
        Args:
            patience: Number of epochs to wait for improvement before stopping
            delta: Minimum change in loss to qualify as improvement
            checkpoint_path: Path to save the best model checkpoint
            verbose: Whether to print messages when saving checkpoints
        """
        self.patience = patience
        self.delta = delta
        self.checkpoint_path = checkpoint_path
        self.verbose = verbose
        
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, val_loss: float, model: torch.nn.Module, epoch: int) -> bool:
        """
        Check if training should stop.
        
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model, epoch)
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model, epoch)
            self.counter = 0
            
        return self.early_stop
    
    def save_checkpoint(self, model: torch.nn.Module, epoch: int):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_loss': self.best_loss,
        }, self.checkpoint_path)
        self.best_epoch = epoch
        if self.verbose:
            print(f"Saved best model (epoch {epoch}, val_loss: {self.best_loss:.4f})")


def calculate_metrics(pred: np.ndarray, true: np.ndarray) -> Dict[str, float]:
    """
    Calculate regression metrics for forecasting evaluation.

    Args:
        pred: Predictions array of any shape
        true: Ground truth array (same shape as pred)

    Returns:
        Dictionary with MSE, MAE, RMSE.
    """
    mse = np.mean((pred - true) ** 2)
    mae = np.mean(np.abs(pred - true))
    rmse = np.sqrt(mse)

    return {
        'MSE': float(mse),
        'MAE': float(mae),
        'RMSE': float(rmse),
    }


def calculate_return_metrics(
    pred_returns: np.ndarray,
    true_returns: np.ndarray,
) -> Dict[str, float]:
    """
    Calculate return-based metrics following Fan & Shen [2024], Wang et al. [2025].

    Computes Information Coefficient (IC) and Rank Information Coefficient (RIC)
    between predicted and realised returns.

    IC  = Pearson correlation  (linear agreement)
    RIC = Spearman correlation (rank-order agreement)

    Args:
        pred_returns: Predicted returns, shape [N, pred_len] or [N, pred_len, C].
        true_returns: Actual returns, same shape.

    Returns:
        Dictionary with IC, RIC, and directional accuracy (DA).
    """
    p = pred_returns.flatten()
    t = true_returns.flatten()

    mask = np.isfinite(p) & np.isfinite(t)
    p, t = p[mask], t[mask]

    if len(p) < 3:
        return {'IC': float('nan'), 'RIC': float('nan'), 'DA': float('nan')}

    ic = float(np.corrcoef(p, t)[0, 1])
    ric = float(stats.spearmanr(p, t).statistic)

    # Directional accuracy: fraction of times predicted sign matches actual sign
    da = float(np.mean(np.sign(p) == np.sign(t)))

    return {'IC': ic, 'RIC': ric, 'DA': da}


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str, 
                    device: str = 'cpu') -> dict:
    """
    Load model from checkpoint.
    
    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint file
        device: Device to load the model on
        
    Returns:
        Checkpoint dictionary with metadata
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} "
          f"(val_loss: {checkpoint['val_loss']:.4f})")
    return checkpoint


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    epochs: int,
    step_size: int = 10,
    gamma: float = 0.5,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Get learning rate scheduler.
    
    Args:
        optimizer: Optimizer instance
        scheduler_type: Type of scheduler ('cosine', 'step', 'none')
        epochs: Total number of training epochs (for cosine scheduler)
        step_size: Step size for step scheduler
        gamma: Gamma for step scheduler
        
    Returns:
        Scheduler instance or None
    """
    if scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_type == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")
