"""
Utility functions and classes for training and evaluation.
"""
import torch
import numpy as np
import os
from typing import Optional


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    
    def __init__(self, patience: int = 5, delta: float = 0.0, 
                 checkpoint_path: str = "checkpoints/best_model.pt",
                 verbose: bool = True):
        """
        Args:
            patience: Number of epochs to wait for improvement
            delta: Minimum change to qualify as improvement
            checkpoint_path: Path to save the best model
            verbose: Print messages when saving
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


def calculate_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    """
    Calculate regression metrics.
    
    Args:
        pred: Predictions array
        true: Ground truth array
        
    Returns:
        Dictionary with MSE, MAE, RMSE
    """
    mse = np.mean((pred - true) ** 2)
    mae = np.mean(np.abs(pred - true))
    rmse = np.sqrt(mse)
    
    return {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
    }


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


def get_model(model_name: str, model_params: dict, device: str) -> torch.nn.Module:
    """
    Get model instance based on model name.
    
    Args:
        model_name: 'TimeMixer' or 'TimeMixer++'
        model_params: Model configuration parameters
        device: Device to load model on
        
    Returns:
        Model instance
    """
    # Base parameters shared by both models
    base_params = {
        'seq_len': model_params['seq_len'],
        'pred_len': model_params['pred_len'],
        'enc_in': model_params['enc_in'],
        'c_out': model_params['c_out'],
        'd_model': model_params['d_model'],
        'd_ff': model_params['d_ff'],
        'e_layers': model_params['e_layers'],
        'dropout': model_params['dropout'],
        'down_sampling_window': model_params['down_sampling_window'],
        'down_sampling_layers': model_params['down_sampling_layers'],
        'down_sampling_method': model_params['down_sampling_method'],
        'time_feat_dim': model_params['time_feat_dim'],
    }
    
    if model_name == 'TimeMixer':
        from models.TimeMixer.TimeMixer import TimeMixer
        # TimeMixer also accepts decomposition parameters
        base_params['decomp_method'] = model_params['decomp_method']
        base_params['moving_avg'] = model_params['moving_avg']
        base_params['top_k'] = model_params['top_k']
        model = TimeMixer(**base_params)
    elif model_name == 'TimeMixer++':
        from models.TimeMixerpp.TimeMixerpp import TimeMixerPlusPlus
        # TimeMixer++ uses MRTI/TID instead of decomposition params
        model = TimeMixerPlusPlus(**base_params)
    elif model_name == 'TimesNet':
        from models.TimesNet.TimesNet import Model as TimesNet
        # TimesNet uses config-style parameters
        class TimesNetConfig:
            def __init__(self, params):
                self.seq_len = params['seq_len']
                self.pred_len = params['pred_len']
                self.enc_in = params['enc_in']
                self.c_out = params['c_out']
                self.d_model = params['d_model']
                self.d_ff = params['d_ff']
                self.e_layers = params['e_layers']
                self.dropout = params['dropout']
                self.top_k = params['top_k']
                self.num_kernels = params['num_kernels']
                self.embed = params['embed']
                self.freq = params['freq']
                self.task_name = 'long_term_forecast'
        config = TimesNetConfig(model_params)
        model = TimesNet(config)
    elif model_name == 'ModernTCN':
        from models.ModernTCN.ModernTCN import ModernTCN
        # ModernTCN specific parameters
        moderntcn_params = {
            'seq_len': model_params['seq_len'],
            'pred_len': model_params['pred_len'],
            'enc_in': model_params['enc_in'],
            'c_out': model_params['c_out'],
            'd_model': model_params['d_model'],
            'd_ff': model_params['d_ff'],
            'e_layers': model_params['e_layers'],
            'dropout': model_params['dropout'],
            'patch_size': model_params['patch_size'],
            'stride': model_params['stride'],
            'kernel_size': model_params['kernel_size'],
            'use_multi_scale': model_params['use_multi_scale'],
            'use_revin': model_params['use_revin'],
            'time_feat_dim': model_params['time_feat_dim'],
            'head_type': model_params['head_type'],
        }
        model = ModernTCN(**moderntcn_params)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model.to(device)


def get_scheduler(optimizer: torch.optim.Optimizer, config) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Get learning rate scheduler.
    
    Args:
        optimizer: Optimizer instance
        config: Training configuration
        
    Returns:
        Scheduler instance or None
    """
    if config.training.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.training.epochs
        )
    elif config.training.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=config.training.scheduler_step_size,
            gamma=config.training.scheduler_gamma
        )
    elif config.training.scheduler == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler: {config.training.scheduler}")


def print_config(config) -> None:
    """Print configuration summary."""
    print("\n" + "=" * 50)
    print("Configuration Summary")
    print("=" * 50)
    print(f"Model: {config.model_name}")
    print(f"Device: {config.device}")
    print(f"\nModel Parameters:")
    print(f"  seq_len: {config.model.seq_len}")
    print(f"  pred_len: {config.model.pred_len}")
    print(f"  d_model: {config.model.d_model}")
    print(f"  d_ff: {config.model.d_ff}")
    print(f"  e_layers: {config.model.e_layers}")
    print(f"\nTraining Parameters:")
    print(f"  batch_size: {config.training.batch_size}")
    print(f"  epochs: {config.training.epochs}")
    print(f"  learning_rate: {config.training.learning_rate}")
    print(f"  patience: {config.training.patience}")
    print("=" * 50 + "\n")
