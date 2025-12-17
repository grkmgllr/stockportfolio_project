"""
Configuration file for TimeMixer/TimeMixer++/TimesNet training and evaluation.
All hyperparameters and settings are centralized here.
"""
import torch
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ModelConfig:
    """Model architecture hyperparameters."""
    seq_len: int = 96                    # Lookback window
    pred_len: int = 96                   # Prediction horizon
    enc_in: int = 7                      # Number of input features (ETTh1: 7)
    c_out: int = 7                       # Number of output features
    d_model: int = 64                    # Model dimension
    d_ff: int = 128                      # Feed-forward dimension
    e_layers: int = 2                    # Number of encoder layers
    dropout: float = 0.1                 # Dropout rate
    
    # TimeMixer/TimeMixer++ specific
    down_sampling_window: int = 2        # Downsampling window size
    down_sampling_layers: int = 2        # Number of downsampling layers
    down_sampling_method: str = "avg"    # Downsampling method: 'avg', 'max', 'conv'
    decomp_method: str = "moving_avg"    # Decomposition: 'moving_avg' or 'dft_decomp'
    moving_avg: int = 25                 # Moving average kernel size
    top_k: int = 5                       # Top-k frequencies for DFT/TimesNet
    time_feat_dim: int = 0               # Time feature dimension (0 = not used)
    
    # TimesNet specific
    num_kernels: int = 6                 # Number of kernels for Inception block
    embed: str = "timeF"                 # Embedding type: 'timeF', 'fixed', 'learned'
    freq: str = "h"                      # Frequency: 's'=secondly, 'h'=hourly, 'd'=daily


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    batch_size: int = 64
    epochs: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    patience: int = 5                    # Early stopping patience
    grad_clip: float = 1.0               # Gradient clipping max norm
    scheduler: str = "cosine"            # LR scheduler: 'cosine', 'step', 'none'
    scheduler_step_size: int = 10        # For step scheduler
    scheduler_gamma: float = 0.5         # For step scheduler
    num_workers: int = 0                 # DataLoader workers (0 for MPS compatibility)
    pin_memory: bool = True              # Pin memory for faster GPU transfer


@dataclass
class DataConfig:
    """Data loading configuration."""
    root_path: str = "data/processed"
    data_path: str = "ETTh1_processed.csv"
    features: Literal["M", "S", "MS"] = "M"  # M: multivariate, S: univariate
    target: str = "OT"                   # Target column for univariate
    scale: bool = True                   # Whether to apply StandardScaler


@dataclass
class Config:
    """Main configuration combining all settings."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    # Model selection: 'TimeMixer', 'TimeMixer++', or 'TimesNet'
    model_name: Literal["TimeMixer", "TimeMixer++", "TimesNet"] = "TimeMixer++"
    
    # Paths
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    # Device selection with optimizations
    device: str = field(default_factory=lambda: (
        "cuda" if torch.cuda.is_available() 
        else "mps" if torch.backends.mps.is_available() 
        else "cpu"
    ))
    
    # GPU Optimizations
    use_amp: bool = False                # Automatic Mixed Precision (CUDA only)
    compile_model: bool = False          # torch.compile (PyTorch 2.0+, CUDA only)
    
    def __post_init__(self):
        """Apply device-specific optimizations."""
        # AMP only works well with CUDA
        if self.device != "cuda":
            self.use_amp = False
            self.compile_model = False
        
        # MPS doesn't support pin_memory well with workers
        if self.device == "mps":
            self.training.num_workers = 0
    
    def get_model_params(self) -> dict:
        """Returns model parameters as a dictionary for model initialization."""
        return {
            'seq_len': self.model.seq_len,
            'pred_len': self.model.pred_len,
            'enc_in': self.model.enc_in,
            'c_out': self.model.c_out,
            'd_model': self.model.d_model,
            'd_ff': self.model.d_ff,
            'e_layers': self.model.e_layers,
            'dropout': self.model.dropout,
            'down_sampling_window': self.model.down_sampling_window,
            'down_sampling_layers': self.model.down_sampling_layers,
            'down_sampling_method': self.model.down_sampling_method,
            'decomp_method': self.model.decomp_method,
            'moving_avg': self.model.moving_avg,
            'top_k': self.model.top_k,
            'time_feat_dim': self.model.time_feat_dim,
            # TimesNet specific
            'num_kernels': self.model.num_kernels,
            'embed': self.model.embed,
            'freq': self.model.freq,
        }


# Default configuration instance
def get_config() -> Config:
    """Returns the default configuration."""
    return Config()
