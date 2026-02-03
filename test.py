"""
Testing/Evaluation script for TimeMixer/TimesNetPure models.
Usage: python test.py [--model TimeMixer|TimesNetPure] [--checkpoint PATH]
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
from dataclasses import dataclass, field
from typing import Literal, Union

from dataset import ETTh1Dataset
from utils import load_checkpoint, calculate_metrics

# Import model configs
from models import (
    TimeMixer, TimeMixerConfig,
    TimesNetForecastModel, TimesNetForecastConfig,
)


@dataclass
class TestConfig:
    """Testing and runtime configuration."""
    batch_size: int = 64
    
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


def print_config(model_name: str, model_cfg: ModelConfig, test_cfg: TestConfig) -> None:
    """Print configuration summary."""
    print("\n" + "=" * 50)
    print("Test Configuration Summary")
    print("=" * 50)
    print(f"Model: {model_name}")
    print(f"Device: {test_cfg.device}")
    print(f"\nModel Parameters:")
    print(f"  seq_len: {model_cfg.seq_len}")
    print(f"  pred_len: {model_cfg.pred_len}")
    print(f"  enc_in: {model_cfg.enc_in}")
    print(f"  c_out: {model_cfg.c_out}")
    print("=" * 50 + "\n")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for testing."""
    parser = argparse.ArgumentParser(description="Evaluate forecasting model")

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["TimeMixer", "TimesNetPure"],
        help="Model to evaluate (default: TimesNetPure).",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file.")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size.")
    parser.add_argument("--data_root", type=str, default=None, help="Dataset root path.")
    parser.add_argument(
        "--features",
        type=str,
        default=None,
        choices=["M", "S", "MS"],
        help="Features mode.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Device to use.",
    )
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Checkpoint directory.")
    parser.add_argument("--save_predictions", action="store_true", help="Save predictions to file.")

    return parser.parse_args()


def apply_overrides(test_cfg: TestConfig, args: argparse.Namespace) -> None:
    """Apply command-line argument overrides to test configuration."""
    if args.model is not None:
        test_cfg.model_name = args.model
    if args.batch_size is not None:
        test_cfg.batch_size = args.batch_size
    if args.data_root is not None:
        test_cfg.data_root = args.data_root
    if args.features is not None:
        test_cfg.features = args.features
    if args.device is not None:
        test_cfg.device = args.device
    if args.checkpoint_dir is not None:
        test_cfg.checkpoint_dir = args.checkpoint_dir


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    test_cfg: TestConfig,
    pred_len: int,
) -> tuple:
    """
    Evaluate model on test set.
    
    Returns:
        metrics: Dictionary with MSE, MAE, RMSE
        all_preds: All predictions
        all_trues: All ground truths
    """
    model.eval()
    
    all_preds = []
    all_trues = []
    test_loss = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(test_cfg.device)
            batch_y = batch_y.to(test_cfg.device)
            
            # Forward pass
            outputs = model(batch_x, None)
            
            # Get predictions
            pred = outputs[:, -pred_len:, :]
            true = batch_y[:, -pred_len:, :]
            
            loss = criterion(pred, true)
            test_loss.append(loss.item())
            
            # Store predictions
            all_preds.append(pred.cpu().numpy())
            all_trues.append(true.cpu().numpy())
    
    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0)
    all_trues = np.concatenate(all_trues, axis=0)
    
    # Calculate metrics
    metrics = calculate_metrics(all_preds, all_trues)
    
    return metrics, all_preds, all_trues


def main():
    # Parse arguments and create test config
    args = parse_args()
    test_cfg = TestConfig()
    apply_overrides(test_cfg, args)
    
    # Get model class and config based on model name
    model_class, model_cfg = get_model_and_config(test_cfg.model_name)
    
    # Print configuration
    print_config(test_cfg.model_name, model_cfg, test_cfg)
    
    # Data loading
    print("Loading Test Data...")
    size = [model_cfg.seq_len, 0, model_cfg.pred_len]
    
    test_dataset = ETTh1Dataset(
        root_path=test_cfg.data_root, 
        flag='test', 
        size=size, 
        features=test_cfg.features
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=test_cfg.batch_size, 
        shuffle=False, 
        drop_last=False
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Model initialization
    model = model_class(model_cfg).to(test_cfg.device)
    
    # Determine checkpoint path
    if args.checkpoint is not None:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = os.path.join(
            test_cfg.checkpoint_dir, 
            f"{test_cfg.model_name}_best.pt"
        )
    
    if not os.path.exists(checkpoint_path):
        print(f"\nError: Checkpoint not found at {checkpoint_path}")
        print("Please run train.py first to train the model.")
        return
    
    load_checkpoint(model, checkpoint_path, test_cfg.device)
    
    # Evaluation
    print("\nEvaluating on Test Set...")
    print("-" * 50)
    
    criterion = nn.MSELoss()
    metrics, all_preds, all_trues = evaluate(
        model, test_loader, criterion, test_cfg, model_cfg.pred_len
    )
    
    # Print results
    print("\nTest Results:")
    print("=" * 50)
    print(f"  MSE:  {metrics['MSE']:.6f}")
    print(f"  MAE:  {metrics['MAE']:.6f}")
    print(f"  RMSE: {metrics['RMSE']:.6f}")
    print("=" * 50)
    
    # Optional: Save predictions
    if args.save_predictions:
        output_dir = "results"
        os.makedirs(output_dir, exist_ok=True)
        
        np.save(os.path.join(output_dir, "predictions.npy"), all_preds)
        np.save(os.path.join(output_dir, "ground_truth.npy"), all_trues)
        print(f"\nPredictions saved to {output_dir}/")
    
    return metrics


if __name__ == "__main__":
    main()
