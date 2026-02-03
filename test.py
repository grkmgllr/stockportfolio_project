"""
Testing/Evaluation script for stock price forecasting.
Evaluates High/Low predictions from OHLCV data.

Usage:
    python test.py --ticker AAPL
    python test.py --ticker AAPL --model TimeMixer
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
from dataclasses import dataclass, field
from typing import Literal

from dataset import YahooDataset
from utils import load_checkpoint, calculate_metrics

# Import models
from models import (
    TimeMixer, TimeMixerConfig,
    TimesNetForecastModel, TimesNetForecastConfig,
)


@dataclass
class TestConfig:
    """Test configuration."""
    model_name: Literal["TimeMixer", "TimesNetPure"] = "TimesNetPure"
    ticker: str = "AAPL"
    data_root: str = "data/raw"
    seq_len: int = 30
    pred_len: int = 5
    batch_size: int = 32
    checkpoint_dir: str = "checkpoints"
    device: str = field(default_factory=lambda: (
        "cuda" if torch.cuda.is_available() 
        else "mps" if torch.backends.mps.is_available() 
        else "cpu"
    ))


def get_model_config(model_name: str, seq_len: int, pred_len: int):
    """Get model config for Yahoo stock prediction."""
    if model_name == "TimesNetPure":
        return TimesNetForecastConfig(
            seq_len=seq_len,
            pred_len=pred_len,
            enc_in=5,
            c_out=2,
            d_model=32,
            d_ff=64,
            e_layers=2,
            top_k=3,
            num_kernels=6,
            dropout=0.1,
        )
    elif model_name == "TimeMixer":
        n_layers = 1 if seq_len % 4 != 0 else 2
        return TimeMixerConfig(
            historical_lookback_length=seq_len,
            forecast_horizon_length=pred_len,
            number_of_input_features=5,
            number_of_output_features=2,
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


def evaluate(model, test_loader, criterion, device, dataset):
    """
    Evaluate model on test set.
    
    Returns predictions and ground truth in original scale.
    """
    model.eval()
    
    all_preds = []
    all_trues = []
    test_loss = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_x, None)
            
            loss = criterion(outputs, batch_y)
            test_loss.append(loss.item())
            
            all_preds.append(outputs.cpu().numpy())
            all_trues.append(batch_y.cpu().numpy())
    
    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0)  # [N, pred_len, 2]
    all_trues = np.concatenate(all_trues, axis=0)  # [N, pred_len, 2]
    
    # Inverse transform to original scale
    # Reshape for scaler: [N * pred_len, 2]
    n_samples, pred_len, n_features = all_preds.shape
    
    preds_flat = all_preds.reshape(-1, n_features)
    trues_flat = all_trues.reshape(-1, n_features)
    
    preds_original = dataset.inverse_transform_y(preds_flat).reshape(n_samples, pred_len, n_features)
    trues_original = dataset.inverse_transform_y(trues_flat).reshape(n_samples, pred_len, n_features)
    
    # Calculate metrics on original scale
    metrics = calculate_metrics(preds_original, trues_original)
    
    # Also calculate per-target metrics
    high_metrics = calculate_metrics(preds_original[:, :, 0], trues_original[:, :, 0])
    low_metrics = calculate_metrics(preds_original[:, :, 1], trues_original[:, :, 1])
    
    return {
        'overall': metrics,
        'high': high_metrics,
        'low': low_metrics,
        'test_loss': np.average(test_loss),
    }, preds_original, trues_original


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate stock price forecasting model"
    )

    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker")
    parser.add_argument("--model", type=str, default="TimesNetPure", 
                        choices=["TimeMixer", "TimesNetPure"])
    parser.add_argument("--seq_len", type=int, default=30)
    parser.add_argument("--pred_len", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--data_root", type=str, default="data/raw")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda", "mps"])
    parser.add_argument("--save_predictions", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create config
    test_cfg = TestConfig(
        model_name=args.model,
        ticker=args.ticker,
        data_root=args.data_root,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
    )
    
    if args.device:
        test_cfg.device = args.device
    
    # Get model config
    model_cfg = get_model_config(test_cfg.model_name, test_cfg.seq_len, test_cfg.pred_len)
    
    print("\n" + "=" * 60)
    print("Stock Price Forecasting - Evaluation")
    print("=" * 60)
    print(f"Ticker: {test_cfg.ticker}")
    print(f"Model: {test_cfg.model_name}")
    print(f"Device: {test_cfg.device}")
    print(f"Lookback: {test_cfg.seq_len} days | Forecast: {test_cfg.pred_len} days")
    print("=" * 60 + "\n")
    
    # Load test dataset
    print("Loading Test Data...")
    test_dataset = YahooDataset(
        ticker=test_cfg.ticker,
        root_path=test_cfg.data_root,
        flag='test',
        seq_len=test_cfg.seq_len,
        pred_len=test_cfg.pred_len,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_cfg.batch_size,
        shuffle=False,
        drop_last=False,
    )
    
    print(f"Test samples: {len(test_dataset)}\n")
    
    # Create model
    model = get_model(test_cfg.model_name, model_cfg).to(test_cfg.device)
    
    # Load checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = os.path.join(
            test_cfg.checkpoint_dir,
            f"{test_cfg.ticker}_{test_cfg.model_name}_best.pt"
        )
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        print("Run train.py first to train the model.")
        return
    
    load_checkpoint(model, checkpoint_path, test_cfg.device)
    
    # Evaluate
    print("\nEvaluating on Test Set...")
    print("-" * 60)
    
    criterion = nn.MSELoss()
    results, preds, trues = evaluate(
        model, test_loader, criterion, test_cfg.device, test_dataset
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("TEST RESULTS (Original Scale)")
    print("=" * 60)
    print(f"\nOverall:")
    print(f"  MSE:  {results['overall']['MSE']:.4f}")
    print(f"  MAE:  {results['overall']['MAE']:.4f}")
    print(f"  RMSE: {results['overall']['RMSE']:.4f}")
    print(f"\nHigh Price Prediction:")
    print(f"  MSE:  {results['high']['MSE']:.4f}")
    print(f"  MAE:  {results['high']['MAE']:.4f}")
    print(f"  RMSE: {results['high']['RMSE']:.4f}")
    print(f"\nLow Price Prediction:")
    print(f"  MSE:  {results['low']['MSE']:.4f}")
    print(f"  MAE:  {results['low']['MAE']:.4f}")
    print(f"  RMSE: {results['low']['RMSE']:.4f}")
    print("=" * 60)
    
    # Save predictions
    if args.save_predictions:
        output_dir = "results"
        os.makedirs(output_dir, exist_ok=True)
        
        np.save(os.path.join(output_dir, f"{test_cfg.ticker}_predictions.npy"), preds)
        np.save(os.path.join(output_dir, f"{test_cfg.ticker}_ground_truth.npy"), trues)
        print(f"\nPredictions saved to {output_dir}/")
    
    return results


if __name__ == "__main__":
    main()
