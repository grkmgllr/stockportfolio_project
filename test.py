"""
Testing/Evaluation script for stock price forecasting.
Evaluates High/Close (+ optional MA) predictions from OHLCV data.

Usage:
    python test.py --ticker AAPL
    python test.py --ticker AAPL --model TimeMixer
    python test.py --ticker AAPL --model LightGBM --ma_targets EMA_20 SMA_50 --save_predictions
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
from dataclasses import dataclass, field
from typing import List, Literal

import pandas as pd

from dataset import ParquetDataset
from utils import load_checkpoint, calculate_metrics

# Import models
from models import (
    TimeMixer, TimeMixerConfig,
    TimesNetModel, TimesNetConfig,
)
from models.LightGBMForecaster import LightGBMForecaster


@dataclass
class TestConfig:
    """
    Test/evaluation configuration for stock price forecasting.
    
    Contains settings for evaluating a trained model on the test set.
    Parameters must match those used during training (especially seq_len, pred_len).
    
    Attributes:
        model_name: Model architecture (must match trained model)
        ticker: Stock ticker symbol (must match trained model)
        data_root: Directory containing {ticker}.csv files
        seq_len: Lookback window (must match training)
        pred_len: Forecast horizon (must match training)
        batch_size: Evaluation batch size
        checkpoint_dir: Directory containing saved checkpoints
        device: Compute device (auto-detected if not specified)
    """
    model_name: Literal["TimeMixer", "TimesNet", "LightGBM"] = "TimesNet"
    ticker: str = "AAPL"
    data_root: str = "data/raw"
    seq_len: int = 14
    pred_len: int = 5
    batch_size: int = 32
    checkpoint_dir: str = "checkpoints"
    device: str = field(default_factory=lambda: (
        "cuda" if torch.cuda.is_available() 
        else "mps" if torch.backends.mps.is_available() 
        else "cpu"
    ))


def get_model_config(model_name: str, seq_len: int, pred_len: int,
                     enc_in: int = 5, c_out: int = 2):
    """Get model config for stock price prediction."""
    if model_name == "TimesNet":
        return TimesNetConfig(
            task_name="long_term_forecast",
            seq_len=seq_len,
            pred_len=pred_len,
            enc_in=enc_in,
            c_out=c_out,
            d_model=32,
            d_ff=64,
            e_layers=2,
            top_k=3,
            num_kernels=6,
            embed="fixed",
            freq="h",
            dropout=0.1,
            num_class=c_out,
        )
    elif model_name == "TimeMixer":
        n_layers = 1 if seq_len % 4 != 0 else 2
        return TimeMixerConfig(
            historical_lookback_length=seq_len,
            forecast_horizon_length=pred_len,
            number_of_input_features=enc_in,
            number_of_output_features=c_out,
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
    if model_name == "TimesNet":
        return TimesNetModel(config)
    elif model_name == "TimeMixer":
        return TimeMixer(config)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def evaluate(model, test_loader, criterion, device, dataset):
    """
    Evaluate model on test set.
    
    Returns predictions and ground truth in original scale, with
    per-target metrics keyed by the target feature name.
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
    
    # Concatenate all batches: [N, pred_len, n_targets]
    all_preds = np.concatenate(all_preds, axis=0)
    all_trues = np.concatenate(all_trues, axis=0)
    
    n_samples, pred_len, n_features = all_preds.shape
    
    preds_flat = all_preds.reshape(-1, n_features)
    trues_flat = all_trues.reshape(-1, n_features)
    
    preds_original = dataset.inverse_transform_y(preds_flat).reshape(n_samples, pred_len, n_features)
    trues_original = dataset.inverse_transform_y(trues_flat).reshape(n_samples, pred_len, n_features)
    
    results = {
        'overall': calculate_metrics(preds_original, trues_original),
        'test_loss': np.average(test_loss),
    }

    # Per-target metrics keyed by feature name
    for i, name in enumerate(dataset.target_features):
        results[name] = calculate_metrics(
            preds_original[:, :, i], trues_original[:, :, i],
        )
    
    return results, preds_original, trues_original


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate stock price forecasting model"
    )

    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker")
    parser.add_argument("--model", type=str, default="TimesNet", 
                        choices=["TimeMixer", "TimesNet", "LightGBM"])
    parser.add_argument("--seq_len", type=int, default=30)
    parser.add_argument("--pred_len", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--data_root", type=str, default="data/raw")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda", "mps"])
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--ma_targets", nargs="*", default=None,
                        help="Moving-average targets to predict (e.g. EMA_20 SMA_50)")

    return parser.parse_args()


def print_results(results: dict, target_names: List[str]) -> None:
    """Print evaluation metrics for all targets."""
    print("\n" + "=" * 60)
    print("TEST RESULTS (Original Scale)")
    print("=" * 60)
    print(f"\nOverall:")
    print(f"  MSE:  {results['overall']['MSE']:.4f}")
    print(f"  MAE:  {results['overall']['MAE']:.4f}")
    print(f"  RMSE: {results['overall']['RMSE']:.4f}")
    for name in target_names:
        print(f"\n{name} Prediction:")
        print(f"  MSE:  {results[name]['MSE']:.4f}")
        print(f"  MAE:  {results[name]['MAE']:.4f}")
        print(f"  RMSE: {results[name]['RMSE']:.4f}")
    print("=" * 60)


def save_predictions(ticker: str, preds: np.ndarray, trues: np.ndarray) -> None:
    """Save predictions and ground truth as .npy files."""
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, f"{ticker}_predictions.npy"), preds)
    np.save(os.path.join(output_dir, f"{ticker}_ground_truth.npy"), trues)
    print(f"\nPredictions saved to {output_dir}/")


def _evaluate_lightgbm(args):
    """Load and evaluate a LightGBM forecaster. Returns (preds, trues, target_names)."""
    from train import _load_raw_df

    ma_targets = args.ma_targets or []
    df, train_end, val_end, target_features = _load_raw_df(
        args.ticker, args.data_root, ma_targets,
    )

    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = os.path.join(
            args.checkpoint_dir,
            f"{args.ticker}_LightGBM_best.joblib",
        )

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Run: python train.py --model LightGBM first."
        )

    forecaster = LightGBMForecaster.load(checkpoint_path)

    history_start = max(0, val_end - forecaster.seq_len - 30)
    df_eval = df.iloc[history_start:].reset_index(drop=True)

    preds = forecaster.predict(df_eval)
    trues = forecaster.get_ground_truth(df_eval)

    return preds, trues, forecaster.target_features


def _evaluate_pytorch(args):
    """Load and evaluate a PyTorch model. Returns (preds, trues, target_names)."""
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

    ma_targets = args.ma_targets or []

    print("Loading Test Data...")
    test_dataset = ParquetDataset(
        ticker=test_cfg.ticker,
        root_path=test_cfg.data_root,
        flag='test',
        seq_len=test_cfg.seq_len,
        pred_len=test_cfg.pred_len,
        ma_targets=ma_targets,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=test_cfg.batch_size,
        shuffle=False,
        drop_last=False,
    )

    print(f"Test samples: {len(test_dataset)}\n")

    model_cfg = get_model_config(
        test_cfg.model_name, test_cfg.seq_len, test_cfg.pred_len,
        enc_in=test_dataset.enc_in,
        c_out=test_dataset.c_out,
    )

    model = get_model(test_cfg.model_name, model_cfg).to(test_cfg.device)

    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        checkpoint_path = os.path.join(
            test_cfg.checkpoint_dir,
            f"{test_cfg.ticker}_{test_cfg.model_name}_best.pt"
        )

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Run train.py first to train the model."
        )

    load_checkpoint(model, checkpoint_path, test_cfg.device)

    print("\nEvaluating on Test Set...")
    print("-" * 60)

    criterion = nn.MSELoss()
    _, preds, trues = evaluate(
        model, test_loader, criterion, test_cfg.device, test_dataset
    )

    return preds, trues, test_dataset.target_features


def main():
    args = parse_args()

    print("\n" + "=" * 60)
    print("Stock Price Forecasting - Evaluation")
    print("=" * 60)
    print(f"Ticker: {args.ticker}")
    print(f"Model: {args.model}")
    print(f"Lookback: {args.seq_len} days | Forecast: {args.pred_len} days")
    if args.ma_targets:
        print(f"MA targets: {args.ma_targets}")
    print("=" * 60 + "\n")

    # Get predictions (model-specific loading)
    if args.model == "LightGBM":
        preds, trues, target_names = _evaluate_lightgbm(args)
    else:
        preds, trues, target_names = _evaluate_pytorch(args)

    # Shared: compute metrics, print, save
    results = {"overall": calculate_metrics(preds, trues)}
    for i, name in enumerate(target_names):
        results[name] = calculate_metrics(preds[:, :, i], trues[:, :, i])

    print_results(results, target_names)

    if args.save_predictions:
        save_predictions(args.ticker, preds, trues)

    return results


if __name__ == "__main__":
    main()
