"""Console reporting for training and evaluation runs.

Kept separate from compute so metric-producing code stays testable
(it returns dicts) and formatting can evolve independently.
"""

import json
import os
from datetime import datetime
from typing import Iterable, List


def print_training_config(train_cfg, model_cfg,
                          target_names: List[str],
                          tickers: List[str] | None = None) -> None:
    """Print the header block shown at the start of a training run."""
    print("\n" + "=" * 60)
    print("Stock Price Forecasting - Training Configuration")
    print("=" * 60)
    if tickers and len(tickers) > 1:
        print(f"Tickers: {', '.join(tickers)} (pooled)")
    else:
        print(f"Ticker: {train_cfg.ticker}")
    print(f"Model: {train_cfg.model_name}")
    print(f"Device: {train_cfg.device}")
    targets_str = ", ".join(target_names)
    print(f"\nTask: Predict {targets_str}")
    print(f"  Input:  {model_cfg.enc_in} features")
    print(f"  Output: {targets_str} ({model_cfg.c_out} features)")
    print(f"  Lookback: {model_cfg.seq_len} days")
    print(f"  Forecast: {model_cfg.pred_len} days")
    print("\nTraining:")
    print(f"  Batch size: {train_cfg.batch_size}")
    print(f"  Epochs: {train_cfg.epochs}")
    print(f"  Learning rate: {train_cfg.learning_rate}")
    print(f"  Patience: {train_cfg.patience}")
    print("=" * 60 + "\n")


def print_epoch(epoch: int, total_epochs: int, train_loss: float,
                val_loss: float, lr: float, elapsed: float) -> None:
    """Single-line epoch summary during training."""
    print(f"Epoch {epoch:3d}/{total_epochs} | "
          f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
          f"LR: {lr:.2e} | Time: {elapsed:.1f}s")


def print_test_results(ticker: str, model_name: str, results: dict,
                       target_names: Iterable[str]) -> None:
    """Compact per-ticker test summary used by main.py's `test` command."""
    print(f"\n{'='*60}")
    print(f"  TEST RESULTS — {ticker} / {model_name}")
    print(f"{'='*60}")
    print(f"  Overall MAE:  ${results['overall']['MAE']:.4f}")
    print(f"  Overall RMSE: ${results['overall']['RMSE']:.4f}")
    for name in target_names:
        print(f"  {name:8s} MAE:  ${results[name]['MAE']:.4f}  "
              f"RMSE: ${results[name]['RMSE']:.4f}")

    if "overall_returns" in results:
        print("\n  RETURN-BASED METRICS")
        rm = results["overall_returns"]
        print(f"  Overall IC:  {rm['IC']:.4f}  "
              f"RIC: {rm['RIC']:.4f}  DA: {rm['DA']:.2%}")
        for name in target_names:
            key = f"{name}_returns"
            if key in results:
                rm_t = results[key]
                print(f"  {name:8s} IC:  {rm_t['IC']:.4f}  "
                      f"RIC: {rm_t['RIC']:.4f}  DA: {rm_t['DA']:.2%}")
    print(f"{'='*60}")


def print_verbose_test_results(results: dict, target_names: List[str]) -> None:
    """Wider report emitted by the standalone-style evaluation flow."""
    print("\n" + "=" * 60)
    print("TEST RESULTS (Original Scale)")
    print("=" * 60)
    print("\nOverall:")
    print(f"  MSE:  {results['overall']['MSE']:.4f}")
    print(f"  MAE:  {results['overall']['MAE']:.4f}")
    print(f"  RMSE: {results['overall']['RMSE']:.4f}")
    for name in target_names:
        print(f"\n{name} Prediction:")
        print(f"  MSE:  {results[name]['MSE']:.4f}")
        print(f"  MAE:  {results[name]['MAE']:.4f}")
        print(f"  RMSE: {results[name]['RMSE']:.4f}")

    if "overall_returns" in results:
        rm = results["overall_returns"]
        print(f"\n{'=' * 60}")
        print("RETURN-BASED METRICS")
        print(f"{'=' * 60}")
        print("\nOverall:")
        print(f"  IC  (Pearson):  {rm['IC']:.4f}")
        print(f"  RIC (Spearman): {rm['RIC']:.4f}")
        print(f"  DA  (Direction): {rm['DA']:.2%}")
        for name in target_names:
            key = f"{name}_returns"
            if key in results:
                rm_t = results[key]
                print(f"\n{name}:")
                print(f"  IC:  {rm_t['IC']:.4f}")
                print(f"  RIC: {rm_t['RIC']:.4f}")
                print(f"  DA:  {rm_t['DA']:.2%}")
    print("=" * 60)


def save_run_metrics(results_dir: str, metrics: dict, config: dict) -> str:
    """Persist metrics + config as timestamped ``run_<ts>.json`` under results_dir."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_file = os.path.join(results_dir, f"run_{timestamp}.json")
    payload = {"timestamp": timestamp, "config": config, "metrics": metrics}
    with open(run_file, "w") as f:
        # default=str so numpy scalars / Timestamps don't crash json.dump
        json.dump(payload, f, indent=2, default=str)
    return run_file
