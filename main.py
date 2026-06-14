"""
Unified entry point for the stock forecasting pipeline.

Usage:
    python main.py resample --ticker AAPL
    python main.py resample --all
    python main.py train --model LightGBM --ticker AAPL
    python main.py test --model LightGBM --ticker AAPL
    python main.py meta-label --ticker AAPL
    python main.py train-meta --ticker AAPL
    python main.py evaluate --ticker AAPL
    python main.py run-all --ticker AAPL --model LightGBM
"""

import argparse
import os
import sys
import json
import time
from datetime import datetime

# Add src/ to Python path so internal imports work unchanged
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from typing import List

ALL_TICKERS = ["AAPL", "NVDA", "META", "GOOGL", "MSFT"]

RESULTS_ROOT = "results"
CHECKPOINTS_ROOT = "checkpoints"
DATA_ROOT = "data/raw"
META_ROOT = "data/meta"


def _results_dir(ticker: str, model: str) -> str:
    """Return ``results/{ticker}/{model}/`` and create it if missing."""
    path = os.path.join(RESULTS_ROOT, ticker, model)
    os.makedirs(path, exist_ok=True)
    return path


def _save_run_metrics(results_dir: str, metrics: dict, config: dict) -> str:
    """Save metrics + config as a timestamped JSON for later comparison."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_file = os.path.join(results_dir, f"run_{timestamp}.json")
    payload = {"timestamp": timestamp, "config": config, "metrics": metrics}
    with open(run_file, "w") as f:
        # default=str so numpy scalars / Timestamps don't crash json.dump
        json.dump(payload, f, indent=2, default=str)
    return run_file


# ─────────────────────────────────────────────────────────────────────
# Subcommands
# ─────────────────────────────────────────────────────────────────────

def cmd_resample(args):
    """Resample minute-bar parquet files to daily CSVs in data/raw/."""
    from scripts.resample_parquet import resample_minute_to_daily, find_parquet_file
    import pandas as pd

    tickers = ALL_TICKERS if args.all else [args.ticker]

    for ticker in tickers:
        print(f"\n{'='*60}")
        print(f"  Resampling {ticker} (start_date={args.start_date})")
        print(f"{'='*60}")

        try:
            parquet_path = find_parquet_file(args.data_root, ticker)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue

        df_raw = pd.read_parquet(parquet_path)
        print(f"  Loaded {len(df_raw):,} minute bars")

        df_daily = resample_minute_to_daily(df_raw, start_date=args.start_date)

        out_path = os.path.join(args.data_root, f"{ticker}.csv")
        df_daily.to_csv(out_path, index=False)
        print(f"  -> {len(df_daily)} daily bars saved to {out_path}")

    print("\nResample complete.")


def cmd_train(args):
    """Train the primary forecasting model (LightGBM or PyTorch)."""
    tickers = args.tickers if hasattr(args, 'tickers') and args.tickers else [args.ticker]
    model_name = args.model
    ma_targets = args.ma_targets or []

    if model_name == "LightGBM":
        # LightGBM trains one model per ticker
        for t in tickers:
            _train_lightgbm(args, t, ma_targets)
    else:
        # neural models support pooled multi-ticker training
        _train_pytorch(args, tickers, model_name, ma_targets)


def _train_lightgbm(args, ticker: str, ma_targets: List[str]):
    """Fit a LightGBM forecaster for one ticker and save checkpoint."""
    from models.LightGBMForecaster import LightGBMForecaster
    from train import _load_raw_df

    df, train_end, val_end, target_features = _load_raw_df(
        ticker, args.data_root, ma_targets,
    )

    df_train = df.iloc[:train_end].copy()
    df_val = df.iloc[train_end:val_end].copy()

    forecaster = LightGBMForecaster(
        seq_len=args.seq_len,
        pred_len=args.pred_len,
    )

    forecaster.fit(
        df_train, df_val,
        target_features=target_features,
        early_stopping_rounds=50,
    )

    os.makedirs(CHECKPOINTS_ROOT, exist_ok=True)
    checkpoint_path = os.path.join(CHECKPOINTS_ROOT, f"{ticker}_LightGBM_best.joblib")
    forecaster.save(checkpoint_path)

    print(f"\nTop-10 Feature Importance (gain):")
    for i, (name, score) in enumerate(forecaster.feature_importance().items()):
        if i >= 10:
            break
        print(f"  {name:25s}: {score:.1f}")


def _train_pytorch(args, tickers: List[str], model_name: str, ma_targets: List[str]):
    """Delegate PyTorch training to train.main() via sys.argv rewrite."""
    sys.argv = [
        "train.py",
        "--model", model_name,
        "--seq_len", str(args.seq_len),
        "--pred_len", str(args.pred_len),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--data_root", args.data_root,
    ]
    if len(tickers) > 1:
        sys.argv += ["--tickers"] + tickers
    else:
        sys.argv += ["--ticker", tickers[0]]
    if ma_targets:
        sys.argv += ["--ma_targets"] + ma_targets

    from train import main as train_main
    train_main()


def cmd_test(args):
    """Evaluate a trained model on the test split and save .npy predictions."""
    import numpy as np
    from utils import calculate_metrics, calculate_return_metrics

    tickers = args.tickers if hasattr(args, 'tickers') and args.tickers else [args.ticker]
    model_name = args.model
    ma_targets = args.ma_targets or []
    is_pooled = len(tickers) > 1

    # For pooled models, checkpoint uses "pooled" prefix
    checkpoint_name = None
    if is_pooled:
        checkpoint_name = os.path.join(
            CHECKPOINTS_ROOT, f"pooled_{model_name}_best.pt"
        )

    all_results = {}
    for ticker in tickers:
        if is_pooled:
            print(f"\n{'─'*60}")
            print(f"  Evaluating {ticker}")
            print(f"{'─'*60}")

        eval_results = None
        if model_name == "LightGBM":
            preds, trues, target_names = _test_lightgbm(args, ticker, ma_targets)
        else:
            preds, trues, target_names, eval_results = _test_pytorch(
                args, ticker, model_name, ma_targets,
                checkpoint_override=checkpoint_name,
            )

        results = {"overall": calculate_metrics(preds, trues)}
        for i, name in enumerate(target_names):
            results[name] = calculate_metrics(preds[:, :, i], trues[:, :, i])

        if eval_results:
            for key in eval_results:
                if key.endswith('_returns'):
                    results[key] = eval_results[key]

        print(f"\n{'='*60}")
        print(f"  TEST RESULTS — {ticker} / {model_name}")
        print(f"{'='*60}")
        print(f"  Overall MAE:  ${results['overall']['MAE']:.4f}")
        print(f"  Overall RMSE: ${results['overall']['RMSE']:.4f}")
        for name in target_names:
            print(f"  {name:8s} MAE:  ${results[name]['MAE']:.4f}  RMSE: ${results[name]['RMSE']:.4f}")

        if 'overall_returns' in results:
            print(f"\n  RETURN-BASED METRICS")
            rm = results['overall_returns']
            print(f"  Overall IC:  {rm['IC']:.4f}  RIC: {rm['RIC']:.4f}  DA: {rm['DA']:.2%}")
            for name in target_names:
                key = f'{name}_returns'
                if key in results:
                    rm_t = results[key]
                    print(f"  {name:8s} IC:  {rm_t['IC']:.4f}  RIC: {rm_t['RIC']:.4f}  DA: {rm_t['DA']:.2%}")
        print(f"{'='*60}")

        out_dir = _results_dir(ticker, model_name)
        np.save(os.path.join(out_dir, "predictions.npy"), preds)
        np.save(os.path.join(out_dir, "ground_truth.npy"), trues)
        np.save(os.path.join(RESULTS_ROOT, f"{ticker}_predictions.npy"), preds)
        np.save(os.path.join(RESULTS_ROOT, f"{ticker}_ground_truth.npy"), trues)

        config = {"ticker": ticker, "model": model_name, "seq_len": args.seq_len, "pred_len": args.pred_len}
        run_file = _save_run_metrics(out_dir, results, config)
        print(f"  Metrics saved: {run_file}")

        all_results[ticker] = results

    return all_results


def _test_lightgbm(args, ticker: str, ma_targets: List[str]):
    """Load saved LightGBM forecaster and predict on the test window."""
    from models.LightGBMForecaster import LightGBMForecaster
    from train import _load_raw_df

    df, train_end, val_end, target_features = _load_raw_df(
        ticker, args.data_root, ma_targets,
    )

    checkpoint_path = os.path.join(CHECKPOINTS_ROOT, f"{ticker}_LightGBM_best.joblib")
    forecaster = LightGBMForecaster.load(checkpoint_path)

    # Predict on full dataframe, then slice to match the alignment
    # expected by generate_meta_labels.py.
    #
    # preds_full[j] corresponds to df row (seq_len + j).
    # generate_meta_labels.py expects entry bar at
    #   val_end + i + seq_len - 1   for test sample i.
    # Setting (seq_len + j) = (val_end + i + seq_len - 1) => j = val_end + i - 1.
    # So test predictions start at j = val_end - 1 in preds_full.
    preds_full = forecaster.predict(df)
    trues_full = forecaster.get_ground_truth(df)

    n_test = (len(df) - val_end) - forecaster.seq_len - forecaster.pred_len + 1
    test_start = val_end - 1
    preds = preds_full[test_start : test_start + n_test]
    trues = trues_full[test_start : test_start + n_test]

    return preds, trues, forecaster.target_features


def _test_pytorch(args, ticker: str, model_name: str, ma_targets: List[str],
                   checkpoint_override: str | None = None):
    """Load saved PyTorch model and evaluate on one ticker's test split."""
    import torch
    from torch.utils.data import DataLoader
    from dataset import ParquetDataset
    from utils import load_checkpoint

    test_dataset = ParquetDataset(
        ticker=ticker,
        root_path=args.data_root,
        flag='test',
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        ma_targets=ma_targets,
        return_targets=True,
    )

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    from test import get_model_config, get_model, evaluate
    model_cfg = get_model_config(model_name, args.seq_len, args.pred_len,
                                  enc_in=test_dataset.enc_in, c_out=test_dataset.c_out,
                                  denorm_indices=test_dataset.denorm_indices,
                                  return_targets=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model(model_name, model_cfg).to(device)

    checkpoint_path = checkpoint_override or os.path.join(
        CHECKPOINTS_ROOT, f"{ticker}_{model_name}_best.pt"
    )
    load_checkpoint(model, checkpoint_path, device)

    import torch.nn as nn
    criterion = nn.MSELoss()
    eval_results, preds, trues = evaluate(model, test_loader, criterion, device, test_dataset)

    return preds, trues, test_dataset.target_features, eval_results


def cmd_meta_label(args):
    """Generate triple-barrier meta-labels from primary model predictions."""
    sys.argv = [
        "generate_meta_labels.py",
        "--ticker", args.ticker,
        "--seq_len", str(args.seq_len),
        "--pred_len", str(args.pred_len),
        "--data_root", args.data_root,
    ]
    if args.ma_targets:
        sys.argv += ["--target_names"] + ["High", "Close"] + args.ma_targets

    from scripts.generate_meta_labels import main as gen_main
    gen_main()


def cmd_train_meta(args):
    """Train the LightGBM meta-classifier with Purged K-Fold CV."""
    sys.argv = [
        "train_meta.py",
        "--ticker", args.ticker,
        "--threshold", str(args.threshold),
    ]
    from train_meta import main as tm_main
    tm_main()


def cmd_evaluate(args):
    """Report precision/recall/F1/PSR before vs after meta-filtering."""
    from trading_logic.evaluation import full_evaluation, print_evaluation_report
    ticker = args.ticker

    meta_pred_path = os.path.join(META_ROOT, f"meta_predictions_{ticker}.csv")
    if not os.path.exists(meta_pred_path):
        print(f"No meta predictions found at {meta_pred_path}")
        print(f"Run: python main.py train-meta --ticker {ticker} first.")
        return

    import pandas as pd
    df = pd.read_csv(meta_pred_path)
    results = full_evaluation(df, threshold=args.threshold)
    print_evaluation_report(results, threshold=args.threshold)


def cmd_run_all(args):
    """Run the complete pipeline: train → test → meta-label → train-meta."""
    ticker = args.ticker
    model = args.model

    print(f"\n{'#'*60}")
    print(f"  FULL PIPELINE: {ticker} / {model}")
    print(f"{'#'*60}")

    t0 = time.time()

    print(f"\n[1/4] Training {model}...")
    cmd_train(args)

    print(f"\n[2/4] Testing {model}...")
    cmd_test(args)

    print(f"\n[3/4] Generating meta-labels...")
    cmd_meta_label(args)

    print(f"\n[4/4] Training meta-classifier...")
    cmd_train_meta(args)

    elapsed = time.time() - t0
    print(f"\n{'#'*60}")
    print(f"  PIPELINE COMPLETE ({elapsed:.1f}s)")
    print(f"{'#'*60}")


# ─────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Stock Forecasting Pipeline — unified entry point",
    )
    subparsers = parser.add_subparsers(dest="command", help="Pipeline commands")

    # --- resample ---
    p_resample = subparsers.add_parser("resample", help="Resample minute bars to daily")
    p_resample.add_argument("--ticker", type=str, default="AAPL")
    p_resample.add_argument("--all", action="store_true", help="Resample all tickers")
    p_resample.add_argument("--start_date", type=str, default="2015-10-01")
    p_resample.add_argument("--data_root", type=str, default=DATA_ROOT)

    # --- shared args for model commands ---
    def add_common_args(p):
        p.add_argument("--ticker", type=str, default="AAPL")
        p.add_argument("--tickers", nargs="+", default=None,
                       help="Multiple tickers for pooled training/eval (e.g. AAPL MSFT GOOGL)")
        p.add_argument("--model", type=str, default="LightGBM",
                       choices=["LightGBM", "TimesNet", "TimeMixer"])
        p.add_argument("--seq_len", type=int, default=30)
        p.add_argument("--pred_len", type=int, default=5)
        p.add_argument("--data_root", type=str, default=DATA_ROOT)
        p.add_argument("--ma_targets", nargs="*", default=None,
                       help="MA targets (e.g. EMA_20 SMA_50)")

    # --- train ---
    p_train = subparsers.add_parser("train", help="Train a forecasting model")
    add_common_args(p_train)
    p_train.add_argument("--epochs", type=int, default=100)
    p_train.add_argument("--batch_size", type=int, default=32)
    p_train.add_argument("--lr", type=float, default=3e-4)

    # --- test ---
    p_test = subparsers.add_parser("test", help="Evaluate a trained model")
    add_common_args(p_test)
    p_test.add_argument("--batch_size", type=int, default=32)

    # --- meta-label ---
    p_meta = subparsers.add_parser("meta-label", help="Generate triple-barrier meta-labels")
    p_meta.add_argument("--ticker", type=str, default="AAPL")
    p_meta.add_argument("--seq_len", type=int, default=30)
    p_meta.add_argument("--pred_len", type=int, default=5)
    p_meta.add_argument("--data_root", type=str, default=DATA_ROOT)
    p_meta.add_argument("--ma_targets", nargs="*", default=None)

    # --- train-meta ---
    p_tmeta = subparsers.add_parser("train-meta", help="Train the meta-classifier")
    p_tmeta.add_argument("--ticker", type=str, default="AAPL")
    p_tmeta.add_argument("--threshold", type=float, default=0.5)

    # --- evaluate ---
    p_eval = subparsers.add_parser("evaluate", help="Evaluate meta-labeling precision lift")
    p_eval.add_argument("--ticker", type=str, default="AAPL")
    p_eval.add_argument("--threshold", type=float, default=0.5)

    # --- run-all ---
    p_all = subparsers.add_parser("run-all", help="Run full pipeline (train→test→meta→evaluate)")
    add_common_args(p_all)
    p_all.add_argument("--epochs", type=int, default=100)
    p_all.add_argument("--batch_size", type=int, default=32)
    p_all.add_argument("--lr", type=float, default=3e-4)
    p_all.add_argument("--threshold", type=float, default=0.5)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    commands = {
        "resample": cmd_resample,
        "train": cmd_train,
        "test": cmd_test,
        "meta-label": cmd_meta_label,
        "train-meta": cmd_train_meta,
        "evaluate": cmd_evaluate,
        "run-all": cmd_run_all,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
