"""
Unified entry point for the stock forecasting pipeline.

This module is a thin CLI dispatcher — every command's real work lives
under ``src/forecasting/``, ``src/meta/``, or ``src/reporting.py``.

Usage:
    python main.py resample --ticker AAPL
    python main.py resample --all
    python main.py train --model TimeMixer --tickers AAPL MSFT GOOGL NVDA META
    python main.py test  --model TimeMixer --ticker  AAPL
    python main.py meta-label --ticker AAPL --model TimeMixer
    python main.py train-meta --ticker AAPL --model TimeMixer
    python main.py evaluate   --ticker AAPL --model TimeMixer
    python main.py run-all    --ticker AAPL --model LightGBM
"""

import argparse
import os
import sys
import time

# src/ on sys.path so bare `from dataset import ...` etc. still work
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np

from paths import (
    CHECKPOINTS_ROOT,
    DATA_ROOT,
    meta_predictions_path,
    results_dir,
)
from reporting import print_test_results, save_run_metrics
from utils import calculate_metrics


ALL_TICKERS = ["AAPL", "NVDA", "META", "GOOGL", "MSFT"]


# ─────────────────────────────────────────────────────────────────────
# Subcommands
# ─────────────────────────────────────────────────────────────────────

def cmd_resample(args):
    """Resample minute-bar parquet files to daily CSVs in data/raw/."""
    import pandas as pd
    from scripts.resample_parquet import find_parquet_file, resample_minute_to_daily

    tickers = ALL_TICKERS if args.all else [args.ticker]
    for ticker in tickers:
        print(f"\n{'=' * 60}")
        print(f"  Resampling {ticker} (start_date={args.start_date})")
        print(f"{'=' * 60}")

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
    from forecasting import lightgbm_runner, pytorch_runner

    tickers = args.tickers if getattr(args, "tickers", None) else [args.ticker]
    ma_targets = args.ma_targets or []

    if args.model == "LightGBM":
        if len(tickers) > 1:
            lightgbm_runner.train_pooled(
                tickers, args.data_root, ma_targets,
                seq_len=args.seq_len, pred_len=args.pred_len,
                patience=args.patience,
            )
        else:
            lightgbm_runner.train_one(
                tickers[0], args.data_root, ma_targets,
                seq_len=args.seq_len, pred_len=args.pred_len,
                patience=args.patience,
            )
    elif args.model == "StockMixer":
        # Cross-stock model — jointly trained on all tickers at once.
        if len(tickers) < 2:
            raise SystemExit(
                "StockMixer requires --tickers with at least 2 symbols "
                "(it is a cross-stock model)."
            )
        from forecasting import crossstock_runner
        crossstock_runner.train(
            tickers, args.model, ma_targets,
            seq_len=args.seq_len, pred_len=args.pred_len,
            epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, patience=args.patience,
            alpha=getattr(args, "alpha", 0.1),
            market_dim=getattr(args, "market_dim", 20),
            seed=getattr(args, "seed", 42),
            data_root=args.data_root,
        )
    else:
        pytorch_runner.train(
            tickers, args.model, ma_targets,
            seq_len=args.seq_len, pred_len=args.pred_len,
            epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, patience=args.patience,
            data_root=args.data_root,
        )


def cmd_test(args):
    """Evaluate a trained model on the test split and save .npy predictions."""
    from forecasting import lightgbm_runner, pytorch_runner

    tickers = args.tickers if getattr(args, "tickers", None) else [args.ticker]
    model_name = args.model
    ma_targets = args.ma_targets or []
    is_pooled = len(tickers) > 1

    checkpoint_override = None
    if is_pooled:
        checkpoint_override = os.path.join(
            CHECKPOINTS_ROOT, f"pooled_{model_name}_best.pt",
        )

    # Cross-stock models score every ticker in one forward pass, so we
    # pre-compute the results dict here and reuse it inside the per-ticker
    # loop below (mirroring the pytorch_runner.evaluate return shape).
    crossstock_results = None
    if model_name == "StockMixer":
        if len(tickers) < 2:
            raise SystemExit(
                "StockMixer test requires --tickers with at least 2 symbols."
            )
        from forecasting import crossstock_runner
        checkpoint_override = os.path.join(
            CHECKPOINTS_ROOT, f"crossstock_{model_name}_best.pt",
        )
        crossstock_results = crossstock_runner.evaluate(
            tickers, model_name, ma_targets,
            seq_len=args.seq_len, pred_len=args.pred_len,
            batch_size=args.batch_size, data_root=args.data_root,
            market_dim=getattr(args, "market_dim", 20),
            checkpoint_override=checkpoint_override,
        )

    all_results = {}
    for ticker in tickers:
        if is_pooled:
            print(f"\n{'─' * 60}")
            print(f"  Evaluating {ticker}")
            print(f"{'─' * 60}")

        eval_results = None
        if model_name == "LightGBM":
            preds, trues, target_names = lightgbm_runner.evaluate(
                ticker, args.data_root, ma_targets,
                seq_len=args.seq_len, pred_len=args.pred_len,
            )
        elif model_name == "StockMixer":
            preds, trues, target_names, eval_results = crossstock_results[ticker]
        else:
            preds, trues, target_names, eval_results = pytorch_runner.evaluate(
                ticker, model_name, ma_targets,
                seq_len=args.seq_len, pred_len=args.pred_len,
                batch_size=args.batch_size, data_root=args.data_root,
                checkpoint_override=checkpoint_override,
            )

        results = {"overall": calculate_metrics(preds, trues)}
        for i, name in enumerate(target_names):
            results[name] = calculate_metrics(preds[:, :, i], trues[:, :, i])

        if eval_results:
            for key, val in eval_results.items():
                if key.endswith("_returns"):
                    results[key] = val

        print_test_results(ticker, model_name, results, target_names)

        out_dir = results_dir(ticker, model_name)
        np.save(os.path.join(out_dir, "predictions.npy"), preds)
        np.save(os.path.join(out_dir, "ground_truth.npy"), trues)

        config = {
            "ticker": ticker, "model": model_name,
            "seq_len": args.seq_len, "pred_len": args.pred_len,
        }
        run_file = save_run_metrics(out_dir, results, config)
        print(f"  Metrics saved: {run_file}")

        all_results[ticker] = results

    return all_results


def cmd_meta_label(args):
    """Generate triple-barrier meta-labels from primary model predictions."""
    sys.argv = [
        "generate.py",
        "--ticker", args.ticker,
        "--model", args.model,
        "--seq_len", str(args.seq_len),
        "--pred_len", str(args.pred_len),
        "--data_root", args.data_root,
    ]
    if args.ma_targets:
        sys.argv += ["--target_names"] + ["High", "Close"] + args.ma_targets

    from meta.generate import main as gen_main
    gen_main()


def cmd_train_meta(args):
    """Train the LightGBM meta-classifier with Purged K-Fold CV."""
    sys.argv = [
        "train.py",
        "--ticker", args.ticker,
        "--model", args.model,
        "--threshold", str(args.threshold),
    ]
    from meta.train import main as tm_main
    tm_main()


def cmd_evaluate(args):
    """Report precision/recall/F1/PSR before vs after meta-filtering."""
    import pandas as pd
    from trading_logic.evaluation import full_evaluation, print_evaluation_report

    path = meta_predictions_path(args.ticker, args.model)
    if not os.path.exists(path):
        print(f"No meta predictions found at {path}")
        print(f"Run: python main.py train-meta --ticker {args.ticker} "
              f"--model {args.model} first.")
        return

    df = pd.read_csv(path)
    results = full_evaluation(df, threshold=args.threshold)
    print_evaluation_report(results, threshold=args.threshold)


def cmd_run_all(args):
    """Run the complete pipeline: train → test → meta-label → train-meta."""
    ticker = args.ticker
    model = args.model

    print(f"\n{'#' * 60}")
    print(f"  FULL PIPELINE: {ticker} / {model}")
    print(f"{'#' * 60}")

    t0 = time.time()
    print(f"\n[1/4] Training {model}...")
    cmd_train(args)

    print(f"\n[2/4] Testing {model}...")
    cmd_test(args)

    print(f"\n[3/4] Generating meta-labels...")
    cmd_meta_label(args)

    print(f"\n[4/4] Training meta-classifier...")
    cmd_train_meta(args)

    print(f"\n{'#' * 60}")
    print(f"  PIPELINE COMPLETE ({time.time() - t0:.1f}s)")
    print(f"{'#' * 60}")


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
                       choices=["LightGBM", "TimesNet", "TimeMixer", "StockMixer"])
        p.add_argument("--seq_len", type=int, default=30)
        p.add_argument("--pred_len", type=int, default=5)
        p.add_argument("--data_root", type=str, default=DATA_ROOT)
        p.add_argument("--ma_targets", nargs="*", default=None,
                       help="MA targets (e.g. EMA_20 SMA_50)")

    # --- train ---
    p_train = subparsers.add_parser("train", help="Train a forecasting model")
    add_common_args(p_train)
    p_train.add_argument("--epochs", type=int, default=200)
    p_train.add_argument("--batch_size", type=int, default=32)
    p_train.add_argument("--lr", type=float, default=2e-4)
    p_train.add_argument("--patience", type=int, default=30,
                         help="Early stopping patience (epochs without improvement)")
    p_train.add_argument("--alpha", type=float, default=0.1,
                         help="Rank-loss weight (StockMixer only). "
                              "0.0 disables the rank term, paper default is 0.1.")
    p_train.add_argument("--market_dim", type=int, default=20,
                         help="Cross-stock hidden dimension m (StockMixer only). "
                              "Paper uses 20 for NASDAQ; sweep to tune for your universe.")
    p_train.add_argument("--seed", type=int, default=42,
                         help="Random seed for reproducibility (StockMixer only).")

    # --- test ---
    p_test = subparsers.add_parser("test", help="Evaluate a trained model")
    add_common_args(p_test)
    p_test.add_argument("--batch_size", type=int, default=32)
    p_test.add_argument("--market_dim", type=int, default=20,
                        help="Must match the value used at train time (StockMixer only).")

    # --- meta-label ---
    p_meta = subparsers.add_parser("meta-label", help="Generate triple-barrier meta-labels")
    p_meta.add_argument("--ticker", type=str, default="AAPL")
    p_meta.add_argument("--model", type=str, default="LightGBM",
                        choices=["LightGBM", "TimesNet", "TimeMixer", "StockMixer"],
                        help="Primary forecaster whose predictions to label")
    p_meta.add_argument("--seq_len", type=int, default=30)
    p_meta.add_argument("--pred_len", type=int, default=5)
    p_meta.add_argument("--data_root", type=str, default=DATA_ROOT)
    p_meta.add_argument("--ma_targets", nargs="*", default=None)

    # --- train-meta ---
    p_tmeta = subparsers.add_parser("train-meta", help="Train the meta-classifier")
    p_tmeta.add_argument("--ticker", type=str, default="AAPL")
    p_tmeta.add_argument("--model", type=str, default="LightGBM",
                         choices=["LightGBM", "TimesNet", "TimeMixer", "StockMixer"],
                         help="Primary forecaster whose meta-labels to train on")
    p_tmeta.add_argument("--threshold", type=float, default=0.5)

    # --- evaluate ---
    p_eval = subparsers.add_parser("evaluate", help="Evaluate meta-labeling precision lift")
    p_eval.add_argument("--ticker", type=str, default="AAPL")
    p_eval.add_argument("--model", type=str, default="LightGBM",
                        choices=["LightGBM", "TimesNet", "TimeMixer", "StockMixer"],
                        help="Primary forecaster whose meta-predictions to evaluate")
    p_eval.add_argument("--threshold", type=float, default=0.5)

    # --- run-all ---
    p_all = subparsers.add_parser("run-all",
                                   help="Run full pipeline (train→test→meta→evaluate)")
    add_common_args(p_all)
    p_all.add_argument("--epochs", type=int, default=200)
    p_all.add_argument("--batch_size", type=int, default=32)
    p_all.add_argument("--lr", type=float, default=2e-4)
    p_all.add_argument("--patience", type=int, default=30,
                       help="Early stopping patience (epochs without improvement)")
    p_all.add_argument("--alpha", type=float, default=0.1,
                       help="Rank-loss weight (StockMixer only). "
                            "0.0 disables the rank term, paper default is 0.1.")
    p_all.add_argument("--market_dim", type=int, default=20,
                       help="Cross-stock hidden dimension m (StockMixer only). "
                            "Paper uses 20 for NASDAQ; sweep to tune for your universe.")
    p_all.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility (StockMixer only).")
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
