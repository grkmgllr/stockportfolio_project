"""
Unified entry point for the stock forecasting pipeline.

Thin CLI dispatcher — the real work lives under ``src/forecasting/``.

Usage:
    python main.py fetch --tickers AAPL MSFT GOOGL --start 2015-01-01
    # Train then test any model on the band target (--target range is default):
    python main.py train --model TimeMixer --tickers AAPL MSFT ... --epochs 40
    python main.py test  --model TimeMixer --tickers AAPL MSFT ...
    python main.py test  --model LightGBM  --tickers AAPL MSFT ... --folds 4   # walk-forward
    # Legacy price forecast:  add  --target price
    # StockMixer (cross-stock) always uses its own path via train/test.
"""

import argparse
import os
import sys

# src/ on sys.path so bare `from dataset import ...` etc. still work
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import numpy as np

from paths import (
    CHECKPOINTS_ROOT,
    DATA_ROOT,
    results_dir,
)
from reporting import print_test_results, save_run_metrics
from utils import calculate_metrics


# ─────────────────────────────────────────────────────────────────────
# Subcommands
# ─────────────────────────────────────────────────────────────────────

def cmd_fetch(args):
    """Download daily OHLCV + features from Yahoo Finance into data/raw/."""
    from scripts.fetch_data import UNIVERSES, fetch_universe

    if args.tickers:
        tickers = args.tickers
    elif getattr(args, "universe", None):
        tickers = UNIVERSES[args.universe]
    elif args.all:
        tickers = UNIVERSES["starter"]
    else:
        tickers = [args.ticker]
    fetch_universe(tickers, start=args.start, end=args.end)


def _resolve_tickers(args):
    """Pick the ticker list: explicit --tickers, else a named --universe,
    else the single --ticker. Shared by train / test / walkforward."""
    if getattr(args, "tickers", None):
        return args.tickers
    if getattr(args, "universe", None):
        from scripts.fetch_data import UNIVERSES
        return UNIVERSES[args.universe]
    return [args.ticker]


def cmd_train(args):
    """Train a model. --target range (band, default) or price (legacy).
    StockMixer always uses its own cross-stock path (via the price branch)."""
    if getattr(args, "target", "range") == "price" or args.model == "StockMixer":
        return _cmd_train_price(args)
    return _cmd_train_range(args)


def cmd_test(args):
    """Evaluate a trained model and print results. --target range (default) or
    price. --folds N runs walk-forward on the band target (LightGBM)."""
    if getattr(args, "folds", 1) > 1:
        return _range_walkforward(args, _resolve_tickers(args))
    if getattr(args, "target", "range") == "price" or args.model == "StockMixer":
        return _cmd_test_price(args)
    return _cmd_test_range(args)


def _cmd_train_price(args):
    """Legacy price target: predict next-pred_len High/Close as % returns."""
    from forecasting import lightgbm_runner, pytorch_runner

    tickers = _resolve_tickers(args)
    ma_targets = args.ma_targets or []
    # Walk-forward folds set these; a normal run leaves them None (ratio split).
    fold_kw = dict(
        end_date=getattr(args, "end_date", None),
        train_end_date=getattr(args, "train_end_date", None),
        val_end_date=getattr(args, "val_end_date", None),
    )

    if args.model == "LightGBM":
        if len(tickers) > 1:
            lightgbm_runner.train_pooled(
                tickers, args.data_root, ma_targets,
                seq_len=args.seq_len, pred_len=args.pred_len,
                patience=args.patience, start_date=args.start_date, **fold_kw,
            )
        else:
            lightgbm_runner.train_one(
                tickers[0], args.data_root, ma_targets,
                seq_len=args.seq_len, pred_len=args.pred_len,
                patience=args.patience, start_date=args.start_date, **fold_kw,
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
            market_dim=getattr(args, "market_dim", 2),
            seed=getattr(args, "seed", 42),
            data_root=args.data_root, start_date=args.start_date, **fold_kw,
        )
    else:
        pytorch_runner.train(
            tickers, args.model, ma_targets,
            seq_len=args.seq_len, pred_len=args.pred_len,
            epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, patience=args.patience,
            data_root=args.data_root, start_date=args.start_date, **fold_kw,
        )


def _cmd_test_price(args):
    """Legacy price target evaluation (MAE / IC / RIC / DA on High/Close)."""
    from forecasting import lightgbm_runner, pytorch_runner

    tickers = _resolve_tickers(args)
    model_name = args.model
    ma_targets = args.ma_targets or []
    is_pooled = len(tickers) > 1
    fold_kw = dict(
        end_date=getattr(args, "end_date", None),
        train_end_date=getattr(args, "train_end_date", None),
        val_end_date=getattr(args, "val_end_date", None),
    )

    checkpoint_override = None
    if is_pooled:
        # LightGBM serialises to joblib; the neural models to .pt
        ext = "joblib" if model_name == "LightGBM" else "pt"
        checkpoint_override = os.path.join(
            CHECKPOINTS_ROOT, f"pooled_{model_name}_best.{ext}",
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
            market_dim=getattr(args, "market_dim", 2),
            checkpoint_override=checkpoint_override,
            start_date=args.start_date, **fold_kw,
        )

    all_results = {}
    for ticker in tickers:
        if is_pooled:
            print(f"\n{'─' * 60}")
            print(f"  Evaluating {ticker}")
            print(f"{'─' * 60}")

        eval_results = None
        if model_name == "LightGBM":
            preds, trues, target_names, eval_results = lightgbm_runner.evaluate(
                ticker, args.data_root, ma_targets,
                seq_len=args.seq_len, pred_len=args.pred_len,
                checkpoint_override=checkpoint_override,
                start_date=args.start_date, **fold_kw,
            )
        elif model_name == "StockMixer":
            preds, trues, target_names, eval_results = crossstock_results[ticker]
        else:
            preds, trues, target_names, eval_results = pytorch_runner.evaluate(
                ticker, model_name, ma_targets,
                seq_len=args.seq_len, pred_len=args.pred_len,
                batch_size=args.batch_size, data_root=args.data_root,
                checkpoint_override=checkpoint_override,
                start_date=args.start_date, **fold_kw,
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


def _build_folds(dates, n_folds: int, test_size: int):
    """Cut the tail of the calendar into consecutive, non-overlapping folds.

    Fold k tests on its own block of ``test_size`` trading days; the block
    immediately before it is that fold's validation set, and everything
    earlier is training (so the train window expands with each fold).
    Returns a list of {train_end_date, val_end_date, end_date} dicts.
    """
    n = len(dates)
    folds = []
    for k in range(n_folds):
        test_start = n - (n_folds - k) * test_size
        test_end = test_start + test_size
        val_start = test_start - test_size
        folds.append({
            "fold": k + 1,
            "train_end_date": dates[val_start - 1],
            "val_end_date": dates[test_start - 1],
            "end_date": dates[test_end - 1],
        })
    return folds


def _cmd_train_range(args):
    """Train the vol-normalised upside/downside band and save a checkpoint.

    target per day t (forward-averaged, Close[t] anchor, sigma=volatility_20):
      upside   = (mean(High[t+1..t+H]) - Close[t]) / Close[t] / sigma[t]
      downside = (mean(Low [t+1..t+H]) - Close[t]) / Close[t] / sigma[t]
    """
    tickers = _resolve_tickers(args)
    print(f"\n{'#' * 64}")
    print(f"  TRAIN (band) — {args.model} | {len(tickers)} ticker(s) | "
          f"train<={args.train_end_date} val<={args.val_end_date}")
    print(f"{'#' * 64}")
    if args.model == "LightGBM":
        from forecasting import lightgbm_runner as R
        R.train_pooled_range(
            tickers, args.data_root, args.seq_len, args.pred_len,
            start_date=args.start_date, train_end_date=args.train_end_date,
            val_end_date=args.val_end_date)
    else:
        from forecasting import pytorch_runner as P
        P.train(
            tickers, args.model, [], seq_len=args.seq_len, pred_len=args.pred_len,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            patience=args.patience, data_root=args.data_root,
            start_date=args.start_date, train_end_date=args.train_end_date,
            val_end_date=args.val_end_date, target_mode="range")


def _range_eval(args, tickers, end_date=None):
    """Load the trained band model and return pooled (preds, trues, dates).

    Neural test sample i and LightGBM test row i map to the same df date, so
    cross-sectional IC is computed on identical stock-day groupings across models.
    """
    if args.model == "LightGBM":
        from forecasting import lightgbm_runner as R
        from models.LightGBMForecaster import LightGBMForecaster
        ckpt = os.path.join(CHECKPOINTS_ROOT, "pooled_LightGBM_range_best.joblib")
        if not os.path.exists(ckpt):
            raise SystemExit(f"No checkpoint at {ckpt}. Run `train` first.")
        fc = LightGBMForecaster.load(ckpt)
        return R.evaluate_range_pooled(
            tickers, args.data_root, fc, start_date=args.start_date,
            train_end_date=args.train_end_date, val_end_date=args.val_end_date,
            end_date=end_date)

    from forecasting import pytorch_runner as P
    from forecasting.data_loading import load_raw_df
    parts_p, parts_t, parts_d = [], [], []
    for t in tickers:
        p, tr = P.evaluate_range(
            t, args.model, seq_len=args.seq_len, pred_len=args.pred_len,
            batch_size=args.batch_size, data_root=args.data_root,
            start_date=args.start_date, train_end_date=args.train_end_date,
            val_end_date=args.val_end_date, end_date=end_date)
        df, _te, val_end, _ = load_raw_df(
            t, args.data_root, ma_targets=[], start_date=args.start_date,
            end_date=end_date, train_end_date=args.train_end_date,
            val_end_date=args.val_end_date)
        d = df["Date"].to_numpy()[args.seq_len:len(df) - args.pred_len]
        s = val_end - 1
        d = d[s:s + len(p)]
        parts_p.append(p); parts_t.append(tr); parts_d.append(d)
    return (np.concatenate(parts_p), np.concatenate(parts_t),
            np.concatenate(parts_d))


def _cmd_test_range(args):
    """Evaluate the trained band model and print the full metric table."""
    tickers = _resolve_tickers(args)
    print(f"\n{'#' * 64}")
    print(f"  TEST (band) — {args.model} | {len(tickers)} ticker(s)")
    print(f"{'#' * 64}")
    preds, trues, dates = _range_eval(args, tickers)
    _print_range_metrics(preds, trues, dates, args.pred_len)
    return preds, trues


def _print_range_metrics(preds, trues, dates, pred_len):
    """Full band metric table: per channel IC, non-overlap IC, MAE, RMSE,
    baseMAE, DA%, up% — plus cross-sectional IC. Identical for every model."""
    def _ic(a, b):
        a, b = a.ravel(), b.ravel()
        m = np.isfinite(a) & np.isfinite(b)
        return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 3 else float("nan")

    def _err(p, t):
        m = np.isfinite(p) & np.isfinite(t)
        p, t = p[m], t[m]
        return (float(np.mean(np.abs(p - t))),
                float(np.sqrt(np.mean((p - t) ** 2))),
                float(np.mean(np.abs(t - t.mean()))))

    idx = np.arange(0, len(preds), pred_len)   # non-overlap sanity
    print(f"\n  {'channel':<12}{'IC':>8}{'IC_no':>8}{'MAE':>8}{'RMSE':>8}"
          f"{'baseMAE':>9}{'DA%':>7}{'up%':>7}  (test n={len(preds)})")
    chans = [("upside", preds[:, 0], trues[:, 0]),
             ("downside", preds[:, 1], trues[:, 1]),
             ("net(up+dn)", preds[:, 0] + preds[:, 1], trues[:, 0] + trues[:, 1])]
    for nm, p, t in chans:
        mae, rmse, base = _err(p, t)
        da = 100 * np.mean(np.sign(p) == np.sign(t)); up = 100 * np.mean(t > 0)
        print(f"  {nm:<12}{_ic(p, t):>8.3f}{_ic(p[idx], t[idx]):>8.3f}"
              f"{mae:>8.3f}{rmse:>8.3f}{base:>9.3f}{da:>7.1f}{up:>7.1f}")
    print(f"\n  upside IC = vol-adjusted 'more-than-expected upside' skill; "
          f"baseMAE = predict-the-mean baseline.")
    if dates is not None:
        _cross_sectional_ic(preds, trues, dates)


def _cross_sectional_ic(preds, trues, dates, min_names=10):
    """Print per-day cross-sectional IC (Spearman rank IC) + ICIR for each
    channel. IC_t = corr across stocks on day t; reported as mean ± std over
    days, with ICIR = mean/std (annualised-style information ratio)."""
    import pandas as pd

    def _spearman(a, b):
        ra = pd.Series(a).rank().to_numpy()
        rb = pd.Series(b).rank().to_numpy()
        if ra.std() == 0 or rb.std() == 0:
            return np.nan
        return float(np.corrcoef(ra, rb)[0, 1])

    print(f"\n  {'channel':<12}{'xs-IC':>8}{'xs-std':>8}{'ICIR':>7}{'days':>7}"
          f"  (>= {min_names} names/day)")
    for i, nm in enumerate(["upside", "downside"]):
        df = pd.DataFrame({"d": dates, "p": preds[:, i], "t": trues[:, i]})
        ics = []
        for _, g in df.groupby("d"):
            if len(g) >= min_names:
                ic = _spearman(g["p"].to_numpy(), g["t"].to_numpy())
                if np.isfinite(ic):
                    ics.append(ic)
        ics = np.array(ics)
        mu, sd = np.nanmean(ics), np.nanstd(ics)
        icir = mu / sd if sd > 0 else float("nan")
        print(f"  {nm:<12}{mu:>8.3f}{sd:>8.3f}{icir:>7.2f}{len(ics):>7}")
    print(f"\n  xs-IC = daily cross-sectional rank IC (Spearman); "
          f"ICIR = mean/std over days.")


def _range_fold_ic(preds, trues, dates, pred_len):
    """Return a dict of pooled + cross-sectional IC for one fold's test set."""
    import pandas as pd

    def _ic(a, b):
        a, b = a.ravel(), b.ravel()
        m = np.isfinite(a) & np.isfinite(b)
        return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 3 else float("nan")

    def _xs(i, min_names=10):
        df = pd.DataFrame({"d": dates, "p": preds[:, i], "t": trues[:, i]})
        ics = []
        for _, g in df.groupby("d"):
            if len(g) >= min_names and g["p"].std() and g["t"].std():
                ra, rb = g["p"].rank().to_numpy(), g["t"].rank().to_numpy()
                ics.append(np.corrcoef(ra, rb)[0, 1])
        return float(np.nanmean(ics)) if ics else float("nan")

    return {
        "n": len(preds),
        "up_ic": _ic(preds[:, 0], trues[:, 0]),
        "dn_ic": _ic(preds[:, 1], trues[:, 1]),
        "net_ic": _ic(preds[:, 0] + preds[:, 1], trues[:, 0] + trues[:, 1]),
        "xs_up": _xs(0),
        "xs_dn": _xs(1),
    }


def _range_walkforward(args, tickers):
    """Walk-forward on the band target: repeat train+eval over consecutive test
    blocks (expanding train window) and report mean ± std. Works for any model
    (neural is slow — one full training per fold)."""
    import copy
    import pandas as pd

    csv_path = os.path.join(args.data_root, f"{tickers[0]}.csv")
    dates = pd.read_csv(csv_path)["Date"].tolist()
    if args.start_date:
        dates = [d for d in dates if d >= args.start_date]
    needed = (args.folds + 1) * args.test_size + args.seq_len + args.pred_len
    if len(dates) < needed:
        raise SystemExit(
            f"Not enough data: {len(dates)} rows but {args.folds} folds of "
            f"--test_size {args.test_size} need ~{needed}.")
    folds = _build_folds(dates, args.folds, args.test_size)

    print(f"\n{'#' * 64}")
    print(f"  WALK-FORWARD (band) — {args.model} | {len(tickers)} ticker(s) | "
          f"{args.folds} folds x {args.test_size} days")
    print(f"{'#' * 64}")

    rows = []
    for f in folds:
        print(f"\n  Fold {f['fold']}/{args.folds}  train<={f['train_end_date']}  "
              f"val<={f['val_end_date']}  test<={f['end_date']}")
        a = copy.copy(args)
        a.train_end_date = f["train_end_date"]
        a.val_end_date = f["val_end_date"]
        _cmd_train_range(a)
        preds, trues, d = _range_eval(a, tickers, end_date=f["end_date"])
        m = _range_fold_ic(preds, trues, d, args.pred_len)
        rows.append(m)
        print(f"    up_ic={m['up_ic']:.3f}  dn_ic={m['dn_ic']:.3f}  "
              f"net_ic={m['net_ic']:.3f}  xs_up={m['xs_up']:.3f}  "
              f"xs_dn={m['xs_dn']:.3f}  (n={m['n']})")

    print(f"\n{'=' * 64}\n  SUMMARY (mean ± std over {args.folds} folds)\n{'=' * 64}")
    for key, label in [("up_ic", "upside IC"), ("dn_ic", "downside IC"),
                       ("net_ic", "net IC"), ("xs_up", "xs-IC upside"),
                       ("xs_dn", "xs-IC downside")]:
        vals = np.array([r[key] for r in rows], dtype=float)
        print(f"  {label:<16}{np.nanmean(vals):>8.3f}  ± {np.nanstd(vals):.3f}")
    return rows



# ─────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Stock Forecasting Pipeline — unified entry point",
    )
    subparsers = parser.add_subparsers(dest="command", help="Pipeline commands")

    # --- fetch ---
    p_fetch = subparsers.add_parser("fetch", help="Download daily OHLCV + features from Yahoo Finance")
    p_fetch.add_argument("--ticker", type=str, default="AAPL")
    p_fetch.add_argument("--tickers", nargs="+", default=None,
                         help="Explicit ticker list (e.g. AAPL MSFT GOOGL)")
    p_fetch.add_argument("--all", action="store_true",
                         help="Fetch the full starter universe (see fetch_data.STARTER_UNIVERSE)")
    p_fetch.add_argument("--universe", choices=["starter", "ndx100"], default=None,
                         help="Named universe to fetch: 'starter' (15) or 'ndx100' (~100).")
    p_fetch.add_argument("--start", type=str, default="2015-01-01")
    p_fetch.add_argument("--end", type=str, default="2025-11-30")

    # --- shared args for model commands ---
    def add_common_args(p):
        p.add_argument("--ticker", type=str, default="AAPL")
        p.add_argument("--tickers", nargs="+", default=None,
                       help="Multiple tickers for pooled training/eval (e.g. AAPL MSFT GOOGL)")
        p.add_argument("--universe", choices=["starter", "ndx100"], default=None,
                       help="Use a named ticker universe instead of --tickers "
                            "(starter=15, ndx100=~94). --tickers overrides this.")
        p.add_argument("--model", type=str, default="LightGBM",
                       choices=["LightGBM", "TimesNet", "TimeMixer", "StockMixer"])
        p.add_argument("--target", type=str, default="range",
                       choices=["range", "price"],
                       help="range = vol-normalised upside/downside band (default); "
                            "price = legacy High/Close return forecast. "
                            "StockMixer always uses its own cross-stock path.")
        p.add_argument("--seq_len", type=int, default=30)
        p.add_argument("--pred_len", type=int, default=5)
        p.add_argument("--data_root", type=str, default=DATA_ROOT)
        p.add_argument("--start_date", type=str, default="2015-01-01",
                       help="Train/test on rows from this ISO date onward.")
        p.add_argument("--train_end_date", type=str, default="2023-11-24",
                       help="Band target: last train date (date-based split).")
        p.add_argument("--val_end_date", type=str, default="2024-05-28",
                       help="Band target: last validation date; test is after.")
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
    p_train.add_argument("--market_dim", type=int, default=2,
                         help="Cross-stock hidden dimension m (StockMixer only). "
                              "Default 2 chosen via seed-averaged ablation on our 5-ticker universe; "
                              "sweep to re-tune for other universes.")
    p_train.add_argument("--seed", type=int, default=42,
                         help="Random seed for reproducibility (StockMixer only).")

    # --- test ---
    p_test = subparsers.add_parser(
        "test", help="Evaluate a trained model (full metric table). "
                     "--folds N runs walk-forward on the band target.")
    add_common_args(p_test)
    p_test.add_argument("--batch_size", type=int, default=256)
    p_test.add_argument("--market_dim", type=int, default=2,
                        help="Must match the value used at train time (StockMixer only).")
    # Walk-forward (--folds > 1): retrains + evaluates over consecutive test
    # blocks. Neural needs the training knobs below (one training per fold).
    p_test.add_argument("--folds", type=int, default=1,
                        help="Walk-forward folds (1 = single split, the default).")
    p_test.add_argument("--test_size", type=int, default=126,
                        help="Trading days per test (and val) block; 126 ~ 6 months.")
    p_test.add_argument("--epochs", type=int, default=40)
    p_test.add_argument("--lr", type=float, default=2e-4)
    p_test.add_argument("--patience", type=int, default=15)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    commands = {
        "fetch": cmd_fetch,
        "train": cmd_train,
        "test": cmd_test,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
