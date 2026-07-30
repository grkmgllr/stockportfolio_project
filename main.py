"""
Unified entry point for the stock forecasting pipeline.

Thin CLI dispatcher — the real work lives under ``src/forecasting/``.

Usage:
    python main.py fetch --tickers AAPL MSFT GOOGL --start 2015-01-01
    # Vol-normalised upside/downside band forecast (the Stage-1 target):
    python main.py range --model LightGBM  --tickers AAPL MSFT ... --start_date 2015-01-01
    python main.py range --model TimeMixer --tickers AAPL MSFT ... --epochs 40
    python main.py range --model LightGBM  --tickers AAPL MSFT ... --folds 4   # walk-forward
    # StockMixer (cross-stock return-ranking model, its own path):
    python main.py train --model StockMixer --tickers AAPL MSFT GOOGL
    python main.py test  --model StockMixer --tickers AAPL MSFT GOOGL
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
    """Train the primary forecasting model (LightGBM or PyTorch)."""
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


def cmd_test(args):
    """Evaluate a trained model on the test split and save .npy predictions."""
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


def cmd_range(args):
    """Stage-1 redesign: train + evaluate the vol-normalised upside/downside band.

    target per day t (forward-averaged, Close[t] anchor, sigma=volatility_20):
      upside   = (mean(High[t+1..t+H]) - Close[t]) / Close[t] / sigma[t]
      downside = (mean(Low [t+1..t+H]) - Close[t]) / Close[t] / sigma[t]
    Feeds the Stage-2 triple barrier (upper=upside, lower=downside).
    """
    tickers = _resolve_tickers(args)
    if getattr(args, "folds", 1) > 1:
        return _cmd_range_walkforward(args, tickers)

    print(f"\n{'#' * 64}")
    print(f"  RANGE FORECAST — {args.model} | {len(tickers)} ticker(s) | "
          f"train<={args.train_end_date} val<={args.val_end_date}")
    print(f"{'#' * 64}")

    if args.model == "LightGBM":
        from forecasting import lightgbm_runner as R
        fc = R.train_pooled_range(
            tickers, args.data_root, args.seq_len, args.pred_len,
            start_date=args.start_date, train_end_date=args.train_end_date,
            val_end_date=args.val_end_date,
        )
        preds, trues, dates = R.evaluate_range_pooled(
            tickers, args.data_root, fc, start_date=args.start_date,
            train_end_date=args.train_end_date, val_end_date=args.val_end_date,
        )
    else:
        # Neural (TimeMixer / TimesNet): same range target, fair comparison.
        from forecasting import pytorch_runner as P
        P.train(
            tickers, args.model, [], seq_len=args.seq_len, pred_len=args.pred_len,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            patience=args.patience, data_root=args.data_root,
            start_date=args.start_date, train_end_date=args.train_end_date,
            val_end_date=args.val_end_date, target_mode="range",
        )
        parts_p, parts_t = [], []
        for t in tickers:
            p, tr = P.evaluate_range(
                t, args.model, seq_len=args.seq_len, pred_len=args.pred_len,
                batch_size=args.batch_size, data_root=args.data_root,
                start_date=args.start_date, train_end_date=args.train_end_date,
                val_end_date=args.val_end_date,
            )
            parts_p.append(p); parts_t.append(tr)
        preds, trues = np.concatenate(parts_p), np.concatenate(parts_t)
        dates = None   # neural eval path does not yet track per-row dates

    def _ic(a, b):
        a, b = a.ravel(), b.ravel()
        m = np.isfinite(a) & np.isfinite(b)
        return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 3 else float("nan")

    idx = np.arange(0, len(preds), args.pred_len)   # non-overlap sanity
    print(f"\n  {'channel':<12}{'IC':>8}{'IC_no':>8}{'DA%':>7}{'up%':>7}  (test n={len(preds)})")
    for i, nm in enumerate(["upside", "downside"]):
        p, t = preds[:, i], trues[:, i]
        da = 100 * np.mean(np.sign(p) == np.sign(t)); up = 100 * np.mean(t > 0)
        print(f"  {nm:<12}{_ic(p, t):>8.3f}{_ic(p[idx], t[idx]):>8.3f}{da:>7.1f}{up:>7.1f}")
    np_, nt = preds[:, 0] + preds[:, 1], trues[:, 0] + trues[:, 1]
    da = 100 * np.mean(np.sign(np_) == np.sign(nt)); up = 100 * np.mean(nt > 0)
    print(f"  {'net(up+dn)':<12}{_ic(np_, nt):>8.3f}{_ic(np_[idx], nt[idx]):>8.3f}{da:>7.1f}{up:>7.1f}")
    print(f"\n  upside IC = vol-adjusted 'more-than-expected upside' skill.")

    # Cross-sectional IC: rank stocks against each other each day, then average
    # the daily rank correlations. This is the metric that reflects "pick the
    # right stock today" skill (and where cross-sectional features should show).
    if dates is not None:
        _cross_sectional_ic(preds, trues, dates)

    return preds, trues


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


def _cmd_range_walkforward(args, tickers):
    """Walk-forward on the range target: repeat train+eval over consecutive
    test blocks (expanding train window) and report mean ± std. LightGBM only."""
    import pandas as pd

    if args.model != "LightGBM":
        raise SystemExit(
            "Walk-forward range is LightGBM-only for now (neural training per "
            "fold is expensive). Use --folds 1 for the neural single-split.")

    from forecasting import lightgbm_runner as R

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
    print(f"  RANGE WALK-FORWARD — LightGBM | {len(tickers)} ticker(s) | "
          f"{args.folds} folds x {args.test_size} days")
    print(f"{'#' * 64}")

    rows = []
    for f in folds:
        print(f"\n  Fold {f['fold']}/{args.folds}  train<={f['train_end_date']}  "
              f"val<={f['val_end_date']}  test<={f['end_date']}")
        fc = R.train_pooled_range(
            tickers, args.data_root, args.seq_len, args.pred_len,
            start_date=args.start_date, train_end_date=f["train_end_date"],
            val_end_date=f["val_end_date"])
        preds, trues, d = R.evaluate_range_pooled(
            tickers, args.data_root, fc, start_date=args.start_date,
            train_end_date=f["train_end_date"], val_end_date=f["val_end_date"],
            end_date=f["end_date"])
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


def cmd_walkforward(args):
    """Re-run train+test over several sequential folds and report mean ± std.

    A single 70/15/15 split leaves one short test window, so differences
    between models sit well inside the noise band. Walk-forward repeats the
    full train/test cycle on consecutive test periods with an expanding train
    window, which is what makes a model comparison credible.
    """
    import copy
    import pandas as pd

    tickers = _resolve_tickers(args)

    csv_path = os.path.join(args.data_root, f"{tickers[0]}.csv")
    if not os.path.exists(csv_path):
        raise SystemExit(
            f"Data not found: {csv_path}\nRun `python main.py fetch` first."
        )
    dates = pd.read_csv(csv_path)["Date"].tolist()
    if args.start_date:
        dates = [d for d in dates if d >= args.start_date]

    # Each fold consumes one test block; the earliest fold also needs a
    # validation block plus enough history to form a window.
    needed = (args.folds + 1) * args.test_size + args.seq_len + args.pred_len
    if len(dates) < needed:
        raise SystemExit(
            f"Not enough data: {len(dates)} rows but {args.folds} folds of "
            f"--test_size {args.test_size} need ~{needed}. Use fewer folds, a "
            f"smaller --test_size, or an earlier --start_date."
        )

    folds = _build_folds(dates, args.folds, args.test_size)

    print(f"\n{'#' * 64}")
    print(f"  WALK-FORWARD — {args.model} | {len(tickers)} ticker(s) | "
          f"{args.folds} folds x {args.test_size} days")
    print(f"{'#' * 64}")

    per_fold = []
    for f in folds:
        print(f"\n{'=' * 64}")
        print(f"  Fold {f['fold']}/{args.folds}  "
              f"train<={f['train_end_date']}  val<={f['val_end_date']}  "
              f"test<={f['end_date']}")
        print(f"{'=' * 64}")

        a = copy.copy(args)
        a.train_end_date = f["train_end_date"]
        a.val_end_date = f["val_end_date"]
        a.end_date = f["end_date"]

        cmd_train(a)
        per_fold.append(cmd_test(a))

    _print_walkforward_summary(args, tickers, folds, per_fold)
    return per_fold


def _print_walkforward_summary(args, tickers, folds, per_fold):
    """Aggregate per-fold metrics into mean ± std, per ticker and overall."""
    def collect(ticker, section, key):
        vals = []
        for res in per_fold:
            block = res.get(ticker, {}).get(section, {})
            if key in block and np.isfinite(block[key]):
                vals.append(block[key])
        return np.array(vals, dtype=float)

    print(f"\n{'#' * 64}")
    print(f"  WALK-FORWARD SUMMARY — {args.model}  "
          f"({len(folds)} folds, mean ± std)")
    print(f"{'#' * 64}")
    print(f"{'Ticker':8} {'MAE $':>16} {'IC':>16} {'DA %':>16}")
    print("-" * 64)

    macro = {"MAE": [], "IC": [], "DA": []}
    for t in tickers:
        mae = collect(t, "overall", "MAE")
        ic = collect(t, "overall_returns", "IC")
        da = collect(t, "overall_returns", "DA")
        for name, arr in (("MAE", mae), ("IC", ic), ("DA", da)):
            if arr.size:
                macro[name].append(arr.mean())

        def fmt(arr, scale=1.0, prec=2):
            if not arr.size:
                return f"{'—':>16}"
            return f"{arr.mean() * scale:>9.{prec}f} ±{arr.std() * scale:<5.{prec}f}"

        print(f"{t:8} {fmt(mae)} {fmt(ic, prec=3)} {fmt(da, 100.0, prec=1)}")

    print("-" * 64)
    parts = []
    for name, scale, prec in (("MAE", 1.0, 2), ("IC", 1.0, 3), ("DA", 100.0, 1)):
        vals = np.array(macro[name], dtype=float)
        parts.append(f"{vals.mean() * scale:>9.{prec}f} {'':6}" if vals.size
                     else f"{'—':>16}")
    print(f"{'MACRO':8} " + " ".join(parts))
    print(f"{'#' * 64}\n")

    print("Note: the ± values are the spread across folds. Treat a difference "
          "between two models as real only if it is clearly larger than this.")


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
        p.add_argument("--seq_len", type=int, default=30)
        p.add_argument("--pred_len", type=int, default=5)
        p.add_argument("--data_root", type=str, default=DATA_ROOT)
        p.add_argument("--start_date", type=str, default="2022-01-01",
                       help="Train/test on rows from this ISO date onward "
                            "(default 2022-01-01). Set earlier to use more of "
                            "the fetched history.")
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
    p_test = subparsers.add_parser("test", help="Evaluate a trained model")
    add_common_args(p_test)
    p_test.add_argument("--batch_size", type=int, default=32)
    p_test.add_argument("--market_dim", type=int, default=2,
                        help="Must match the value used at train time (StockMixer only).")

    # --- walkforward ---
    p_wf = subparsers.add_parser(
        "walkforward",
        help="Train+test over several sequential folds and report mean ± std",
    )
    add_common_args(p_wf)
    p_wf.add_argument("--folds", type=int, default=4,
                      help="Number of consecutive test periods (default 4).")
    p_wf.add_argument("--test_size", type=int, default=126,
                      help="Trading days per test (and val) block; 126 ~ 6 months.")
    p_wf.add_argument("--epochs", type=int, default=200)
    p_wf.add_argument("--batch_size", type=int, default=32)
    p_wf.add_argument("--lr", type=float, default=2e-4)
    p_wf.add_argument("--patience", type=int, default=30)
    p_wf.add_argument("--alpha", type=float, default=0.1)
    p_wf.add_argument("--market_dim", type=int, default=2)
    p_wf.add_argument("--seed", type=int, default=42)

    # --- range (Stage-1 redesign: vol-normalised upside/downside band) ---
    p_range = subparsers.add_parser(
        "range",
        help="LightGBM upside/downside band forecast (forward-avg High/Low, "
             "Close anchor, vol-normalised) — single train/val/test split",
    )
    add_common_args(p_range)
    p_range.add_argument("--train_end_date", type=str, default="2023-11-24")
    p_range.add_argument("--val_end_date", type=str, default="2024-05-28")
    # Neural-only (ignored by LightGBM): TimeMixer / TimesNet training.
    p_range.add_argument("--epochs", type=int, default=60)
    p_range.add_argument("--batch_size", type=int, default=256)
    p_range.add_argument("--lr", type=float, default=2e-4)
    p_range.add_argument("--patience", type=int, default=15)
    # Walk-forward: >1 fold repeats train+eval over consecutive test blocks
    # (expanding train window) and reports mean ± std. LightGBM only for now.
    p_range.add_argument("--folds", type=int, default=1,
                         help="Walk-forward folds (1 = single split, the default).")
    p_range.add_argument("--test_size", type=int, default=126,
                         help="Trading days per test (and val) block; 126 ~ 6 months.")

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
        "range": cmd_range,
        "walkforward": cmd_walkforward,
        "test": cmd_test,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
