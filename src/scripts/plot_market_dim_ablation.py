"""
Plot the StockMixer market-dimension (m) ablation with multi-seed variance.

Reads per-(m, seed) test metrics from `results/ablation_seed/md{M}_seed{S}/{TICKER}/`
and validation loss from `logs/stockmixer_md{M}_seed{S}.log`, then draws the
mean across seeds as a bold line with a shaded ±1 std band — the standard way
to display seed uncertainty in ablation figures.

If no seeded runs are found, falls back to the single-seed layout under
`results/ablation/md{M}/{TICKER}/`.
"""
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TICKERS = ["AAPL", "MSFT", "GOOGL", "NVDA", "META"]
M_VALUES = [2, 3, 4, 5]
SEEDS = [0, 1, 2]

ABL_SEED_ROOT = Path("results/ablation_seed")
ABL_SINGLE_ROOT = Path("results/ablation")
LOG_ROOT = Path("logs")
OUT_PATH = Path("results/market_dim_ablation.png")


def val_loss_from_log(m: int, seed: int | None) -> float:
    """Read 'Best validation loss:' line from a training log."""
    if seed is None:
        log = LOG_ROOT / f"stockmixer_md{m}.log"
    else:
        log = LOG_ROOT / f"stockmixer_md{m}_seed{seed}.log"
    if not log.exists():
        return float("nan")
    for line in log.read_text().splitlines():
        if "Best validation loss:" in line:
            return float(line.split(":")[-1].strip())
    return float("nan")


def metric(m: int, seed: int | None, ticker: str, key_path: list[str]) -> float:
    """Read a nested key from metrics.json for one (m, seed, ticker) run."""
    if seed is None:
        p = ABL_SINGLE_ROOT / f"md{m}" / ticker / "metrics.json"
    else:
        p = ABL_SEED_ROOT / f"md{m}_seed{seed}" / ticker / "metrics.json"
    if not p.exists():
        return float("nan")
    data = json.loads(p.read_text())["metrics"]
    for k in key_path:
        data = data[k]
    return float(data)


def collect_multiseed():
    """Returns arrays shape [n_seeds, n_m] for scalars, [n_seeds, n_m, n_tickers] for per-ticker."""
    val_loss = np.array([[val_loss_from_log(m, s) for m in M_VALUES] for s in SEEDS])
    ic = np.array([
        [[metric(m, s, t, ["overall_returns", "IC"]) for t in TICKERS] for m in M_VALUES]
        for s in SEEDS
    ])
    da = 100.0 * np.array([
        [[metric(m, s, t, ["overall_returns", "DA"]) for t in TICKERS] for m in M_VALUES]
        for s in SEEDS
    ])
    return val_loss, ic, da


def band(ax, x, ys, color, label):
    """Draw mean line + ±1 std shaded band over the seed axis (axis=0)."""
    mean = np.nanmean(ys, axis=0)
    std = np.nanstd(ys, axis=0)
    ax.plot(x, mean, marker="o", linewidth=2.2, color=color, label=label, zorder=3)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.20, zorder=2)


def main():
    val_loss, ic, da = collect_multiseed()

    # Mean IC / DA per ticker averaged over seeds (for per-ticker curves)
    ic_by_ticker = np.nanmean(ic, axis=0)     # [n_m, n_tickers]
    da_by_ticker = np.nanmean(da, axis=0)     # [n_m, n_tickers]
    # Cross-ticker means, one value per (seed, m) — for the seed-band overlays
    ic_cross_mean = np.nanmean(ic, axis=2)    # [n_seeds, n_m]
    da_cross_mean = np.nanmean(da, axis=2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    # (a) Validation loss — one curve, seed variance shown as band
    ax = axes[0]
    band(ax, M_VALUES, val_loss, color="#1f77b4", label="mean ± 1 std")
    best_m = M_VALUES[int(np.nanargmin(np.nanmean(val_loss, axis=0)))]
    ax.axvline(best_m, linestyle="--", color="grey", alpha=0.6,
               label=f"best m = {best_m}")
    ax.set_xlabel("market dimension m")
    ax.set_ylabel("validation loss (MSE + α·rank)")
    ax.set_title(f"(a) Validation Loss  ({len(SEEDS)} seeds)")
    ax.set_xticks(M_VALUES)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # (b) IC — per-ticker light lines (seed-averaged) + black mean-of-means with band
    ax = axes[1]
    for i, t in enumerate(TICKERS):
        ax.plot(M_VALUES, ic_by_ticker[:, i], marker="o",
                alpha=0.45, linewidth=1, label=t)
    band(ax, M_VALUES, ic_cross_mean, color="black", label="mean ± 1 std")
    ax.axhline(0, color="grey", linewidth=0.6)
    ax.set_xlabel("market dimension m")
    ax.set_ylabel("Overall IC")
    ax.set_title(f"(b) Information Coefficient  ({len(SEEDS)} seeds)")
    ax.set_xticks(M_VALUES)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

    # (c) DA — same structure as (b)
    ax = axes[2]
    for i, t in enumerate(TICKERS):
        ax.plot(M_VALUES, da_by_ticker[:, i], marker="o",
                alpha=0.45, linewidth=1, label=t)
    band(ax, M_VALUES, da_cross_mean, color="black", label="mean ± 1 std")
    ax.axhline(50, color="grey", linewidth=0.6, linestyle=":")
    ax.set_xlabel("market dimension m")
    ax.set_ylabel("Directional Accuracy (%)")
    ax.set_title(f"(c) Directional Accuracy  ({len(SEEDS)} seeds)")
    ax.set_xticks(M_VALUES)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

    fig.suptitle(
        f"StockMixer — Market Dimension Ablation (N=5 stocks, {len(SEEDS)} seeds)",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    print(f"Saved → {OUT_PATH}")

    # Console summary for LaTeX table
    print("\nUnderlying values (mean ± std over seeds):")
    print(f"{'m':>3}  {'val_loss':>18}  {'mean_IC':>15}  {'mean_DA':>15}")
    for i, m in enumerate(M_VALUES):
        vl = val_loss[:, i]
        ic_c = ic_cross_mean[:, i]
        da_c = da_cross_mean[:, i]
        print(f"{m:>3}  "
              f"{np.nanmean(vl):.4e} ± {np.nanstd(vl):.1e}  "
              f"{np.nanmean(ic_c):>7.4f} ± {np.nanstd(ic_c):.4f}   "
              f"{np.nanmean(da_c):>6.2f} ± {np.nanstd(da_c):.2f}")


if __name__ == "__main__":
    main()
