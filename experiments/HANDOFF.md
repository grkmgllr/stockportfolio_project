# Project Handoff — Stock Forecasting + Meta-Labeling Pipeline

*Context document so a fresh chat can continue. Written 2026-07-28.*

---

## 0. What this project is

A two-stage algorithmic-trading research pipeline (ENS 491/492 grad project,
Sabancı University), **writing a paper**:

- **Stage-1 (forecaster):** predict a per-stock forward signal.
  Models: **LightGBM**, **TimeMixer**, **TimesNet** (pooled), **StockMixer**
  (cross-stock).
- **Stage-2 (meta-labeling, López de Prado):** Triple Barrier Method turns the
  Stage-1 signal into labeled trades; a LightGBM meta-classifier filters which
  trades to take (Purged K-Fold CV).

**Universe:** 78 NASDAQ-100 stocks with full daily history from 2015
(`experiments/tickers.txt`; filtered from 94 so the panel is aligned). Data:
Yahoo daily OHLCV 2015-02 → 2025-11, one CSV per ticker in `data/raw/`, each with
OHLCV + 8 causal engineered features (`src/features.py`).

**Compute:** HPC via Slurm ONLY (never login node). partition/qos/account all
`cuda`. LightGBM run ≈1 min; neural ≈10-12 min (batch 256, 40 epochs, untuned).
GPUs are usually full — check with `sinfo`/`squeue`; A4000/A30 on cn14 often free.

---

## 1. The core story (what we found)

Starting point was a walk-forward run of 4 models on the price target. A critical
review uncovered that the **headline metrics were misleading**, and a redesign
followed. The honest conclusions:

1. **All architectures tie → the ceiling is the FEATURES, not the model.**
   LightGBM ≈ TimeMixer ≈ TimesNet (IC ≈ 0.22 on price target); LightGBM matches
   the neural nets at ~100× lower cost. Ordering LightGBM > TimeMixer > TimesNet
   holds on every target.

2. **The old "High/Close price" target inflated metrics.** Reported DA≈62% was
   the average of a structurally-biased **High** target (DA 72% = base rate = zero
   skill; the 5-day-ahead High is almost always above today's Close) and the
   tradeable **Close** direction (DA≈52% = coin flip). Close-MAE ≈ random-walk MAE.

3. **The "SMA-5 anchor" idea is a proven trap.** Anchoring the return to a
   trailing average instead of the entry price injects a quantity KNOWN at time t
   into the target → fake IC 0.57 while the true forecasting IC is ≈0. **Rule: the
   anchor must be the real entry price (Close[t]); any backward-looking average
   fakes the metric.** (Diagnostic in `experiments/diag.py`.)

4. **The old "66% → 90%" meta-labeling precision was in-sample overfitting.**
   Honest temporal holdout: the meta-classifier has **no out-of-sample lift**
   (5-stock −0.07, 78-stock +0.01). It memorises profitable trades in-sample.
   **Meta-labeling cannot create signal absent from the weak primary model.**

**Bottom line for the paper:** honest, defensible negative/positive results.
Both stages share one ceiling — **feature signal** — so feature enrichment is the
real lever.

---

## 2. The redesigned Stage-1 target (LOCKED, and in the code)

```
INPUT   : current 28 scale-invariant LightGBM features (same set for all models)
OUTPUT  : 2 channels, forward-averaged, Close[t] anchor, volatility-normalised
            upside   = (mean(High[t+1..t+5]) - Close[t]) / Close[t] / sigma[t]
            downside = (mean(Low [t+1..t+5]) - Close[t]) / Close[t] / sigma[t]
          sigma[t] = volatility_20 (20-day log-ret std, known at t)
          -> feeds Stage-2 triple barrier (upper=upside, lower=downside)
POOLING : build (X,y) per-ticker then concatenate (no cross-ticker leak)
SPLIT   : single train/val/test by date for iteration; walk-forward for FINAL only
          (train<=2023-11-24, val<=2024-05-28, test after)
MODEL   : LightGBM workhorse; neural for the fair comparison
DATA    : ffill only (bfill removed — it back-fills FUTURE values = leak)
```

Why vol-norm is safe (unlike the anchor): dividing by σ is a *scale*, not a
*shift*, so it injects no known additive term; it also removes the trivial
vol-driven magnitude, isolating real signal, and matches Stage-2 (barriers in σ).

### Verified results (single-split, 78 tickers, test n≈26,832, vol-norm units)

Range target — upside is the primary metric (ranking skill), IC_no = non-overlap:

| Model | upside IC | IC_no | downside IC | net IC | MAE | baseMAE |
|:---|---:|---:|---:|---:|---:|---:|
| LightGBM  | **0.124** | 0.102 | 0.107 | 0.039 | 1.151 | 1.163 |
| TimeMixer | 0.099 | 0.097 | 0.050 | 0.026 | 1.163 | 1.163 |
| TimesNet  | 0.083 | 0.075 | 0.045 | 0.022 | 1.162 | 1.163 |

- Upside IC 0.124 (LightGBM) beats a naive-climatology baseline (≈0.003) and holds
  on non-overlapping samples → **real but small** signal.
- **MAE is not discriminating** — every model sits at the predict-mean baseline
  (IC≈0.12 explains only ~1.5% of variance). Report IC, not MAE.
- **DA = up% base-rate** for up/dn individually → no directional skill; but Stage-2
  needs barrier *magnitudes*, not direction, so that's fine.

---

## 3. What is in the code now (branch `stage1-redesign`, on origin=grkmgllr)

Range mode is a **parallel, opt-in** capability; the legacy `price` mode is the
default and **untouched** (Stage-2 old path still works).

- `LightGBMForecaster(target_mode="range")` — `_build_range_arrays`,
  `fit_pooled_range`, `predict_range`, `range_ground_truth`.
- `StockDataset(target_mode="range", range_horizon=…)` — yields `[up, dn]`; RevIN
  denorm off (return basis). Model built with `pred_len=1, c_out=2`.
- `pytorch_runner.train(target_mode="range")` + `evaluate_range` — TimeMixer /
  TimesNet share this path (model-agnostic).
- `main.py range --model {LightGBM|TimeMixer|TimesNet} --tickers … --start_date 2015-01-01`
  — single subcommand, prints IC/IC_no/DA/up%.
- `triple_barrier.apply_triple_barrier(pred_low_col=…)` — both barriers from model.
- `src/meta/generate_range.py` — pooled range meta-dataset (ffill, date split,
  causal features + pred_upside/downside/band/skew).
- `bfill` leak fixed in `data_loading.py` and `dataset.py` (still present in the
  legacy `src/meta/generate.py`, unused by the range path).

Run commands (from repo root, via sbatch — never login node):
```
python main.py range --model LightGBM  --tickers $(cat logs/ndx100_full_history.txt) --start_date 2015-01-01
python main.py range --model TimeMixer --tickers $(cat logs/ndx100_full_history.txt) --start_date 2015-01-01 --epochs 40 --batch_size 256
```
Ad-hoc eval-only + meta scripts live in `logs/` (gitignored, shared-fs):
`run_range_meta.py`, `run_range_meta2.py` (in-sample vs OOS), `eval_range_full.py`.
The validation harness is `experiments/run_v3.py` (`on|off` = vol-norm).

---

## 4. Reproducibility / gotchas

- **Slurm only.** `#SBATCH -p cuda --qos=cuda --account=cuda [--gres=gpu:1]`.
  Use `sbatch` (survives session drops); `srun` inside a login shell dies with it.
- **`/tmp` scratchpad is login-local — compute nodes cannot see it.** Put scripts
  and Slurm `--output` on the shared filesystem (the repo, e.g. `logs/`).
- **Python stdout is buffered under Slurm** — set `export PYTHONUNBUFFERED=1`.
- Checkpoints: `checkpoints/pooled_LightGBM_range_best.joblib`,
  `pooled_range_{TimeMixer,TimesNet}_best.pt` (range);
  `pooled_{LightGBM,TimeMixer,TimesNet}_best.*` (legacy price).
- Isolated worktree `../stockportfolio_exp/` was used for iteration (symlinks
  venv+data); the design now lives in the main repo, so the worktree is optional.

---

## 5. Open items / next steps (priority order)

1. **Feature enrichment — the #1 lever** (raises BOTH stages, since the ceiling is
   features): market/sector return, beta, cross-sectional rank features (momentum
   rank, vol rank across the universe), and ideally fundamentals/earnings.
2. **Walk-forward for the final/paper numbers** on the range target, all models
   (range is currently single-split only; add `--folds` to the `range` command).
3. **Tighten the meta OOS eval**: add a formal embargo (pred_len gap) to the
   temporal holdout so the ~0-lift result is airtight; fix bfill in
   `meta/generate.py` if the legacy path is ever used.
4. **StockMixer range port** (cross-stock `dataset_crossstock` + `crossstock_runner`
   path) if a 4th model is wanted in the fair comparison. Note StockMixer was
   weakest on price and its cross-stock channel showed no benefit (market_dim
   ablation: smaller m was better).
5. Optional: cross-sectional IC (rank stocks per day) and trade-level Sharpe once a
   strategy is defined (Sharpe was explicitly deferred).

## 6. Decisions still open for the user

- Keep the legacy `price` mode or deprecate it once range is finalized (currently
  kept for the price-vs-range comparison and the old Stage-2).
- Vol-norm is ON in the locked range design (matches barrier use).

## 7. Git state

- Branch `stage1-redesign` on `origin` (grkmgllr) has everything through the range
  meta-labeling commit. Merge to `main` is done via GitHub PRs (#8, #9 already
  merged; the latest range-meta commit needs one more PR).
- Do NOT push to the `lab` remote.
- Full technical write-up: `experiments/experiment_report.md` (§1–10).
