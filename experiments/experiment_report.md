# Stage-1 Forecaster — Experiment Report

**Universe:** 78 NASDAQ-100 stocks with full history from 2015 (filtered from 94
NDX-100 tickers; those without ≥2015 data dropped so the cross-stock panel is
aligned). Daily OHLCV from Yahoo Finance, 2015-02 → 2025-11.

---

## TL;DR

- Three very different architectures (LightGBM, TimeMixer, TimesNet) converge on
  the **same** accuracy (IC ≈ 0.22). The bottleneck is the **feature signal**, not
  the model.
- The headline metrics were **misleading**: reported "DA ≈ 62%" is the average of a
  structurally-biased **High** target (72%, = base rate, zero skill) and the actual
  tradeable **Close** direction (≈52%, = base rate, zero skill).
- A proposed "stable reference" (SMA-5 anchor) was tested and **proven to be a
  fake-IC artifact** — it injects a quantity known at prediction time into the
  target. Real forecasting IC under that target is ≈ 0.
- After a clean redesign — **upside/downside (High/Low), forward-averaged,
  Close[t] anchor, volatility-normalized** — we get a **real, verified** signal:
  **upside IC ≈ 0.13** (vol-adjusted), which beats a naive climatology baseline
  (≈0.00) and holds on non-overlapping samples. This directly feeds the Stage-2
  triple-barrier method.

---

## 1. Initial walk-forward results (3 folds × 126 trading days, OOS)

Universe-level means (MACRO over 78 stocks):

| Model | MAE $ | IC | RIC | DA % | Train time |
|:---|---:|---:|---:|---:|---:|
| LightGBM | 6.10 | 0.218 | 0.235 | 62.5 | ~7 min |
| TimeMixer | 6.17 | 0.220 | 0.230 | 62.0 | ~12 h |
| TimesNet | 6.17 | 0.220 | 0.237 | 62.5 | ~17 h |
| StockMixer (md=4) | 6.31 | 0.192 | 0.226 | 60.8 | ~20 min |

**Finding:** differences are within the across-fold noise band. LightGBM matches
the neural models at ~100× lower cost. StockMixer's market-dimension ablation
showed *smaller m is better* (monotone), i.e. the cross-stock channel is not
adding value in this setup.

## 2. The headline metrics are inflated — per-channel breakdown

The model predicts two targets (next-5-day **High** and **Close**). Split apart:

| Channel | IC | DA % | "always-up" base rate |
|:---|---:|---:|---:|
| High | 0.155 | 72.1 | **72.4%** |
| Close | 0.097 | 52.8 | **52.9%** |

- **DA == base rate on both channels** → the model adds **zero directional skill**;
  it just predicts the majority class ("up").
- High looks good only because the 5-day-ahead High almost always exceeds today's
  Close (structural upward bias) — not skill.
- The genuinely tradeable target (Close direction) is a coin flip (52.5%), and
  Close-MAE ≈ random-walk MAE (6.21 vs 6.20) — **no better than "tomorrow = today".**

## 3. Why the "SMA-5 anchor" idea fails (a proven trap)

Supervisor's intuition: `Close[t]` is a noisy single-day reference; use something
more stable. Naively implementing this as `anchor = SMA_5[t]` gives:

```
target = (Close[t+k] − SMA_5[t]) / SMA_5[t]
       = [Close[t+k] − Close[t]]/…   +   [Close[t] − SMA_5[t]]/…
          └ future (unknown)              └ KNOWN at t (it is a feature)
```

IC decomposition (test n = 29,094):

| Quantity | IC |
|:---|---:|
| IC(pred, target) — headline | **0.567** |
| IC(pred, pure future return) — only tradeable part | **−0.025** |
| IC(known term, target) — trivial "predict the known part" | 0.573 |
| IC(pred, known term) — is the model just echoing it? | **0.963** |

**Conclusion:** the impressive 0.567 is entirely the known-at-t component; real
forecasting skill is ≈ 0. **Rule:** the anchor must be the price you actually enter
at (`Close[t]` or `Open[t+1]`); any backward-looking average injects a known term
and fakes the metric.

## 4. Final locked design

```
INPUT   : current 28 scale-invariant LightGBM features (same set for all models)
OUTPUT  : 2 channels, forward-averaged, Close[t] anchor, volatility-normalized
            up = (mean(High[t+1..t+5]) − Close[t]) / Close[t] / σ[t]
            dn = (mean(Low [t+1..t+5]) − Close[t]) / Close[t] / σ[t]
          σ[t] = volatility_20 (20-day log-return std, known at t)
          → upside feeds the upper barrier, downside the lower barrier (Stage-2)
POOLING : build (X,y) inside each ticker, then concatenate (no cross-ticker leak)
SPLIT   : single train/val/test by date for iteration; walk-forward for final only
MODEL   : LightGBM (workhorse); neural models deferred (equal accuracy, 100× cost)
DATA    : ffill only (bfill removed — it back-fills future values = leakage)
EVAL    : IC + non-overlap IC + DA vs base-rate + naive-climatology baseline
```

Why vol-normalization is safe (unlike the anchor): dividing by σ[t] is a *scaling*,
not a *shift*, so it injects no additive known component. It also removes the
trivial volatility-driven magnitude, isolating real signal — and it matches the
Stage-2 usage where barriers are set in units of σ.

## 5. Results of the final design (single-split, OOS test n = 29,094)

| vol-norm | Channel | Model IC | non-overlap IC | naive-climatology IC |
|:---|:---|---:|---:|---:|
| OFF | upside | 0.218 | 0.183 | 0.045 |
| OFF | downside | 0.103 | 0.091 | 0.018 |
| **ON** | **upside** | **0.128** | **0.123** | **0.003** |
| ON | downside | 0.097 | 0.100 | −0.013 |

- **Upside is genuinely predictable.** Vol-adjusted IC 0.128 vs naive 0.003 → the
  model is not just repeating recent range; it ranks stocks/dates by
  *higher-than-expected* upside. Non-overlap IC (0.123) confirms no overlap
  inflation.
- **~40% of the raw upside IC (0.218 → 0.128) was just volatility**; 0.128 is the
  honest residual skill.
- **Downside** also carries real skill (≈0.10, robust to vol-norm).
- **Net direction (up+dn)** is still weak (DA ≈ 53% = base rate) — but the project's
  triple-barrier method needs *barrier magnitudes* (upside/downside range), not
  direction, so this is acceptable.

## 6. Methodology notes (for the report / advisor)

- Predictions are **out-of-sample**: each fold/test window is strictly after its
  train+val window (no shuffle, no look-ahead).
- Pooled training builds features & targets **inside each ticker** before
  concatenating, so rolling/lag/shift never cross a ticker boundary.
- Every reported gain is checked against (a) a **naive baseline**
  (random-walk / climatology / always-up base rate) and (b) a **non-overlapping**
  IC, to rule out base-rate and autocorrelation artifacts.

## 7. Open items / next steps

1. **Feature enrichment (highest lever):** market/sector return, beta,
   cross-sectional rank features — the IC ceiling is a feature-signal ceiling.
2. **Stage-2 triple-barrier:** wire upside/downside predictions into
   upper/lower barriers + meta-classifier.
3. **Port the locked target into the main pipeline** (`dataset.py`,
   `lightgbm_forecaster.py`): remove bfill, single forward-avg High/Low target,
   vol-normalization; then re-validate with **walk-forward**.
4. Report cross-sectional IC (rank across the universe per day) and, later,
   trade-level Sharpe once a strategy is defined.

## 8. Reproducibility

- Isolated worktree: `stockportfolio_exp/` (symlinks `venv`, `data`).
- Harness: `experiments/run_v3.py` (final design) — `python run_v3.py on|off`.
- Diagnostics: `experiments/diag.py` (SMA-anchor IC decomposition),
  `experiments/run_v2.py` (anchor/target variants).
- Slurm: `experiments/*.sbatch` (partition `cuda`, LightGBM ≈ 70 s per run).

---

## 9. Fair model comparison on the range target (all models, one pipeline)

The range target (§4) is now integrated into the actual pipeline, not just the
harness:
- `LightGBMForecaster(target_mode="range")` + `main.py range --model LightGBM`
- `StockDataset(target_mode="range")` + `pytorch_runner` range path +
  `main.py range --model TimeMixer|TimesNet` (model-agnostic; same code path)

All three pooled forecasters trained + evaluated on the **identical** single
split (train ≤ 2023-11-24, val ≤ 2024-05-28, test after), 78 tickers,
test n = 26,832. IC_no = non-overlap (stride = pred_len) sanity check.

| Model | upside IC | upside IC_no | downside IC | net IC | train time |
|:---|---:|---:|---:|---:|---:|
| LightGBM  | **0.124** | 0.102 | **0.107** | **0.039** | ~1 min |
| TimeMixer | 0.099 | 0.097 | 0.050 | 0.026 | ~10 min |
| TimesNet  | 0.083 | 0.075 | 0.045 | 0.022 | ~12 min |

- LightGBM leads on every channel and metric; ordering **LightGBM > TimeMixer >
  TimesNet** matches the price-target result — the neural architectures do not
  beat gradient boosting here, now measured fairly on the corrected target.
- All honest (IC_no ≈ IC). Neural runs used 40 epochs, untuned (TimesNet mildly
  overfit); a final walk-forward + light tuning is a pre-paper step, but the
  ordering is decisive.
- StockMixer (cross-stock) is not yet ported to range mode — separate data path.

### 9.1 Full metrics (all channels, from saved checkpoints)

Vol-normalised units. `baseMAE` = predict-the-mean baseline. test n = 26,832.

| Model | Channel | IC | IC_no | MAE | RMSE | baseMAE | DA% | up% |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| LightGBM | upside | **0.124** | 0.102 | 1.151 | 1.689 | 1.163 | 71.6 | 71.6 |
| | downside | **0.107** | 0.120 | 1.174 | 1.719 | 1.188 | 67.2 | 32.8 |
| | net | 0.039 | 0.024 | 2.297 | 3.373 | 2.298 | 52.5 | 52.6 |
| TimeMixer | upside | 0.099 | 0.097 | 1.163 | 1.696 | 1.163 | 71.6 | 71.6 |
| | downside | 0.050 | 0.060 | 1.180 | 1.726 | 1.188 | 67.2 | 32.8 |
| | net | 0.026 | 0.032 | 2.303 | 3.377 | 2.298 | 52.5 | 52.6 |
| TimesNet | upside | 0.083 | 0.075 | 1.162 | 1.697 | 1.163 | 71.6 | 71.6 |
| | downside | 0.045 | 0.052 | 1.183 | 1.727 | 1.188 | 67.2 | 32.8 |
| | net | 0.022 | 0.020 | 2.305 | 3.378 | 2.298 | 51.3 | 52.6 |

**MAE is not a discriminating metric here.** Every model's MAE/RMSE sits at the
predict-the-mean baseline (LightGBM upside 1.151 vs baseMAE 1.163 — a ~1%
reduction; TimeMixer/TimesNet ≈ 1.163 = baseline). With IC ≈ 0.12 only ~1.5% of
target variance is explained, too little to move MAE. The signal is therefore
**real but small, visible in IC (ranking) not in MAE** — which is exactly why IC
(and, for the trading use, cross-sectional IC / Sharpe) is the primary metric and
MAE alone would understate the differences. Model ordering LightGBM > TimeMixer >
TimesNet is consistent across IC and the (marginal) MAE.

---

## 10. Stage-2 meta-labeling on the range model (both barriers from the model)

The range predictions now drive the triple barrier directly:
`upper = Close[t]·(1+upside·σ)`, `lower = Close[t]·(1+downside·σ)`
(`triple_barrier.apply_triple_barrier` gained an optional `pred_low_col`;
`src/meta/generate_range.py` builds the pooled meta-dataset, ffill-only,
Stage-1 date split, causal features + pred_upside/downside/band/skew).

**Trade distribution (78 tickers, 25,272 trades):** 47.9% take-profit / 51.4%
stop-loss / 0.7% timeout — balanced, non-degenerate.

**Why the old "66% → 90%" precision was misleading — in-sample vs OOS:**

| Universe | eval | thr | precision | lift |
|:---|:---|:---|---:|---:|
| 5 mega-caps | in-sample | 0.6 | **0.907** | (= the old headline) |
| 5 mega-caps | **OOS holdout** | 0.6 | **0.438** | **−0.066** |
| 78 full | in-sample | 0.6 | 0.861 | — |
| 78 full | **OOS holdout** | 0.5 | 0.475 | **+0.011** |

**The "66→90" was pure in-sample overfitting.** Trained and evaluated on the
same rows, the meta-classifier memorises which trades were profitable (5 stocks
→ 90%). On genuinely unseen future data the filter has **no skill** — negative
lift on the 5-stock set, negligible (+0.01) on 78. This is consistent with the
weak Stage-1 signal (IC≈0.12): **meta-labeling cannot manufacture signal that is
not in the primary model.** Both stages share one ceiling — feature signal.

(Caveat: the OOS holdout has no formal embargo yet; the small boundary overlap
would only inflate OOS, so the ~0-lift conclusion is conservative. Old
`meta/generate.py` still has a bfill leak; the new range path does not.)
