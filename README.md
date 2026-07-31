# Stock Forecasting Pipeline — Volatility-Normalised Range Prediction

![Python](https://img.shields.io/badge/Python-3.9-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.8-orange)
![LightGBM](https://img.shields.io/badge/LightGBM-4.6-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

**ENS 491/492 — Graduation Project** | **Sabanci University**

A research pipeline that forecasts a **volatility-normalised upside/downside
band** for each stock over a short horizon, and compares gradient boosting
against neural time-series models on **identical inputs**. Trained on a **pooled
panel of 78 NASDAQ-100 stocks** (Yahoo daily OHLCV, 2015–2025).

## Project Team

| Name | Role |
| :--- | :--- |
| **Alanur Ersoy** | Researcher / Developer |
| **Ege Serin** | Researcher / Developer |
| **Gorkem Guller** | Researcher / Developer |

**Supervisor:** Mehmet Emre Ozfatura

## What it predicts

Instead of predicting the (near-unpredictable) future price or its direction, the
model predicts **how far a stock swings up and down** over the next `H` days,
measured in units of its own recent volatility:

```
upside[t]   = ( mean(High[t+1..t+H]) - Close[t] ) / Close[t] / sigma[t]
downside[t] = ( mean(Low [t+1..t+H]) - Close[t] ) / Close[t] / sigma[t]
sigma[t]    = volatility_20  (std of the last 20 daily log-returns, known at t)
```

Two channels (`upside`, `downside`), forward-averaged over `H = 5` days, anchored
to the current Close, and divided by `sigma` — a *scaling*, not a *shift*, so it
injects no look-ahead. This band is the natural input to a triple-barrier trading
rule (upper = upside, lower = downside).

## Pipeline overview

```
Yahoo daily OHLCV (fetch)         78 NASDAQ-100 tickers, 2015-2025
        |
        v
  features.py                     Canonical causal feature set (12) + raw OHLCV
        |                         -> the SAME 17 columns for every model
        v
   +----------+-----------+----------+
   |          |           |          |
 LightGBM  TimeMixer   TimesNet   StockMixer      Stage-1: band forecast
   |          |           |          |            (StockMixer = cross-stock,
   +----------+-----------+----------+             return-ranking variant)
        |
        v
  IC / cross-sectional IC / walk-forward          Honest evaluation
```

## Latest results (reference baseline)

**LightGBM**, range target, 78 tickers, `start_date=2015-01-01`. These are the
current numbers — kept here as a baseline to compare future changes against.

**Single split** (train ≤ 2023-11-24, val ≤ 2024-05-28, test n = 26,832):

| Channel | IC | IC (non-overlap) | cross-sectional IC | ICIR |
|---|---|---|---|---|
| upside | **0.127** | 0.105 | **0.120** | 0.59 |
| downside | 0.110 | 0.122 | 0.097 | 0.44 |
| net (up+dn) | 0.044 | 0.025 | — | — |

**Walk-forward** (4 folds × 126 days, expanding train window; mean ± std):

| Channel | IC | cross-sectional IC |
|---|---|---|
| upside | **0.128 ± 0.024** | 0.123 ± 0.028 |
| downside | 0.098 ± 0.036 | 0.076 ± 0.035 |
| net | 0.036 ± 0.027 | — |

Upside IC is positive in all four folds — the signal is real and regime-robust,
not a single-split artifact. The band *magnitude* is what carries skill; net
up/down *direction* sits at the base rate (~53% DA).

> IC = Pearson corr(pred, true) pooled over stock-days. Cross-sectional IC = daily
> Spearman rank corr across stocks, averaged over days. ICIR = mean/std over days.
> Feature set: unified 17 columns (see below). Neural models (TimeMixer / TimesNet)
> on the unified features: comparison run pending.

## Project structure

```
stockportfolio_project/
├── main.py                          # Unified CLI: fetch / train / test / walkforward / range
├── src/
│   ├── features.py                  # Canonical causal feature set (single source of truth)
│   ├── features_cross.py            # Optional cross-sectional/market features (opt-in)
│   ├── dataset.py                   # StockDataset (PyTorch, per-ticker sequences)
│   ├── dataset_crossstock.py        # Cross-stock dataset (StockMixer)
│   ├── paths.py                     # Filesystem locations
│   ├── utils.py / reporting.py      # Metrics + reporting
│   ├── forecasting/
│   │   ├── lightgbm_runner.py       # LightGBM: price + range train/eval
│   │   ├── pytorch_runner.py        # TimeMixer / TimesNet train/eval
│   │   ├── crossstock_runner.py     # StockMixer train/eval
│   │   └── data_loading.py          # Shared CSV loading + splits
│   ├── models/
│   │   ├── LightGBMForecaster/      # GBDT forecaster
│   │   ├── TimeMixer/  TimesNet/    # Neural forecasters
│   │   └── StockMixer/              # Cross-stock model
│   └── scripts/
│       └── fetch_data.py            # Yahoo daily OHLCV downloader + feature engineering
└── data/raw/                        # One CSV per ticker (gitignored)
```

## Quick start

### 1. Setup

```bash
git clone https://github.com/grkmgllr/stockportfolio_project.git
cd stockportfolio_project
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu128   # or CPU build
```

### 2. Fetch data

```bash
# The 78-ticker universe:
python main.py fetch --tickers $(cat experiments/tickers.txt) --start 2015-01-01 --end 2025-11-30
# Or a few names:
python main.py fetch --tickers AAPL MSFT GOOGL NVDA META
```

### 3. Train & test (band forecast — the current target)

`train` fits a model and saves a checkpoint; `test` loads it and prints the full
metric table (IC, non-overlap IC, MAE, RMSE, baseMAE, DA%, up% per channel, plus
cross-sectional IC/ICIR). Same interface for every model.

```bash
TICKERS=$(cat experiments/tickers.txt)

# LightGBM (~1 min, no GPU):
python main.py train --model LightGBM  --tickers $TICKERS
python main.py test  --model LightGBM  --tickers $TICKERS

# Neural (GPU recommended):
python main.py train --model TimeMixer --tickers $TICKERS --epochs 40 --batch_size 256
python main.py test  --model TimeMixer --tickers $TICKERS

# Walk-forward (credible model comparison):
python main.py test  --model LightGBM  --tickers $TICKERS --folds 4 --test_size 126
```

`--target range` is the default; the split dates default to
`--train_end_date 2023-11-24 --val_end_date 2024-05-28`. Cross-sectional/market
features are opt-in: prefix a command with `RANGE_WITH_CROSS=1`.

### 4. Legacy price forecast (kept for comparison) — add `--target price`

```bash
python main.py train --model TimeMixer --tickers AAPL MSFT GOOGL --target price --epochs 200
python main.py test  --model TimeMixer --tickers AAPL MSFT GOOGL --target price
python main.py train --model StockMixer --tickers AAPL MSFT GOOGL  # cross-stock (own path)
```

## Models

- **LightGBM** — gradient-boosted trees on the flat feature row; the workhorse
  (matches the neural nets at ~100× lower cost).
- **TimeMixer** — MLP-based multi-scale mixing over the feature sequence.
- **TimesNet** — 2D temporal-variation CNN ([Wu et al., 2023](https://arxiv.org/abs/2210.02186)).
- **StockMixer** — cross-stock model that ranks stocks jointly (return-target variant).

## Feature set (the single source of truth)

All models consume the **same 17 columns**: raw OHLCV (5) plus the canonical 12
causal features in `features.py`:

| Group | Features |
|---|---|
| Momentum | `ret_1d`, `ret_5d`, `ret_20d` |
| Volatility | `volatility_20`, `atr`, `bb_width` |
| Range structure | `high_close_ratio`, `low_close_ratio`, `high_low_range`, `price_pos_20d` |
| Oscillator / volume | `rsi_14`, `vol_ma_ratio` |

## Evaluation metrics

| Metric | Description |
|---|---|
| IC | Pearson correlation between predicted and actual band, pooled over stock-days |
| IC (non-overlap) | Same, on a non-overlapping (stride = `pred_len`) subsample — leakage sanity check |
| Cross-sectional IC / ICIR | Daily Spearman rank IC across stocks, averaged; ICIR = mean/std over days |
| Walk-forward mean ± std | The above, repeated over expanding-window folds (credible comparison) |

## References

- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Wu, H., et al. (2023). "TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis." ICLR 2023.
- Ke, G., et al. (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." NeurIPS 2017.

## License

Open-source under the **MIT License**.
