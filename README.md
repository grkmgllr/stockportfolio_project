# Stock Price Forecasting & Meta-Labeling Trading Pipeline

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6-orange)
![LightGBM](https://img.shields.io/badge/LightGBM-4.6-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

**ENS 491/492 - Graduation Project** | **Sabanci University**

A two-stage algorithmic trading pipeline that combines **neural time-series forecasting** with **meta-labeling** (Lopez de Prado) to filter trade signals and improve precision. Trained on a **pooled dataset of 5 major tech stocks** (AAPL, MSFT, GOOGL, NVDA, META) using return-based prediction.

## Project Team

| Name | Role |
| :--- | :--- |
| **Alanur Ersoy** | Researcher / Developer |
| **Ege Serin** | Researcher / Developer |
| **Gorkem Guller** | Researcher / Developer |

**Supervisor:** Mehmet Emre Ozfatura

## Pipeline Overview

```
Raw Minute Bars (Parquet)
        |
        v
  resample_parquet.py          Data Cleaning (RTH filter, drop weekends/holidays)
        |
        v
  5 Ticker CSVs                AAPL, MSFT, GOOGL, NVDA, META (2022+)
  (849-941 daily bars each)
        |
   +---------+---------+
   |         |         |
TimesNet  TimeMixer  LightGBM   Stage 1: Price Forecasting
   |         |         |         (return-based targets, pooled training)
   +---------+---------+
        |
        v
  Triple Barrier Method        Stage 2: Meta-Labeling
        |
        v
  Feature Engineering           (ATR, RSI, MACD, volatility)
        |
        v
  LightGBM Meta-Classifier     Signal Filter (Purged K-Fold CV)
        |
        v
  Filtered Trade Signals        Precision: 66% -> 90%
```

## Key Results

### Stage 1 — Pooled TimeMixer (5-day horizon, return-based prediction)

| Ticker | MAE ($) | IC | RIC | DA |
|--------|---------|-------|-------|-------|
| AAPL | 4.73 | 0.163 | 0.173 | 64.2% |
| GOOGL | 4.50 | **0.189** | **0.205** | 70.9% |
| NVDA | **3.87** | 0.091 | 0.121 | 70.9% |
| MSFT | 6.85 | 0.106 | 0.181 | **72.7%** |
| META | 14.41 | 0.145 | 0.183 | 63.3% |

> IC = Information Coefficient (Pearson), RIC = Rank IC (Spearman), DA = Directional Accuracy

### Stage 2 — Meta-Labeling (Signal Filtering)

| Metric | Baseline | Filtered | Change |
|--------|----------|----------|--------|
| Precision | 66.0% | **89.7%** | **+23.7 pp** |
| F1 Score | 79.5% | **89.7%** | +10.2 pp |
| Sharpe Ratio | -0.52 | **6.62** | -- |
| PSR | 36.9% | **99.8%** | -- |

## Project Structure

```
stockportfolio_project/
├── data/
│   ├── raw/                      # Raw parquet + resampled daily CSVs (5 tickers)
│   └── meta/                     # Meta-labels and predictions
├── models/
│   ├── TimesNet/                 # CNN-based temporal 2D variation
│   ├── TimeMixer/                # MLP-based multi-scale mixing
│   ├── LightGBMForecaster/       # GBDT with return-based prediction
│   └── meta_classifier/          # LightGBM binary classifier
├── trading_logic/
│   ├── triple_barrier.py         # Triple Barrier Method (labeling)
│   ├── purged_cv.py              # Purged K-Fold cross-validation
│   └── evaluation.py             # Precision, F1, PSR metrics
├── scripts/
│   ├── resample_parquet.py       # Minute bars -> clean daily bars
│   └── generate_meta_labels.py   # Feature engineering bridge
├── docs/
│   └── RESULTS_REPORT.md         # Full results report
├── main.py                       # Unified entry point (train/test/meta/run-all)
├── dataset.py                    # ParquetDataset (PyTorch Dataset, multi-ticker)
├── train.py                      # Training script (single/pooled multi-ticker)
├── test.py                       # Evaluation script (per-ticker metrics)
├── train_meta.py                 # Meta-classifier training
└── utils.py                      # Metrics (MSE/MAE/IC/RIC/DA), early stopping
```

## Quick Start

### 1. Setup

```bash
git clone https://github.com/grkmgllr/stockportfolio_project.git
cd stockportfolio_project
python -m venv venv
source venv/bin/activate       # Linux/Mac
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 2. Data Preparation

Place your minute-bar parquet files in `data/raw/`, then resample:

```bash
python scripts/resample_parquet.py --ticker AAPL --start_date 2022-01-01
python scripts/resample_parquet.py --all   # All 5 tickers
```

### 3. Train & Test Models

**Pooled multi-ticker training (recommended):**

```bash
# Train on all 5 tickers simultaneously
python train.py --model TimeMixer --tickers AAPL MSFT GOOGL NVDA META \
                --seq_len 30 --pred_len 5 --epochs 200 --lr 0.0002

# Evaluate per-ticker
python test.py --model TimeMixer --tickers AAPL MSFT GOOGL NVDA META \
               --seq_len 30 --pred_len 5
```

**Single-ticker training:**

```bash
python train.py --model TimeMixer --ticker AAPL --seq_len 30 --pred_len 5 --epochs 100
python test.py  --model TimeMixer --ticker AAPL --seq_len 30 --pred_len 5
```

**Via unified entry point (main.py):**

```bash
python main.py train --model TimeMixer --tickers AAPL MSFT GOOGL NVDA META --epochs 200
python main.py test  --model TimeMixer --tickers AAPL MSFT GOOGL NVDA META
python main.py run-all --model TimeMixer --ticker AAPL  # Full pipeline (train+test+meta)
```

### 4. Meta-Labeling Pipeline

```bash
# Generate meta-labels from primary model predictions
python scripts/generate_meta_labels.py --ticker AAPL --seq_len 30 --pred_len 5

# Train meta-classifier
python train_meta.py --ticker AAPL
```

## Models

### TimeMixer
Uses MLP-based multi-scale mixing with Past-Decomposable-Mixing blocks. Best performing model with **69K parameters** — achieves IC up to 0.189 and DA up to 72.7% across 5 tickers.

### TimesNet
Transforms 1D time series into 2D tensors to capture intra-period and inter-period variations using CNNs. Based on [Wu et al., 2023](https://arxiv.org/abs/2210.02186). **2.3M parameters**.

### LightGBM Forecaster
Gradient boosted decision trees with 31 hand-crafted features (returns, RSI, MACD, ATR, Bollinger width). Uses **return-based prediction** — predicts percentage returns from the anchor Close price.

### Meta-Classifier
A secondary LightGBM classifier trained on market-context features to filter the primary model's trade signals. Uses **Purged K-Fold** cross-validation to prevent data leakage.

## Key Technical Decisions

- **Return-based targets**: All models predict percentage returns instead of absolute prices. This improved TimeMixer MAE by 39.8% and enables cross-ticker generalization.
- **Pooled multi-ticker training**: A single model is trained on all 5 tickers simultaneously via `ConcatDataset`. Return-based targets make this possible since all stocks are in the same scale.
- **Post-COVID data only**: All data is filtered to 2022-01-01+ to avoid pandemic-era anomalies.
- **Per-ticker evaluation**: Despite pooled training, test metrics are computed per-ticker to measure generalization.

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| MAE / RMSE | Price error in dollars (after converting returns back to prices) |
| IC | Information Coefficient — Pearson correlation between predicted and actual returns |
| RIC | Rank IC — Spearman rank correlation (ordinal agreement) |
| DA | Directional Accuracy — fraction of correct return sign predictions |

## References

- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Wu, H., et al. (2023). "TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis." ICLR 2023.
- Fan, J. & Shen, Y. (2024). Information Coefficient metrics for financial forecasting.
- Ke, G., et al. (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." NeurIPS 2017.

## License

This project is open-source and available under the **MIT License**.
