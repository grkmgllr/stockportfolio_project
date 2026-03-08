# Stock Price Forecasting & Meta-Labeling Trading Pipeline

![Python](https://img.shields.io/badge/Python-3.13-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-cu128-orange)
![LightGBM](https://img.shields.io/badge/LightGBM-4.x-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

**ENS 491/492 - Graduation Project** | **Sabanci University**

A two-stage algorithmic trading pipeline that combines **neural time-series forecasting** with **meta-labeling** (Lopez de Prado) to filter trade signals and improve precision.

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
  AAPL.csv (939 daily bars)
        |
   +---------+---------+
   |         |         |
TimesNet  TimeMixer  LightGBM   Stage 1: Price Forecasting
   |         |         |
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

### Stage 1 — Price Forecasting (AAPL, 5-day horizon)

| Model | MAE ($) | RMSE ($) | Parameters |
|-------|---------|----------|------------|
| **LightGBM** | **5.62** | **8.19** | 10 sub-models |
| TimesNet | 6.94 | 9.56 | 2.3M |
| TimeMixer | 7.96 | 11.00 | 69K |

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
│   ├── raw/                      # Raw parquet + resampled daily CSV
│   └── meta/                     # Meta-labels and predictions
├── models/
│   ├── TimesNet/                 # CNN-based temporal 2D variation
│   ├── TimeMixer/                # MLP-based multi-scale mixing
│   ├── LightGBMForecaster/       # GBDT with delta-based prediction
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
├── dataset.py                    # ParquetDataset (PyTorch Dataset)
├── train.py                      # Training script (all 3 models)
├── test.py                       # Evaluation script (all 3 models)
├── train_meta.py                 # Meta-classifier training
└── utils.py                      # Metrics, early stopping, schedulers
```

## Quick Start

### 1. Setup

```bash
git clone https://github.com/grkmgllr/stockportfolio_project.git
cd stockportfolio_project
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 2. Data Preparation

Place your minute-bar parquet file in `data/raw/`, then resample:

```bash
python scripts/resample_parquet.py --ticker AAPL --start_date 2022-01-01
```

### 3. Train & Test Models

```bash
# TimesNet (GPU)
python train.py --model TimesNet --ticker AAPL --seq_len 14 --pred_len 5 --epochs 100
python test.py  --model TimesNet --ticker AAPL --seq_len 14 --pred_len 5 --save_predictions

# TimeMixer (GPU)
python train.py --model TimeMixer --ticker AAPL --seq_len 14 --pred_len 5 --epochs 100
python test.py  --model TimeMixer --ticker AAPL --seq_len 14 --pred_len 5 --save_predictions

# LightGBM (CPU)
python train.py --model LightGBM --ticker AAPL --seq_len 14 --pred_len 5
python test.py  --model LightGBM --ticker AAPL --seq_len 14 --pred_len 5 --save_predictions
```

### 4. Meta-Labeling Pipeline

```bash
# Generate meta-labels from primary model predictions
python scripts/generate_meta_labels.py --ticker AAPL --seq_len 14 --pred_len 5

# Train meta-classifier
python train_meta.py --ticker AAPL
```

## Models

### TimesNet
Transforms 1D time series into 2D tensors to capture intra-period and inter-period variations using CNNs. Based on [Wu et al., 2023](https://arxiv.org/abs/2210.02186).

### TimeMixer
Uses MLP-based multi-scale mixing with Past-Decomposable-Mixing blocks for efficient time series forecasting.

### LightGBM Forecaster
Gradient boosted decision trees with hand-crafted features (returns, RSI, MACD, ATR, Bollinger width). Uses **delta-based prediction** — predicts price changes from the last Close, then converts back to absolute prices.

### Meta-Classifier
A secondary LightGBM classifier trained on market-context features to filter the primary model's trade signals. Uses **Purged K-Fold** cross-validation to prevent data leakage.

## References

- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Wu, H., et al. (2023). "TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis." ICLR 2023.
- Ke, G., et al. (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." NeurIPS 2017.

## License

This project is open-source and available under the **MIT License**.
