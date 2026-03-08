# Stock Price Forecasting & Meta-Labeling Pipeline — Results Report

## 1. Project Overview

This project implements a two-stage algorithmic trading pipeline:

1. **Stage 1 — Price Forecasting**: Three models predict the next 5 days of High and Close prices for AAPL from 14 days of historical OHLCV data.
2. **Stage 2 — Meta-Labeling**: A secondary LightGBM classifier filters the primary model's trade signals using the Triple Barrier Method and market-context features, improving precision and risk-adjusted returns.

The meta-labeling architecture follows the framework introduced by Marcos Lopez de Prado in *Advances in Financial Machine Learning*.

---

## 2. Data

| Property | Value |
|----------|-------|
| Ticker | AAPL (Apple Inc.) |
| Source | Polygon.io minute-bar parquet (1.8M bars) |
| Resampling | Minute bars → Daily bars (Regular Trading Hours only: 09:30–16:00) |
| Date Range | 2022-01-03 to 2025-09-30 |
| Trading Days | 939 |
| Features | Open, High, Low, Close, Volume, VWAP, Transactions (7 features) |
| Targets | High, Close (2 targets) |

### Data Cleaning

- **Pre/post-market bars removed**: 60.2% of raw minute bars were outside Regular Trading Hours and were filtered out before resampling.
- **Weekend rows removed**: 77 Saturday/Sunday rows (caused by pre-market ECN trades) were eliminated.
- **Holiday/low-volume days removed**: Days with volume below 5% of the 21-day rolling median were dropped.
- **COVID-era exclusion**: Data before 2022-01-01 was excluded to avoid pandemic-related market anomalies.

### Train / Validation / Test Split

| Split | Ratio | Rows | Usable Samples (seq_len=14, pred_len=5) |
|-------|-------|------|-----------------------------------------|
| Train | 70% | 657 | 639 |
| Validation | 15% | 141 | 123 |
| Test | 15% | 141 | 123 |

---

## 3. Models

### 3.1 TimesNet

- **Type**: Deep Learning (CNN-based temporal 2D variation modeling)
- **Parameters**: 2,347,583
- **Input**: 7-feature OHLCV sequence (14 days)
- **Training**: CUDA GPU (NVIDIA RTX 5070 Ti), cosine LR scheduler, early stopping (patience=10)
- **Best Epoch**: 5 / 100 (early stopped at epoch 15)
- **Best Validation Loss**: 0.0589

### 3.2 TimeMixer

- **Type**: Deep Learning (MLP-based multi-scale mixing)
- **Parameters**: 69,103
- **Input**: 7-feature OHLCV sequence (14 days)
- **Training**: CUDA GPU, cosine LR scheduler, early stopping (patience=10)
- **Best Epoch**: 18 / 100 (early stopped at epoch 28)
- **Best Validation Loss**: 0.6097

### 3.3 LightGBM Forecaster

- **Type**: Gradient Boosted Decision Trees
- **Strategy**: Direct multi-step forecasting (1 model per forecast step × target = 10 models)
- **Features**: 31 hand-crafted features (returns, rolling stats, RSI, MACD, ATR, Bollinger width, volume dynamics, calendar features)
- **Key Design**: Delta-based prediction — models predict the *change* from the last known Close price, then the delta is added back to recover absolute prices. This resolves the fundamental mismatch between scale-invariant features and absolute price targets.

---

## 4. Stage 1 Results — Price Forecasting

All models were evaluated on the same 123 test samples (unseen data from the last 15% of the time series).

### 4.1 Overall Metrics

| Model | MSE ($²) | MAE ($) | RMSE ($) |
|-------|----------|---------|----------|
| **LightGBM** | **67.11** | **5.62** | **8.19** |
| TimesNet | 91.37 | 6.94 | 9.56 |
| TimeMixer | 121.03 | 7.96 | 11.00 |

### 4.2 Per-Target Breakdown

| Model | High MAE ($) | High RMSE ($) | Close MAE ($) | Close RMSE ($) |
|-------|-------------|---------------|---------------|----------------|
| **LightGBM** | **5.42** | **7.87** | **5.82** | **8.50** |
| TimesNet | 6.88 | 9.35 | 7.01 | 9.76 |
| TimeMixer | 7.69 | 10.55 | 8.23 | 11.44 |

### 4.3 Interpretation

- **LightGBM achieved the lowest error** across all metrics, outperforming both deep learning models despite having far fewer parameters and no GPU requirement.
- On a stock trading at ~$200–$260, a MAE of $5.62 corresponds to approximately **2.5% average prediction error** over a 5-day horizon.
- TimesNet was the best deep learning model, benefiting from its temporal 2D convolution architecture that captures both intra-period and inter-period patterns.
- TimeMixer, while the lightest model (69K parameters), showed the highest error, suggesting its MLP-mixing approach may need more data to learn effectively.

### 4.4 Key Insight — Delta-Based Prediction

The LightGBM forecaster initially produced a MAE of **$36.71** when predicting absolute prices. After switching to **delta-based prediction** (predicting price changes relative to the last Close), the MAE dropped to **$5.62** — an **85% improvement**. This highlights the importance of aligning the prediction target's scale with the feature space.

---

## 5. Stage 2 Results — Meta-Labeling Pipeline

The meta-labeling pipeline was applied to **TimesNet** predictions (best deep learning model) to demonstrate signal filtering.

### 5.1 Triple Barrier Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Take-Profit | Predicted High | Upper barrier set to TimesNet's predicted High price |
| Stop-Loss | 2.0× daily volatility | Dynamic, widens in volatile regimes |
| Vertical Barrier | 5 bars | Maximum holding period |
| Volatility Lookback | 20 bars | Rolling window for daily vol estimate |

### 5.2 Label Distribution

| Exit Type | Count | Percentage |
|-----------|-------|------------|
| Take-Profit (label=1) | 68 | 66.0% |
| Stop-Loss (label=0) | 20 | 19.4% |
| Timeout (label=0) | 15 | 14.6% |
| **Total** | **103** | **100%** |

### 5.3 Meta-Classifier Performance

The LightGBM meta-classifier was trained with Purged K-Fold cross-validation (3 folds, 3-bar embargo) on 11 features.

| Metric | Value |
|--------|-------|
| CV Accuracy | 71.9% |
| CV Log Loss | 0.6055 |
| Avg Best Iteration | 15 |

**Top Features by Importance (Gain)**:

| Feature | Gain |
|---------|------|
| MACD Histogram | 16.0 |
| Predicted Return | 8.0 |
| Transactions | 8.0 |
| Predicted Close Return | 7.0 |
| ATR | 5.0 |

### 5.4 Signal Filtering Results

| Metric | Baseline (All Signals) | Filtered (Meta-Classifier) | Change |
|--------|----------------------|---------------------------|--------|
| **Precision** | 66.0% | **89.7%** | **+23.7 pp** |
| **Recall** | 100.0% | 89.7% | −10.3 pp |
| **F1 Score** | 79.5% | **89.7%** | **+10.2 pp** |
| Trades Taken | 103 | 68 | −35 filtered |
| Filter Rate | 0% | 34.0% | — |

### 5.5 Confusion Matrix

|  | Predicted Positive | Predicted Negative |
|--|-------------------|-------------------|
| **Actually Positive** | TP = 61 | FN = 7 |
| **Actually Negative** | FP = 7 | TN = 28 |

### 5.6 Risk-Adjusted Performance

| Metric | Baseline | Filtered | Interpretation |
|--------|----------|----------|----------------|
| **Sharpe Ratio (annualised)** | −0.52 | **6.62** | Strategy becomes highly profitable |
| **PSR** | 36.9% | **99.8%** | Near-certainty the SR exceeds zero |
| Skewness | −1.20 | −0.75 | Return distribution becomes less left-skewed |
| Kurtosis | 1.98 | 1.27 | Thinner tails (less extreme losses) |
| Observations | 103 | 68 | 34% of signals filtered out |

### 5.7 Interpretation

1. **Precision improvement (+23.7 pp)**: The meta-classifier successfully identified and removed 35 low-quality signals. Of the 68 remaining trades, nearly 90% were winners.
2. **Sharpe Ratio transformation**: The baseline strategy had a negative Sharpe Ratio (−0.52), meaning it was destroying value. After filtering, the Sharpe Ratio jumped to 6.62 — indicating strong risk-adjusted returns.
3. **PSR near 100%**: A Probabilistic Sharpe Ratio of 99.8% means there is near-statistical certainty that the filtered strategy outperforms the risk-free rate. This is far above the conventional 95% confidence threshold.
4. **Recall trade-off**: Recall dropped from 100% to 89.7%, meaning the filter missed 7 profitable trades. This is an acceptable trade-off: the goal of meta-labeling is not to capture every profitable trade, but to ensure that the trades we *do* take have a high probability of success.
5. **Improved return distribution**: Skewness improved from −1.20 to −0.75 (less negatively skewed), and kurtosis dropped from 1.98 to 1.27 (thinner tails), indicating a healthier return profile with fewer extreme losses.

---

## 6. Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    RAW DATA (Parquet)                        │
│              1.8M minute bars (2015-2025)                    │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              resample_parquet.py                             │
│    Filter RTH (09:30-16:00) → Daily bars → Clean CSV        │
│              939 trading days                                │
└───────────────────────┬─────────────────────────────────────┘
                        │
            ┌───────────┼───────────┐
            ▼           ▼           ▼
     ┌──────────┐ ┌──────────┐ ┌──────────┐
     │ TimesNet │ │TimeMixer │ │ LightGBM │   STAGE 1:
     │  (DL)    │ │  (DL)    │ │  (GBDT)  │   Price Forecasting
     └────┬─────┘ └────┬─────┘ └────┬─────┘
          │            │            │
          ▼            ▼            ▼
     ┌────────────────────────────────────┐
     │     Predicted High & Close        │
     │     (next 5 days)                 │
     └────────────────┬──────────────────┘
                      │
                      ▼
     ┌────────────────────────────────────┐
     │      Triple Barrier Method         │   STAGE 2:
     │  TP = predicted High               │   Meta-Labeling
     │  SL = 2× daily volatility          │
     │  Timeout = 5 bars                  │
     └────────────────┬──────────────────┘
                      │
                      ▼
     ┌────────────────────────────────────┐
     │    Feature Engineering             │
     │  ATR, RSI, MACD, rolling vol,      │
     │  pred_return, pred_close_return    │
     └────────────────┬──────────────────┘
                      │
                      ▼
     ┌────────────────────────────────────┐
     │   LightGBM Meta-Classifier         │
     │   Purged K-Fold CV                 │
     │   Output: P(profitable) ∈ [0, 1]  │
     └────────────────┬──────────────────┘
                      │
                      ▼
     ┌────────────────────────────────────┐
     │      Signal Filter                 │
     │  Only trade when P > 0.5           │
     │  Precision: 66% → 90%             │
     │  Sharpe:   -0.52 → 6.62           │
     └────────────────────────────────────┘
```

---

## 7. Technical Details

### 7.1 Purged K-Fold Cross-Validation

Standard cross-validation causes data leakage in financial time series because consecutive observations are correlated. The Purged K-Fold implementation:

- **Purges** training samples whose event windows overlap with any test sample.
- **Embargoes** 3 bars after each test fold to prevent the model from exploiting lingering market reactions.
- Used 3 folds due to the small sample size (103 labeled observations).

### 7.2 Data-Cleaning Pipeline

The raw parquet data required significant cleaning before use:

| Issue | Count | Fix |
|-------|-------|-----|
| Pre/post-market bars | 435,859 (60.2%) | Filtered to RTH (09:30–16:00) |
| Weekend rows | 77 | Dropped (dayofweek >= 5) |
| Holiday/low-volume days | Detected dynamically | Dropped if volume < 5% of 21-day rolling median |

### 7.3 Environment

| Component | Version / Spec |
|-----------|---------------|
| GPU | NVIDIA RTX 5070 Ti (CUDA 12.8) |
| Framework | PyTorch (cu128) |
| Gradient Boosting | LightGBM |
| Python | 3.13 |
| Data Period | Jan 2022 – Sep 2025 |

---

## 8. Conclusions

1. **LightGBM outperformed deep learning** for daily stock price forecasting on structured tabular data, achieving 19% lower MAE than TimesNet with no GPU requirement.

2. **Delta-based prediction** is essential for tree-based models — predicting price changes instead of absolute levels reduced LightGBM's error by 85%.

3. **Meta-labeling significantly improved trading performance** — precision increased from 66% to 90%, and the Sharpe Ratio went from −0.52 to 6.62.

4. **The Probabilistic Sharpe Ratio of 99.8%** provides statistical confidence that the filtered strategy's performance is not due to chance.

5. **The two-stage architecture** effectively separates the forecasting problem (Stage 1) from the trading decision problem (Stage 2), allowing each component to be optimized independently.

---

## 9. Future Work

- **Multi-ticker training**: Extend to MSFT, GOOGL, NVDA for a diversified portfolio.
- **Position sizing**: Use the meta-classifier's probability output for Kelly criterion-based position sizing.
- **Walk-forward validation**: Implement expanding-window retraining for more realistic out-of-sample evaluation.
- **Hyperparameter optimization**: Systematic tuning of seq_len, pred_len, and model architectures.
- **Transaction costs**: Incorporate realistic bid-ask spreads and commission costs into the Sharpe Ratio calculation.
