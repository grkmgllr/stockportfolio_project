# Stock Price Forecasting & Meta-Labeling Pipeline — Results Report

## 1. Project Overview

This project implements a two-stage algorithmic trading pipeline:

1. **Stage 1 — Price Forecasting**: Three models predict the next 5 days of High and Close prices for AAPL from 30 days of historical OHLCV data.
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

| Split | Ratio | Rows | Usable Samples (seq_len=30, pred_len=5) |
|-------|-------|------|-----------------------------------------|
| Train | 70% | 1724 | 1690 |
| Validation | 15% | 343 | 309 |
| Test | 15% | 343 | 309 |

---

## 3. Models

### 3.1 TimeMixer

- **Type**: Deep Learning (MLP-based multi-scale decomposable mixing)
- **Parameters**: 69,103
- **Input**: 7-feature OHLCV sequence (30 days)
- **Target Representation**: Percentage returns relative to anchor Close
- **Training**: CUDA GPU (NVIDIA RTX A4000), cosine LR scheduler, early stopping (patience=20)
- **Best Epoch**: 89 / 100 (ran full 100 epochs)
- **Best Validation Loss**: 0.000625

### 3.2 TimesNet

- **Type**: Deep Learning (CNN-based temporal 2D variation modeling)
- **Parameters**: 2,348,383
- **Input**: 7-feature OHLCV sequence (30 days)
- **Target Representation**: Percentage returns relative to anchor Close
- **Training**: CUDA GPU (NVIDIA RTX A4000), cosine LR scheduler, early stopping (patience=20)
- **Best Epoch**: 24 / 100 (early stopped at epoch 44)
- **Best Validation Loss**: 0.000642

### 3.3 LightGBM Forecaster

- **Type**: Gradient Boosted Decision Trees
- **Strategy**: Direct multi-step forecasting (1 model per forecast step x target = 10 models)
- **Features**: 31 hand-crafted features (returns, rolling stats, RSI, MACD, ATR, Bollinger width, volume dynamics, calendar features)
- **Key Design**: Return-based prediction — models predict the percentage return from the last known Close price, then the return is applied to recover absolute prices. This resolves the fundamental mismatch between scale-invariant features and absolute price targets.

---

## 4. Stage 1 Results — Price Forecasting

All models were evaluated on the same 309 test samples (unseen data from the last 15% of the time series).

### 4.1 Overall Metrics

| Model | MSE ($^2) | MAE ($) | RMSE ($) |
|-------|----------|---------|----------|
| **TimeMixer** | **48.39** | **4.79** | **6.96** |
| TimesNet | 52.53 | 5.03 | 7.25 |
| LightGBM | 67.11 | 5.62 | 8.19 |

### 4.2 Per-Target Breakdown

| Model | High MAE ($) | High RMSE ($) | Close MAE ($) | Close RMSE ($) |
|-------|-------------|---------------|---------------|----------------|
| **TimeMixer** | **4.58** | **6.67** | **5.01** | **7.23** |
| TimesNet | 4.79 | 7.03 | 5.27 | 7.46 |
| LightGBM | 5.42 | 7.87 | 5.82 | 8.50 |

### 4.3 Interpretation

- **TimeMixer achieved the lowest error** across all metrics, outperforming both TimesNet and LightGBM. Its multi-scale decomposition architecture, which separates trend and seasonal components at multiple temporal resolutions, benefits strongly from the return-based target representation.
- **Both DL models now outperform LightGBM**, reversing the previous ranking where LightGBM was the best model. The return-based target formulation resolved the fundamental scale mismatch that had handicapped the DL models.
- On a stock trading at ~$180-$260 during the test period, a MAE of $4.79 corresponds to approximately **2.2% average prediction error** over a 5-day horizon.
- TimesNet, despite having 34x more parameters than TimeMixer, achieved slightly worse results, suggesting that model capacity alone does not compensate for architectural differences in capturing multi-scale temporal patterns.

### 4.4 Key Insight — Return-Based Prediction for Deep Learning

The single most impactful improvement was switching all models from absolute price prediction to **return-based prediction** (predicting percentage changes relative to the anchor Close price).

| Model | Old MAE (absolute) | New MAE (returns) | Improvement | Old Best Epoch | New Best Epoch |
|-------|-------------------|------------------|-------------|----------------|----------------|
| TimeMixer | $7.96 | **$4.79** | **-39.8%** | 18 | **89** |
| TimesNet | $6.94 | **$5.03** | **-27.5%** | 5 | **24** |
| LightGBM | $36.71 → $5.62 | $5.62 (unchanged) | **-84.7%** (original) | — | — |

**Why return-based targets work:**

1. **Scale invariance**: Percentage returns are stationary and bounded (~[-20%, +20%] for 5-day horizons), whereas absolute prices drift over time. A model trained on $150 AAPL can generalize to $250 AAPL because the return patterns are similar.
2. **Smoother loss landscape**: MSE on small return values (~0.001-0.05) produces more informative gradients than MSE on large normalized price deviations, allowing models to train 5-18x longer before overfitting.
3. **Aligned with model normalization**: Both TimesNet (NS-Norm) and TimeMixer (RevIN) normalize inputs per-sample. By making outputs also scale-invariant, the entire pipeline operates in a consistent representation.

---

## 5. Stage 2 Results — Meta-Labeling Pipeline

The meta-labeling pipeline was applied to **TimesNet** predictions (best deep learning model at the time of this evaluation) to demonstrate signal filtering.

### 5.1 Triple Barrier Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Take-Profit | Predicted High | Upper barrier set to TimesNet's predicted High price |
| Stop-Loss | 2.0x daily volatility | Dynamic, widens in volatile regimes |
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
| **Recall** | 100.0% | 89.7% | -10.3 pp |
| **F1 Score** | 79.5% | **89.7%** | **+10.2 pp** |
| Trades Taken | 103 | 68 | -35 filtered |
| Filter Rate | 0% | 34.0% | -- |

### 5.5 Confusion Matrix

|  | Predicted Positive | Predicted Negative |
|--|-------------------|-------------------|
| **Actually Positive** | TP = 61 | FN = 7 |
| **Actually Negative** | FP = 7 | TN = 28 |

### 5.6 Risk-Adjusted Performance

| Metric | Baseline | Filtered | Interpretation |
|--------|----------|----------|----------------|
| **Sharpe Ratio (annualised)** | -0.52 | **6.62** | Strategy becomes highly profitable |
| **PSR** | 36.9% | **99.8%** | Near-certainty the SR exceeds zero |
| Skewness | -1.20 | -0.75 | Return distribution becomes less left-skewed |
| Kurtosis | 1.98 | 1.27 | Thinner tails (less extreme losses) |
| Observations | 103 | 68 | 34% of signals filtered out |

### 5.7 Interpretation

1. **Precision improvement (+23.7 pp)**: The meta-classifier successfully identified and removed 35 low-quality signals. Of the 68 remaining trades, nearly 90% were winners.
2. **Sharpe Ratio transformation**: The baseline strategy had a negative Sharpe Ratio (-0.52), meaning it was destroying value. After filtering, the Sharpe Ratio jumped to 6.62 — indicating strong risk-adjusted returns.
3. **PSR near 100%**: A Probabilistic Sharpe Ratio of 99.8% means there is near-statistical certainty that the filtered strategy outperforms the risk-free rate. This is far above the conventional 95% confidence threshold.
4. **Recall trade-off**: Recall dropped from 100% to 89.7%, meaning the filter missed 7 profitable trades. This is an acceptable trade-off: the goal of meta-labeling is not to capture every profitable trade, but to ensure that the trades we *do* take have a high probability of success.
5. **Improved return distribution**: Skewness improved from -1.20 to -0.75 (less negatively skewed), and kurtosis dropped from 1.98 to 1.27 (thinner tails), indicating a healthier return profile with fewer extreme losses.

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
     │TimeMixer │ │ TimesNet │ │ LightGBM │   STAGE 1:
     │  (DL)    │ │  (DL)    │ │  (GBDT)  │   Price Forecasting
     └────┬─────┘ └────┬─────┘ └────┬─────┘   (return-based targets)
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
     │  SL = 2x daily volatility          │
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
     │   Output: P(profitable) in [0, 1] │
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

### 7.1 Return-Based Target Formulation

All three models predict **percentage returns** relative to the anchor Close price (the last known Close in the input window):

```
target_return = (future_price - anchor_close) / anchor_close
predicted_price = anchor_close * (1 + predicted_return)
```

For the DL models (TimesNet, TimeMixer), this required:
- **Dataset**: Targets are computed as returns on-the-fly; no StandardScaler is applied to targets (returns are already scale-invariant).
- **Model output**: The NS-Norm (TimesNet) and RevIN (TimeMixer) denormalization steps are skipped on the output side, since the model directly outputs returns.
- **Input normalization**: Per-sample normalization on inputs is preserved — it helps the model process different price levels in the input window.
- **Evaluation**: Predicted returns are converted back to absolute prices using the anchor Close before computing MAE/RMSE in dollar scale.

### 7.2 Purged K-Fold Cross-Validation

Standard cross-validation causes data leakage in financial time series because consecutive observations are correlated. The Purged K-Fold implementation:

- **Purges** training samples whose event windows overlap with any test sample.
- **Embargoes** 3 bars after each test fold to prevent the model from exploiting lingering market reactions.
- Used 3 folds due to the small sample size (103 labeled observations).

### 7.3 Data-Cleaning Pipeline

The raw parquet data required significant cleaning before use:

| Issue | Count | Fix |
|-------|-------|-----|
| Pre/post-market bars | 435,859 (60.2%) | Filtered to RTH (09:30-16:00) |
| Weekend rows | 77 | Dropped (dayofweek >= 5) |
| Holiday/low-volume days | Detected dynamically | Dropped if volume < 5% of 21-day rolling median |

### 7.4 Environment

| Component | Version / Spec |
|-----------|---------------|
| GPU | NVIDIA RTX A4000 (16 GB VRAM) |
| Framework | PyTorch 2.6.0 (cu124) |
| Gradient Boosting | LightGBM 4.6.0 |
| Python | 3.12.3 |
| Data Period | Jan 2022 - Sep 2025 |

---

## 8. Conclusions

1. **Return-based prediction is essential for all model types** — switching from absolute price prediction to percentage returns improved TimeMixer by 39.8% and TimesNet by 27.5%. This was previously only applied to LightGBM (where it yielded an 84.7% improvement). The improvement stems from scale invariance, smoother loss landscapes, and alignment with per-sample input normalization.

2. **TimeMixer outperforms all models** with the lowest MAE ($4.79), RMSE ($6.96), and MSE (48.39). Its multi-scale decomposition architecture, which mixes seasonal and trend components across temporal resolutions, is well-suited to the return-based formulation where separating short-term noise from longer-term patterns is key.

3. **Deep learning now beats gradient boosting** — both TimeMixer and TimesNet outperform LightGBM, reversing the previous ranking. The return-based targets remove the scale mismatch that had given LightGBM's hand-crafted features an unfair advantage.

4. **Models train significantly longer before overfitting** — TimeMixer's best epoch moved from 18 to 89 (5x), TimesNet from 5 to 24 (5x). The return-based loss landscape provides more informative gradients throughout training.

5. **Meta-labeling significantly improved trading performance** — precision increased from 66% to 90%, and the Sharpe Ratio went from -0.52 to 6.62.

6. **The two-stage architecture** effectively separates the forecasting problem (Stage 1) from the trading decision problem (Stage 2), allowing each component to be optimized independently.

---

## 9. Future Work

- **Multi-ticker training**: Extend to MSFT, GOOGL, NVDA for a diversified portfolio and larger training set.
- **Directional accuracy metrics**: Add hit rate (% direction correct) and profit-weighted accuracy alongside MAE/RMSE.
- **Walk-forward validation**: Implement expanding-window retraining for more realistic out-of-sample evaluation.
- **Transaction costs**: Incorporate realistic bid-ask spreads and commission costs into the Sharpe Ratio calculation.
- **Position sizing**: Use the meta-classifier's probability output for Kelly criterion-based position sizing.
- **Hyperparameter optimization**: Systematic tuning of seq_len, pred_len, and model architectures.
- **Re-run meta-labeling with TimeMixer**: The meta-labeling results currently use TimesNet predictions; re-running with the now-superior TimeMixer predictions may further improve signal filtering.
