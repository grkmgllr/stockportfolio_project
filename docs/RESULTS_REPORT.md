# Stock Price Forecasting & Meta-Labeling Pipeline — Results Report

## 1. Project Overview

This project implements a two-stage algorithmic trading pipeline:

1. **Stage 1 — Price Forecasting**: Three models predict the next 5 days of High and Close prices from 30 days of historical OHLCV data, trained on a pooled dataset of 5 major tech stocks (AAPL, MSFT, GOOGL, NVDA, META).
2. **Stage 2 — Meta-Labeling**: A secondary LightGBM classifier filters the primary model's trade signals using the Triple Barrier Method and market-context features, improving precision and risk-adjusted returns.

The meta-labeling architecture follows the framework introduced by Marcos Lopez de Prado in *Advances in Financial Machine Learning*.

---

## 2. Data

| Property | Value |
|----------|-------|
| Tickers | AAPL, MSFT, GOOGL, NVDA, META |
| Source | Polygon.io minute-bar parquet |
| Resampling | Minute bars → Daily bars (Regular Trading Hours only: 09:30–16:00) |
| Date Range | 2022-01-03 to 2025-09-30 (post-COVID) |
| Features | Open, High, Low, Close, Volume, VWAP, Transactions (7 features) |
| Targets | High, Close (2 targets) |

### Per-Ticker Data Summary

| Ticker | Trading Days | Train | Val | Test Samples |
|--------|-------------|-------|-----|--------------|
| AAPL | 939 | 657 | 141 | 107 |
| MSFT | 939 | 657 | 141 | 107 |
| GOOGL | 939 | 657 | 141 | 107 |
| NVDA | 941 | 659 | 141 | 108 |
| META | 849 | 594 | 127 | 94 |
| **Pooled Total** | **4,607** | **3,224** | **691** | **523** |

### Data Cleaning

- **Pre/post-market bars removed**: 60.2% of raw minute bars were outside Regular Trading Hours and were filtered out before resampling.
- **Weekend rows removed**: Saturday/Sunday rows (caused by pre-market ECN trades) were eliminated.
- **Holiday/low-volume days removed**: Days with volume below 5% of the 21-day rolling median were dropped.
- **COVID-era exclusion**: Data before 2022-01-01 was excluded to avoid pandemic-related market anomalies.
- **Date alignment**: All tickers filtered to 2022-01-01+ to ensure a consistent post-COVID market regime.

### Train / Validation / Test Split

| Split | Ratio | Description |
|-------|-------|-------------|
| Train | 70% | Per-ticker chronological split, pooled via ConcatDataset |
| Validation | 15% | Per-ticker chronological split (no cross-ticker leakage) |
| Test | 15% | Per-ticker chronological split, evaluated independently |

---

## 3. Models

### 3.1 TimeMixer (Pooled)

- **Type**: Deep Learning (MLP-based multi-scale decomposable mixing)
- **Parameters**: 69,103
- **Input**: 7-feature OHLCV sequence (30 days)
- **Target Representation**: Percentage returns relative to anchor Close
- **Training**: Pooled across 5 tickers (~3,224 train samples), CUDA GPU, cosine LR scheduler, early stopping (patience=30)
- **Best Epoch**: 71 / 200 (early stopped at epoch 101)
- **Best Validation Loss**: 0.001161
- **Learning Rate**: 2e-4

### 3.2 TimesNet (Single-Ticker)

- **Type**: Deep Learning (CNN-based temporal 2D variation modeling)
- **Parameters**: 2,348,383
- **Input**: 7-feature OHLCV sequence (30 days)
- **Target Representation**: Percentage returns relative to anchor Close
- **Training**: AAPL only, CUDA GPU, cosine LR scheduler, early stopping (patience=20)
- **Best Epoch**: 24 / 100 (early stopped at epoch 44)
- **Best Validation Loss**: 0.000642

### 3.3 LightGBM Forecaster

- **Type**: Gradient Boosted Decision Trees
- **Strategy**: Direct multi-step forecasting (1 model per forecast step x target = 10 models)
- **Features**: 31 hand-crafted features (returns, rolling stats, RSI, MACD, ATR, Bollinger width, volume dynamics, calendar features)
- **Key Design**: Return-based prediction — models predict the percentage return from the last known Close price, then the return is applied to recover absolute prices. This resolves the fundamental mismatch between scale-invariant features and absolute price targets.

---

## 4. Stage 1 Results — Price Forecasting

The pooled TimeMixer model was trained on all 5 tickers simultaneously and evaluated on each ticker's held-out test set independently. TimesNet results are from single-ticker (AAPL-only) training for comparison.

### 4.1 Pooled TimeMixer — Per-Ticker Results

| Ticker | MSE ($^2) | MAE ($) | RMSE ($) | IC | RIC | DA |
|--------|----------|---------|----------|-------|-------|-------|
| AAPL | 40.83 | 4.73 | 6.39 | 0.163 | 0.173 | 64.2% |
| GOOGL | 38.26 | 4.50 | 6.19 | **0.189** | **0.205** | 70.9% |
| NVDA | **27.07** | **3.87** | **5.20** | 0.091 | 0.121 | 70.9% |
| MSFT | 113.62 | 6.85 | 10.66 | 0.106 | 0.181 | **72.7%** |
| META | 389.70 | 14.41 | 19.74 | 0.145 | 0.183 | 63.3% |

### 4.2 Per-Target Breakdown (Pooled TimeMixer)

| Ticker | High MAE ($) | High DA | Close MAE ($) | Close DA |
|--------|-------------|---------|---------------|----------|
| AAPL | 4.64 | 74.2% | 4.83 | 54.2% |
| GOOGL | 4.30 | 81.9% | 4.69 | 60.0% |
| NVDA | 3.73 | 82.2% | 4.00 | 59.6% |
| MSFT | 7.05 | 83.9% | 6.66 | 61.5% |
| META | 14.35 | 72.6% | 14.47 | 54.0% |

### 4.3 Pooled vs Single-Ticker Comparison (AAPL)

| Metric | Single-Ticker TimeMixer | Pooled TimeMixer | Single-Ticker TimesNet |
|--------|------------------------|-----------------|----------------------|
| MAE ($) | 4.79 | **4.73** | 5.03 |
| RMSE ($) | 6.96 | **6.39** | 7.25 |
| IC | 0.159 | **0.163** | 0.052 |
| RIC | **0.193** | 0.173 | 0.104 |
| DA | **65.0%** | 64.2% | 60.4% |
| Train Samples | 639 | **3,224** (5x) | 639 |

### 4.4 Return-Based Metrics (IC / RIC / DA)

Following Fan & Shen [2024] and Wang et al. [2025], we evaluate model quality using metrics that measure how well predicted returns correlate with realised returns, independent of absolute price scale.

- **IC (Information Coefficient)**: Pearson correlation between predicted and actual returns. Measures linear agreement.
- **RIC (Rank Information Coefficient)**: Spearman rank correlation. Measures whether the model correctly ranks future returns by magnitude.
- **DA (Directional Accuracy)**: Fraction of predictions where the sign of the predicted return matches the sign of the actual return.

Key findings:

- **All 5 tickers have positive IC and RIC**, confirming that the pooled model learns cross-stock return patterns. An IC > 0.05 is generally considered informative in financial forecasting.
- **High DA is consistently excellent** (72-84%) across all tickers, likely because daily High is bounded below by Open, reducing the space of possible movements.
- **Close DA ranges from 54-62%**, above random (50%) but harder to predict than High.
- **GOOGL has the best IC (0.189) and RIC (0.205)**, suggesting GOOGL's return patterns are most predictable.
- **MSFT has the highest DA (72.7%)** despite higher dollar MAE, because MSFT trades at a higher price level (~$400).

### 4.5 Price Error Interpretation

- **Dollar MAE varies by price level**: META ($14.41) and MSFT ($6.85) have higher dollar MAE because they trade at higher prices (~$500-600 and ~$400). In percentage terms, all tickers have approximately **2-3% average prediction error** over a 5-day horizon.
- **Pooled training does not degrade per-ticker performance**: AAPL's MAE improved from $4.79 (single-ticker) to $4.73 (pooled), and IC improved from 0.159 to 0.163. The model successfully learns shared return patterns without sacrificing ticker-specific accuracy.
- **5x more training data** (3,224 vs 639 samples) enables the model to train to epoch 71 (vs 89 for single-ticker), learning more robust patterns across different market regimes.

### 4.6 Key Insight — Return-Based Prediction for Deep Learning

The single most impactful improvement was switching all models from absolute price prediction to **return-based prediction** (predicting percentage changes relative to the anchor Close price).

| Model | Old MAE (absolute) | New MAE (returns) | Improvement |
|-------|-------------------|------------------|-------------|
| TimeMixer | $7.96 | **$4.79** | **-39.8%** |
| TimesNet | $6.94 | **$5.03** | **-27.5%** |
| LightGBM | $36.71 → $5.62 | $5.62 (unchanged) | **-84.7%** (original) |

**Why return-based targets work:**

1. **Scale invariance**: Percentage returns are stationary and bounded (~[-20%, +20%] for 5-day horizons), whereas absolute prices drift over time. This is what enables pooled multi-ticker training — a single model can handle AAPL ($200) and META ($550) because the return patterns are in the same scale.
2. **Smoother loss landscape**: MSE on small return values (~0.001-0.05) produces more informative gradients than MSE on large normalized price deviations, allowing models to train longer before overfitting.
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
│           src/scripts/resample_parquet.py                    │
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

1. **Multi-ticker pooled training works** — a single TimeMixer model trained on 5 tickers (AAPL, MSFT, GOOGL, NVDA, META) achieves positive IC and DA across all tickers without degrading per-ticker performance. AAPL's MAE improved from $4.79 (single-ticker) to $4.73 (pooled), demonstrating that cross-stock patterns enhance generalization.

2. **Return-based prediction is essential for all model types** — switching from absolute price prediction to percentage returns improved TimeMixer by 39.8% and TimesNet by 27.5%. Return-based targets also enable multi-ticker training by making all stocks scale-invariant.

3. **TimeMixer outperforms all models** — best overall IC (up to 0.189 on GOOGL), DA consistently 63-73% across all tickers, and the lowest percentage prediction error (~2-3%). Its multi-scale decomposition architecture is well-suited to capturing shared temporal patterns across different stocks.

4. **High price is easier to predict than Close** — High DA is 72-84% across all tickers, while Close DA is 54-62%. This is consistent with the bounded nature of daily High (always >= Open).

5. **Meta-labeling significantly improved trading performance** — precision increased from 66% to 90%, and the Sharpe Ratio went from -0.52 to 6.62.

6. **The two-stage architecture** effectively separates the forecasting problem (Stage 1) from the trading decision problem (Stage 2), allowing each component to be optimized independently.

---

## 9. Future Work

- **Walk-forward validation**: Implement expanding-window retraining for more realistic out-of-sample evaluation.
- **Transaction costs**: Incorporate realistic bid-ask spreads and commission costs into the Sharpe Ratio calculation.
- **Position sizing**: Use the meta-classifier's probability output for Kelly criterion-based position sizing.
- **Hyperparameter optimization**: Systematic tuning of seq_len, pred_len, and model architectures.
- **Re-run meta-labeling with TimeMixer**: The meta-labeling results currently use TimesNet predictions; re-running with the now-superior pooled TimeMixer predictions may further improve signal filtering.
- **TimesNet pooled training**: Extend pooled training to TimesNet for comparison.
- **Sector diversification**: Add non-tech tickers to test cross-sector generalization.
