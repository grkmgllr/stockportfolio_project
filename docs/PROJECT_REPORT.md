# Stock Price Forecasting with Meta-Labeling: A Multi-Phase Deep Learning Framework

## Abstract

This project presents a multi-phase framework for stock price forecasting and automated trade filtering. The system combines deep learning-based time series forecasting (TimesNet, TimeMixer) with a gradient-boosted meta-classifier (LightGBM) to predict future stock prices and evaluate the profitability of trading signals. The framework implements the Triple Barrier Method and Purged K-Fold Cross-Validation from Lopez de Prado's *Advances in Financial Machine Learning* to generate economically meaningful labels and prevent data leakage. A recent extension adds moving average (EMA-20, SMA-50) prediction targets, providing the meta-classifier with trend-context features that improve trade filtering precision from a 48.9% baseline to 93.7%.

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [Data Pipeline](#2-data-pipeline)
3. [Primary Forecasting Models](#3-primary-forecasting-models)
   - 3.1 [TimesNet](#31-timesnet)
   - 3.2 [TimeMixer](#32-timemixer)
4. [Moving Average Prediction Targets](#4-moving-average-prediction-targets)
5. [Triple Barrier Method](#5-triple-barrier-method)
6. [Feature Engineering](#6-feature-engineering)
7. [Meta-Classifier (LightGBM)](#7-meta-classifier-lightgbm)
8. [Purged K-Fold Cross-Validation](#8-purged-k-fold-cross-validation)
9. [Evaluation Framework](#9-evaluation-framework)
10. [Experimental Results](#10-experimental-results)
11. [Project Structure](#11-project-structure)
12. [Usage Guide](#12-usage-guide)
13. [References](#13-references)

---

## 1. System Architecture

The framework is organized into six sequential phases, each building on the output of the previous one:

```
Phase 1: Data Acquisition & Preprocessing
    │  Raw OHLCV data → StandardScaler normalization → sliding window samples
    ▼
Phase 2: Primary Model Training
    │  TimesNet / TimeMixer learns to predict High, Close, EMA_20, SMA_50
    ▼
Phase 3: Prediction & Evaluation
    │  Model predicts on test set → saves .npy arrays
    ▼
Phase 4: Meta-Label Generation (Triple Barrier)
    │  Simulates trades using predicted High as take-profit
    │  Labels each signal as profitable (1) or unprofitable (0)
    │  Engineers technical indicator features
    ▼
Phase 5: Meta-Classifier Training (LightGBM + Purged K-Fold)
    │  Learns which signals are likely profitable
    ▼
Phase 6: Evaluation
    │  Compares baseline vs filtered strategies
    │  Computes Probabilistic Sharpe Ratio (PSR)
```

The key insight is the **separation of concerns**: the primary deep learning model handles *direction and price-level prediction*, while the secondary machine learning model handles *signal filtering and position sizing*. This follows the meta-labeling paradigm introduced by Lopez de Prado (2018).

---

## 2. Data Pipeline

### 2.1 Data Source

The system supports two data sources:

- **Yahoo Finance API** (`scripts/fetch_data.py`): Downloads daily OHLCV data using the `yfinance` library.
- **Minute-bar Parquet files** (`scripts/resample_parquet.py`): Resamples high-frequency data to daily bars, preserving volume-weighted average price (VWAP) and transaction count.

### 2.2 Input Features

| Feature | Description | Source |
|---------|-------------|--------|
| Open | Opening price | Raw data |
| High | Intraday high | Raw data |
| Low | Intraday low | Raw data |
| Close | Closing price | Raw data |
| Volume | Trading volume | Raw data |
| Vwap | Volume-weighted average price | Extended (optional) |
| Transactions | Number of transactions | Extended (optional) |

The dataset automatically detects whether extended columns (Vwap, Transactions) are available and adjusts `enc_in` accordingly (5 or 7 features).

### 2.3 Target Variables

| Target | Description | Type |
|--------|-------------|------|
| High | Future daily high price | Base target |
| Close | Future daily close price | Base target |
| EMA_20 | 20-period exponential moving average | Optional MA target |
| SMA_50 | 50-period simple moving average | Optional MA target |

### 2.4 Preprocessing

**Normalization:** StandardScaler is fit exclusively on the training split and applied to all splits (train, validation, test). Separate scalers are maintained for input features (`scaler_x`) and target features (`scaler_y`) to enable proper inverse transformation during evaluation.

**Train/Validation/Test Split:** Date-based sequential split with default ratios of 70% / 15% / 15%. No shuffling is applied, preserving temporal ordering.

**Sliding Window Construction:** Each sample consists of:
- Input: `seq_len` consecutive bars (default: 30 days)
- Target: The next `pred_len` bars (default: 5 days)

```
Sample i:
  Input:  data[i : i + seq_len]         → shape [seq_len, enc_in]
  Target: data[i + seq_len : i + seq_len + pred_len]  → shape [pred_len, c_out]
```

### 2.5 Moving Average Computation

When MA targets are enabled, the moving averages are computed on the full dataset *before* splitting to ensure rolling windows see complete history. Leading NaN rows from the warm-up period (max MA period - 1) are trimmed.

- **EMA-20:** Exponential weighted mean with span=20, using the `adjust=False` convention (recursive EMA formula).
- **SMA-50:** Simple rolling mean with window=50.

Both are causal (non-look-ahead): the value at time *t* depends only on data at time *t* and earlier, making pre-split computation safe from data leakage.

---

## 3. Primary Forecasting Models

### 3.1 TimesNet

**Reference:** Wu et al., "TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis," ICLR 2023.

TimesNet is the default primary model. Its central idea is that 1D time series can exhibit complex temporal patterns that become clearer when "unfolded" into 2D representations based on their dominant periodicities.

#### Architecture

**Step 1: Non-Stationary Normalization**

Each input sample is normalized per-channel across the time dimension:

```
means = x.mean(dim=time)              → [B, 1, C]
stdev = sqrt(var(x, dim=time) + ε)    → [B, 1, C]
x_norm = (x - means) / stdev
```

This stabilizes training by removing per-sample scale differences, following the Non-Stationary Transformer (Liu et al., 2022) approach. The mean is detached from the computation graph so gradients do not flow through the normalization statistics.

**Step 2: Data Embedding**

The normalized input passes through three additive embedding components:

- **Token Embedding:** 1D convolution (`Conv1d`, kernel_size=3, circular padding) that projects each time step from `enc_in` to `d_model` dimensions.
- **Positional Embedding:** Sinusoidal encoding (max_len=5000) providing absolute position information.
- **Temporal Embedding:** Discrete calendar features (month, day, weekday, hour, minute) embedded via learned lookup tables.

Output shape: `[B, seq_len, d_model]`

**Step 3: Time Alignment**

A linear layer expands the time dimension from `seq_len` to `seq_len + pred_len`:

```
align_time = nn.Linear(seq_len, seq_len + pred_len)
```

This operates on each feature channel independently (transposed application), effectively interpolating/extrapolating the embedded representation into the forecast horizon.

**Step 4: TimesBlock (Core Innovation)**

Each TimesBlock performs three sub-operations:

**(a) FFT-Based Period Discovery:**

The block computes the real FFT along the time axis and selects the top-k frequencies by amplitude (averaged across batch and channels). Each selected frequency index `f` corresponds to a period `p = T / f`. The DC component (f=0) is suppressed.

**(b) 1D → 2D Folding:**

For each discovered period `p`, the 1D sequence of length `T` is reshaped into a 2D tensor of shape `[B, C, ceil(T/p), p]`. Zero-padding is applied if `T` is not divisible by `p`. This transformation exposes both intra-period (columns) and inter-period (rows) patterns as spatial structure.

**(c) Inception-Style 2D Convolution:**

The 2D representation is processed by an Inception block with `num_kernels` parallel convolutional branches (kernel sizes 1, 3, 5, 7, 9, 11). The structure is:

```
InceptionBlock(d_model → d_ff) → GELU → InceptionBlock(d_ff → d_model)
```

Each branch captures patterns at different spatial scales. Outputs are averaged across branches.

**(d) Aggregation:**

After unfolding back to 1D, the outputs from all top-k periods are combined via a softmax-weighted sum using the FFT amplitudes as attention weights:

```
attention = softmax(amplitudes, dim=period)
output = Σ(attention_k × output_k)  +  residual
```

**Step 5: Output Projection**

A linear layer projects from `d_model` to `c_out` (number of target features). Only the last `pred_len` time steps are returned as the forecast.

**Step 6: Denormalization**

The output is rescaled to the original data scale using the saved means and standard deviations (sliced to `c_out` channels).

#### Configuration (as used in this project)

| Parameter | Value | Description |
|-----------|-------|-------------|
| seq_len | 30 | Lookback window (days) |
| pred_len | 5 | Forecast horizon (days) |
| enc_in | 5 or 7 | Input features (auto-detected) |
| c_out | 2 or 4 | Output targets (with/without MAs) |
| d_model | 32 | Embedding dimension |
| d_ff | 64 | FFN hidden dimension |
| e_layers | 2 | Number of TimesBlocks |
| top_k | 3 | Dominant periods per block |
| num_kernels | 6 | Inception branches |
| dropout | 0.1 | Dropout probability |

#### Data Flow Summary

```
Input [B, 30, 7]
  → NS-Normalize [B, 30, 7]
  → Embed [B, 30, 32]
  → Align Time [B, 35, 32]
  → TimesBlock ×2 [B, 35, 32]
  → Project [B, 35, 4]
  → Denormalize [B, 35, 4]
  → Slice last 5 → Output [B, 5, 4]
```

---

### 3.2 TimeMixer

**Reference:** Wang et al., "TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting," ICLR 2024.

TimeMixer is an alternative primary model that takes a fundamentally different approach: instead of converting time series to 2D images, it decomposes the signal into seasonal and trend components at multiple temporal scales and mixes them through learned interactions.

#### Architecture

**Step 1: Multi-Scale Input Construction**

The input sequence is progressively downsampled to create a pyramid of temporal resolutions:

```
Scale 0 (finest):   [B, L, C_in]         e.g., [B, 30, 7]
Scale 1 (coarser):  [B, L/2, C_in]       e.g., [B, 15, 7]
Scale 2 (coarsest): [B, L/4, C_in]       e.g., [B, 7, 7]  (if applicable)
```

Downsampling is performed via average pooling (`AvgPool1d`) along the time axis. The number of downsampling layers is automatically determined based on divisibility constraints.

**Step 2: Per-Scale Embedding**

Each scale has its own linear embedding layer that projects from `C_in` to `d_model`:

```
scale_embeddings[i] = nn.Linear(C_in, d_model)
```

Output: list of `[B, T_i, d_model]` tensors, one per scale.

**Step 3: Past Decomposable Mixing (PDM)**

The PDM block is the core of TimeMixer. It processes the multi-scale representations through three stages:

**(a) Decomposition:**

Each scale is decomposed into seasonal and trend components using a centered moving average:

```
trend = MovingAverage(x, kernel_size=25)
seasonal = x - trend
```

The moving average uses symmetric padding (replicate mode) to avoid boundary artifacts.

**(b) Multi-Scale Season Mixing (Bottom-Up):**

Seasonal components are mixed from fine to coarse scales. Each mixer is a `TemporalLinearMixer` — a two-layer MLP that maps between different time dimensions:

```
Linear(T_fine, hidden) → GELU → Dropout → Linear(hidden, T_coarse)
```

The mixing proceeds bottom-up: the finest scale's seasonal pattern is projected and added to the next coarser scale, cascading upward.

**(c) Multi-Scale Trend Mixing (Top-Down):**

Trend components are mixed in the opposite direction (coarse to fine). The coarsest trend is projected and added to finer scales, propagating long-term trend information downward.

**(d) Fusion and Residual:**

The mixed seasonal and trend components are summed, passed through a channel-wise FFN, and combined with the original input via a residual connection:

```
fused = season_mixed + trend_mixed
output = LayerNorm(original + FFN(fused))
```

When `use_channel_independence=False` (default), a pre-mix channel FFN allows cross-channel interaction before decomposition.

**Step 4: Future Multi-Predictor Mixing (FMM)**

Each scale independently generates a forecast by applying a temporal linear projection:

```
predictor[i] = nn.Linear(T_i, pred_len)    # maps T_i → H for each feature
```

The per-scale forecasts (all of shape `[B, pred_len, d_model]`) are summed to form the ensemble prediction:

```
fused = Σ predictor_i(encoded_i)
```

This allows different scales to contribute complementary forecast information: fine scales capture short-term dynamics while coarse scales capture longer-term patterns.

**Step 5: Output Projection**

```
output_projection = nn.Linear(d_model, c_out)
```

#### Configuration (as used in this project)

| Parameter | Value | Description |
|-----------|-------|-------------|
| historical_lookback_length | 30 | Lookback window |
| forecast_horizon_length | 5 | Forecast horizon |
| number_of_input_features | 5 or 7 | Auto-detected |
| number_of_output_features | 2 or 4 | With/without MAs |
| model_embedding_dimension | 64 | Embedding dimension |
| feedforward_hidden_dimension | 128 | FFN hidden dimension |
| number_of_pdm_blocks | 2 | PDM stack depth |
| downsampling_window_size | 2 | Pool kernel size |
| moving_average_kernel_size | 25 | Decomposition kernel |
| use_channel_independence | False | Cross-channel mixing enabled |

#### Data Flow Summary

```
Input [B, 30, 7]
  → Multi-scale: [B,30,7], [B,15,7]
  → Embed: [B,30,64], [B,15,64]
  → PDM Block ×2:
      Decompose → Season Mix (↑) → Trend Mix (↓) → Fuse
  → FMM: per-scale Linear → Sum → [B, 5, 64]
  → Project → Output [B, 5, 4]
```

---

### 3.3 Additional Models (Not Active in Pipeline)

The repository includes two additional model architectures that are implemented but not currently integrated into the training/testing pipeline:

**TimeMixer++:** An extension of TimeMixer that introduces Multi-Resolution Time Imaging (MRTI) — using FFT to discover dominant periods and fold the 1D series into 2D images — combined with Time Image Decomposition (TID), a 2D version of the season/trend decomposition using 2D average pooling.

**ModernTCN:** A pure convolutional architecture using patch embedding (Conv1d with stride) and large-kernel depthwise separable convolutions (kernel_size=51). Optionally includes multi-scale depthwise convolutions with branches at kernel sizes 7, 15, 31, and 51. Uses RevIN (Reversible Instance Normalization) for input/output normalization.

---

## 4. Moving Average Prediction Targets

### 4.1 Motivation

Raw stock prices (High, Close) are inherently noisy, making them difficult targets for neural networks. Moving averages smooth out daily fluctuations and expose the underlying trend, providing:

1. **Smoother learning targets** that reduce overfitting risk.
2. **Trend-direction signals** that are more actionable than exact price levels.
3. **Crossover information** (EMA-20 vs SMA-50) that is one of the most widely used trading signals.

### 4.2 Design Choice: Augmentation, Not Replacement

The MA targets are appended *alongside* the original High and Close targets, not as replacements. This preserves the triple barrier's take-profit mechanism (which requires predicted High) while providing the meta-classifier with additional trend-context features.

### 4.3 Implementation

The `ParquetDataset` class accepts an optional `ma_targets` parameter:

```python
dataset = YahooDataset(
    ticker="AAPL",
    ma_targets=["EMA_20", "SMA_50"],  # appends to ['High', 'Close']
)
# Result: target_features = ['High', 'Close', 'EMA_20', 'SMA_50'], c_out = 4
```

MA configurations are defined in a class-level registry:

| Name | Method | Period | Formula |
|------|--------|--------|---------|
| EMA_20 | Exponential | 20 | `close.ewm(span=20, adjust=False).mean()` |
| SMA_50 | Simple | 50 | `close.rolling(window=50).mean()` |

**Why EMA-20 and SMA-50?**

- EMA-20 reacts faster to recent price changes (appropriate for the 5-day forecast horizon) and is consistent with the MACD indicator already used in the meta-classifier.
- SMA-50 is the most widely watched institutional moving average, often acting as support/resistance. Its stability makes it a reliable prediction target.

---

## 5. Triple Barrier Method

**Reference:** Lopez de Prado, *Advances in Financial Machine Learning*, Chapter 3, 2018.

The Triple Barrier Method transforms raw price predictions into economically meaningful binary labels by simulating trades with three exit conditions.

### 5.1 Barrier Construction

For each signal bar (where the primary model generates a prediction), three barriers are set around the entry price:

**Upper Barrier (Take-Profit):**
The predicted High from the primary model, clipped to be at least marginally above entry:

```
upper_barrier = max(pred_high, entry × 1.0001)
```

This directly evaluates the model's price-level forecast: "Did the market actually reach the price the model predicted?"

**Lower Barrier (Stop-Loss):**
Dynamic, volatility-based:

```
daily_vol = rolling_std(log_returns, window=20)
sl_pct = max(daily_vol × sl_multiplier, min_sl_pct)
lower_barrier = entry × (1 - sl_pct)
```

Default: `sl_multiplier=2.0`, `min_sl_pct=0.005`. In high-volatility regimes the stop widens to accommodate larger swings; in calm markets it tightens to protect capital.

**Vertical Barrier (Timeout):**
Maximum holding period in bars (default: 5). If neither TP nor SL is hit, the trade exits at a loss (label = 0).

### 5.2 Label Assignment

The forward scan determines which barrier is touched first:

```
For each signal bar i:
    Look forward over [i+1, i+1+vertical_barrier_periods)
    Find first bar where actual High ≥ upper_barrier  → TP hit
    Find first bar where actual Low  ≤ lower_barrier  → SL hit

    If TP hits first (or same bar): label = 1 ("take_profit")
    If SL hits first:               label = 0 ("stop_loss")
    If neither hits:                label = 0 ("timeout")
```

### 5.3 Event Windows

Each labeled observation records `t_start` (signal bar) and `t_end` (exit bar). These event spans define the time interval during which a trade is "alive" and are consumed by Purged K-Fold Cross-Validation to prevent data leakage.

---

## 6. Feature Engineering

The feature engineering bridge (`scripts/generate_meta_labels.py`) transforms primary model predictions into training data for the meta-classifier. It produces two categories of features:

### 6.1 Market-Context Features

Computed from actual price data at the signal bar, these features describe the market *regime* at the time of the signal:

| Feature | Category | Formula | Interpretation |
|---------|----------|---------|----------------|
| ATR (14) | Volatility | Rolling mean of True Range | Absolute volatility level |
| Rolling Vol (20) | Volatility | Rolling std of simple returns | Scale-normalized volatility |
| RSI (14) | Momentum | 100 - 100/(1 + RS) where RS = avg_gain/avg_loss | Overbought (>70) / oversold (<30) |
| MACD Line | Momentum | EMA(12) - EMA(26) | Short vs medium trend gap |
| MACD Signal | Momentum | EMA(MACD Line, 9) | Smoothed MACD |
| MACD Histogram | Momentum | MACD Line - MACD Signal | Momentum acceleration |
| Vwap | Price level | Volume-weighted average price | Institutional reference |
| Transactions | Activity | Number of trades | Market participation |
| Daily Vol | Volatility | Rolling std of log returns | Used by triple barrier |

### 6.2 Prediction-Derived Features

Computed from the primary model's forecasts:

| Feature | Formula | Interpretation |
|---------|---------|----------------|
| pred_return | (pred_high / close) - 1 | How ambitious is the TP target? |
| pred_close_return | (pred_close / close) - 1 | Expected close price movement |
| pred_ema20_vs_close | (pred_ema20 / close) - 1 | Predicted short-term trend vs current price |
| pred_sma50_vs_close | (pred_sma50 / close) - 1 | Predicted medium-term trend vs current price |
| pred_ma_crossover | pred_ema20 - pred_sma50 | Predicted trend direction |

### 6.3 Data Alignment

Test predictions from the primary model are aligned with the raw CSV using index arithmetic. For test sample `i`:

- Entry bar index: `val_end + i + seq_len - 1` (last bar of the input window)
- The entry bar's actual Close price serves as the trade entry price

For multi-step forecasts (`pred_len > 1`):
- `pred_high`: Maximum predicted High across the entire horizon (most optimistic level)
- `pred_close`: Last-step predicted Close (end-of-horizon estimate)
- `pred_ema20`, `pred_sma50`: Last-step values (where the MAs are heading)

---

## 7. Meta-Classifier (LightGBM)

**Reference:** Ke et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree," NeurIPS 2017.

### 7.1 Purpose

The meta-classifier addresses a fundamental question: "Given that the primary model says BUY here, should we actually take this trade?" It learns to distinguish scenarios where the primary model's predictions are reliable from those where they are not.

### 7.2 Model Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| objective | binary | Binary classification (profitable/unprofitable) |
| boosting_type | gbdt | Gradient-boosted decision trees |
| n_estimators | 500 | Max trees (early stopping controls actual count) |
| learning_rate | 0.05 | Conservative rate for small datasets |
| num_leaves | 31 | Default complexity |
| min_child_samples | 20 | Prevents overfitting on small leaf nodes |
| subsample | 0.8 | Row subsampling per tree |
| colsample_bytree | 0.8 | Feature subsampling per tree |
| reg_alpha | 0.1 | L1 regularization |
| reg_lambda | 0.1 | L2 regularization |
| is_unbalance | True | Handles class imbalance |

### 7.3 Training Procedure

1. **Per-fold training:** Within each Purged K-Fold split, a LightGBM model is trained with early stopping (default 50 rounds) monitoring validation log-loss.
2. **Best iteration averaging:** The optimal number of trees is averaged across folds.
3. **Final retraining:** A single model is retrained on all data using the averaged best iteration count.

### 7.4 Output

- `predict_proba(X)`: Probability that the trade will be profitable (0 to 1).
- `predict(X, threshold)`: Binary decision at the specified threshold (default 0.5).

---

## 8. Purged K-Fold Cross-Validation

**Reference:** Lopez de Prado, *Advances in Financial Machine Learning*, Chapter 7, 2018.

### 8.1 The Data Leakage Problem

Standard K-Fold cross-validation assumes i.i.d. samples, which is violated in financial time series for two reasons:

1. **Overlapping event windows:** A trade opened in the training period may not close until after the test period begins, leaking future information into the training set.
2. **Serial correlation:** Observations immediately adjacent to the test period carry correlated information.

### 8.2 Solution: Purging + Embargo

**Purging:** Any training sample whose event window `[t_start, t_end]` overlaps with the test period `[test_start, test_end]` is removed from the training set:

```
Purge sample i if:  t_start[i] ≤ test_end  AND  t_end[i] ≥ test_start
```

**Embargo:** After purging, an additional `n_embargo` samples following each test fold are removed from training to guard against serial correlation:

```
Remove training samples in [test_end + 1, test_end + n_embargo]
```

### 8.3 Fold Structure

Folds are contiguous time blocks (no shuffling):

```
Fold 0: [████ TEST ████] ── train ── ── train ── ── train ── ── train ──
Fold 1: ── train ── [████ TEST ████] ── train ── ── train ── ── train ──
Fold 2: ── train ── ── train ── [████ TEST ████] ── train ── ── train ──
...
```

Each fold is used once as the test set. The remaining folds are used for training after purging and embargo are applied.

---

## 9. Evaluation Framework

### 9.1 Primary Model Metrics

The primary model is evaluated on the original price scale (after inverse StandardScaler transformation):

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| MSE | mean((pred - true)²) | Average squared error |
| MAE | mean(\|pred - true\|) | Average absolute error in dollars |
| RMSE | √MSE | Error in same units as price |

Metrics are computed both overall and per-target (High, Close, EMA_20, SMA_50).

### 9.2 Meta-Classifier Metrics

**Classification metrics** (per fold and averaged):

- **Accuracy:** Fraction of correctly classified signals.
- **Log-loss:** Cross-entropy loss (used for early stopping).
- **Precision:** Of signals the filter passes, what fraction are truly profitable.
- **Recall:** Of truly profitable signals, what fraction does the filter pass.
- **F1 Score:** Harmonic mean of precision and recall.

**Baseline vs. Filtered comparison:**

- *Baseline:* Take all signals → precision equals the base rate (class balance).
- *Filtered:* Only take signals where `meta_proba ≥ threshold` → precision should exceed baseline.

### 9.3 Probabilistic Sharpe Ratio (PSR)

**Reference:** Bailey and Lopez de Prado, "The Sharpe Ratio Efficient Frontier," Journal of Risk, 2012.

PSR answers: "What is the probability that the observed Sharpe Ratio is greater than a benchmark, given the sample size and the distribution of returns?"

The formula:

```
PSR = Φ( (SR̂ - SR*) × √(T-1) / √(1 - γ₃·SR̂ + ((γ₄-1)/4)·SR̂²) )
```

Where:
- `SR̂` = sample Sharpe Ratio (per-period)
- `SR*` = benchmark Sharpe Ratio (default: 0)
- `T` = number of observations
- `γ₃` = skewness of returns
- `γ₄` = excess kurtosis of returns
- `Φ` = standard normal CDF

**Trade return computation:**

- Profitable trade (label=1): `return = (pred_high / entry) - 1`
- Stop-loss exit: `return = (lower_barrier / entry) - 1`
- Timeout: `return = 0`

---

## 10. Experimental Results

### 10.1 Primary Model Performance (AAPL, TimesNet)

Configuration: `seq_len=30`, `pred_len=5`, `enc_in=7`, `c_out=4`

**Training:** Best epoch 12/100, validation loss 0.0490 (early stopping at epoch 22).

**Test Set Results (Original Price Scale, ~$210 stock):**

| Target | MSE | MAE ($) | RMSE ($) |
|--------|-----|---------|----------|
| Overall | 53.32 | 4.87 | 7.30 |
| High | 83.38 | 6.57 | 9.13 |
| Close | 101.70 | 6.99 | 10.08 |
| **EMA_20** | **11.81** | **2.68** | **3.44** |
| **SMA_50** | **16.38** | **3.24** | **4.05** |

**Key finding:** Moving average targets achieve 2.5-3× lower RMSE than raw price targets, confirming that smoother targets are significantly easier to learn.

### 10.2 Meta-Classifier Performance (AAPL)

Configuration: 5-fold Purged K-Fold, 14 features, 131 samples.

| Metric | Value |
|--------|-------|
| CV Accuracy | 77.9% |
| CV Log-loss | 0.491 |
| Baseline Precision | 48.9% |
| **Filtered Precision** | **93.7%** |
| Signals Passing Filter | 63/131 (48.1%) |

**Feature Importance (gain):**

| Rank | Feature | Gain | Category |
|------|---------|------|----------|
| 1 | pred_return | 63 | Prediction-derived |
| 2 | daily_vol | 24 | Volatility |
| 3 | **pred_ma_crossover** | **20** | **MA-derived (new)** |
| 4 | macd_line | 19 | Momentum |
| 5 | atr | 18 | Volatility |
| 6 | macd_signal | 17 | Momentum |
| 7 | macd_hist | 16 | Momentum |
| 8 | Vwap | 15 | Price level |
| 9 | **pred_sma50_vs_close** | **14** | **MA-derived (new)** |
| 10 | rolling_vol | 13 | Volatility |

**Key finding:** The `pred_ma_crossover` feature (predicted EMA-20 minus predicted SMA-50) ranks 3rd in feature importance, demonstrating that predicted trend direction provides genuinely useful signal for trade filtering.

### 10.3 Label Distribution

| Exit Type | Count | Percentage |
|-----------|-------|------------|
| Take-profit | 64 | 48.9% |
| Timeout | 48 | 36.6% |
| Stop-loss | 19 | 14.5% |

---

## 11. Project Structure

```
stockportfolio_project/
├── data/
│   ├── raw/                    # Raw OHLCV CSV files and parquet data
│   ├── processed/              # Normalized tensors (legacy)
│   └── meta/                   # Meta-label CSVs for LightGBM
├── models/
│   ├── __init__.py             # Model registry
│   ├── base.py                 # Abstract ForecastModel base class
│   ├── TimesNet/               # TimesNet architecture
│   │   ├── model.py            # TimesNetConfig, TimesNetModel
│   │   ├── blocks.py           # TimesBlock, fold_time, unfold_time
│   │   ├── layers.py           # DataEmbedding, InceptionBlockV1
│   │   └── utils.py            # FFT period discovery
│   ├── TimeMixer/              # TimeMixer architecture
│   │   ├── TimeMixer.py        # TimeMixerConfig, TimeMixer
│   │   ├── blocks.py           # PastDecomposableMixing, mixing layers
│   │   └── decomposition.py    # Moving average decomposition
│   ├── TimeMixerpp/            # TimeMixer++ (available, not active)
│   ├── ModernTCN/              # ModernTCN (available, not active)
│   └── meta_classifier/
│       └── lightgbm_model.py   # MetaClassifier (LightGBM wrapper)
├── trading_logic/
│   ├── triple_barrier.py       # Triple Barrier Method
│   ├── purged_cv.py            # PurgedKFold cross-validation
│   └── evaluation.py           # PSR, precision/recall/F1, full eval
├── scripts/
│   ├── fetch_data.py           # Yahoo Finance data download
│   ├── process_data.py         # StandardScaler preprocessing
│   ├── resample_parquet.py     # Minute → daily resampling
│   └── generate_meta_labels.py # Meta-label + feature engineering
├── dataset.py                  # ParquetDataset (YahooDataset)
├── train.py                    # Primary model training
├── train_meta.py               # Meta-classifier training
├── test.py                     # Primary model evaluation
├── utils.py                    # EarlyStopping, metrics, scheduler
├── requirements.txt            # Python dependencies
└── docs/
    ├── META_LABELING_REPORT.md # Meta-labeling methodology report
    └── PROJECT_REPORT.md       # This document
```

---

## 12. Usage Guide

### 12.1 Environment Setup

```bash
pip install -r requirements.txt
```

Required: `torch`, `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `pyarrow`, `scipy`, `joblib`, `yfinance`

### 12.2 Full Pipeline (with Moving Average Targets)

```bash
# Phase 1-2: Train the primary model
python3 train.py --ticker AAPL --ma_targets EMA_20 SMA_50

# Phase 3: Evaluate and save predictions
python3 test.py --ticker AAPL --ma_targets EMA_20 SMA_50 --save_predictions

# Phase 4: Generate meta-labels and features
python3 scripts/generate_meta_labels.py --ticker AAPL \
    --target_names High Close EMA_20 SMA_50

# Phase 5: Train the meta-classifier
python3 train_meta.py --ticker AAPL
```

### 12.3 Key CLI Arguments

| Script | Argument | Default | Description |
|--------|----------|---------|-------------|
| train.py | --ticker | AAPL | Stock ticker |
| train.py | --model | TimesNet | TimeMixer or TimesNet |
| train.py | --seq_len | 30 | Lookback window |
| train.py | --pred_len | 5 | Forecast horizon |
| train.py | --ma_targets | None | EMA_20, SMA_50 |
| train.py | --epochs | 100 | Max training epochs |
| train.py | --patience | 10 | Early stopping patience |
| test.py | --save_predictions | False | Save .npy for meta-labeling |
| generate_meta_labels.py | --target_names | High Close | Ordered target names |
| generate_meta_labels.py | --sl_multiplier | 2.0 | Stop-loss in vol units |
| generate_meta_labels.py | --vertical_barrier | 5 | Max holding period |
| train_meta.py | --n_splits | 5 | Purged K-Fold splits |
| train_meta.py | --n_embargo | 5 | Embargo bars |
| train_meta.py | --threshold | 0.5 | Classification threshold |

### 12.4 Without Moving Average Targets (Backward Compatible)

```bash
python3 train.py --ticker AAPL
python3 test.py --ticker AAPL --save_predictions
python3 scripts/generate_meta_labels.py --ticker AAPL
python3 train_meta.py --ticker AAPL
```

---

## 13. References

1. Wu, H., Hu, T., Liu, Y., Zhou, H., Wang, J., & Long, M. (2023). *TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis.* International Conference on Learning Representations (ICLR).

2. Wang, S., Wu, H., Shi, X., Hu, T., Luo, H., Ma, L., Zhang, J.Y., & Zhou, J. (2024). *TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting.* International Conference on Learning Representations (ICLR).

3. Lopez de Prado, M. (2018). *Advances in Financial Machine Learning.* Wiley.
   - Chapter 3: Triple Barrier Method
   - Chapter 7: Cross-Validation in Finance (Purged K-Fold)

4. Liu, Y., Wu, H., Wang, J., & Long, M. (2022). *Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting.* NeurIPS.

5. Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* NeurIPS.

6. Bailey, D. H., & Lopez de Prado, M. (2012). *The Sharpe Ratio Efficient Frontier.* Journal of Risk, 15(2), 3-44.

---

*Report generated for the Stock Portfolio Forecasting Project. Data period: January 2021 – September 2025 (AAPL daily bars).*
