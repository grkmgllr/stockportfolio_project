"""
Training script for the End-to-End Hybrid Model (TimeMixer + LightGBM).

Loads the dataset using the existing pipeline, extracts target labels
for the LightGBM model (1 for Uptrend, 0 for Downtrend), and trains
the hybrid model end-to-end.  Finally, it evaluates the model's
accuracy, precision, recall, and F1 score on the test set.

Usage
-----
    python train_hybrid.py
"""

import os
import random
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error, mean_absolute_error

from dataset import ParquetDataset
from models.HybridModel import HybridTimeMixerLGBM
from models.TimeMixer.TimeMixer import TimeMixerConfig

# ---------------------------------------------------------------------------
#  Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

def set_seed(seed: int = SEED) -> None:
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
#  Label generation
# ---------------------------------------------------------------------------

def get_multistep_continuous_labels(
    data_loader: DataLoader,
    close_idx_x: int,
    close_idx_y: int,
    dataset: ParquetDataset,
) -> np.ndarray:
    """
    Creates target variables as continuous percentage returns for the entire horizon.

    Returns (future - current) / current for each step in pred_len.

    Args:
        data_loader (DataLoader): The DataLoader to iterate over.
        close_idx_x (int): Channel index for 'Close' in input features.
        close_idx_y (int): Channel index for 'Close' in target features.
        dataset (ParquetDataset): Dataset used to inverse transform.

    Returns:
        np.ndarray: An array of continuous return values of shape [N, pred_len].
    """
    labels = []
    for batch_x, batch_y in data_loader:
        B, S, C_in = batch_x.shape
        _, P, C_out = batch_y.shape

        x_flat = batch_x.reshape(-1, C_in).numpy()
        y_flat = batch_y.reshape(-1, C_out).numpy()

        x_raw = dataset.inverse_transform_x(x_flat).reshape(B, S, C_in)
        y_raw = dataset.inverse_transform_y(y_flat).reshape(B, P, C_out)

        current_close = x_raw[:, -1, close_idx_x][:, None]  # [B, 1]
        future_closes = y_raw[:, :, close_idx_y]            # [B, P]

        # Calculate percentage returns for each step in the horizon
        batch_returns = (future_closes - current_close) / (np.abs(current_close) + 1e-8)
        labels.append(batch_returns)

    return np.concatenate(labels, axis=0) # [Total_samples, pred_len]

def get_current_close(data_loader: DataLoader, close_idx_x: int, dataset: ParquetDataset) -> np.ndarray:
    """Extracts the actual unscaled current close prices for evaluation."""
    current_closes = []
    for batch_x, _ in data_loader:
        B, S, C_in = batch_x.shape
        x_flat = batch_x.reshape(-1, C_in).numpy()
        x_raw = dataset.inverse_transform_x(x_flat).reshape(B, S, C_in)
        current_closes.append(x_raw[:, -1, close_idx_x])
    return np.concatenate(current_closes)


def plot_forecast(ticker: str, actual: np.ndarray, predicted: np.ndarray, sample_idx: int = 0):
    """Saves a plot comparing actual vs predicted prices for a specific test sample."""
    os.makedirs("forecast_plots", exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(actual)+1), actual, marker='o', label='Actual Price', color='blue')
    plt.plot(range(1, len(predicted)+1), predicted, marker='x', linestyle='--', label='Predicted Price', color='red')
    plt.title(f"{ticker} Forecast - Sample {sample_idx}")
    plt.xlabel("Days Ahead")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    filename = os.path.join("forecast_plots", f"forecast_{ticker}.png")
    plt.savefig(filename)
    plt.close()
    print(f"\n>>> Forecast plot saved as {filename}")

def main() -> None:
    """
    Main function to execute the end-to-end hybrid model training pipeline.
    """
    set_seed(SEED)

    ticker = "NVDA"
    data_root = "data/raw"
    seq_len = 30
    pred_len = 5

    # Use only core OHLCV features (Vwap/Transactions add noise on small data)
    ohlcv_features = ['Open', 'High', 'Low', 'Close', 'Volume']
    # Add moving average targets so TimeMixer also learns smoother trend signals
    ma_targets = ['EMA_20', 'SMA_50']
    batch_size = 32
    epochs = 30

    print(f"\n{'=' * 60}")
    print(f"  END-TO-END HYBRID MODEL TRAINING ({ticker})")
    print(f"{'=' * 60}")

    # ------------------------------------------------------------------
    # 1. Load Data (train / val / test)
    # ------------------------------------------------------------------
    print("\n>>> Loading Datasets...")
    ds_kwargs = dict(
        ticker=ticker, root_path=data_root,
        seq_len=seq_len, pred_len=pred_len,
        input_features=ohlcv_features,
        ma_targets=ma_targets,
    )
    train_dataset = ParquetDataset(flag='train', **ds_kwargs)
    val_dataset   = ParquetDataset(flag='val',   **ds_kwargs)
    test_dataset  = ParquetDataset(flag='test',  **ds_kwargs)

    # shuffle=False is required so that extracted features align with labels
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    # ------------------------------------------------------------------
    # 2. Resolve correct Close indices for input & target feature lists
    # ------------------------------------------------------------------
    input_features = train_dataset.input_features
    target_features = train_dataset.target_features
    print(f"Input Features:  {input_features}")
    print(f"Target Features: {target_features}")

    close_idx_x = input_features.index("Close")
    close_idx_y = target_features.index("Close")
    print(f"Close index in batch_x (input):  {close_idx_x}")
    print(f"Close index in batch_y (target): {close_idx_y}")

    # ------------------------------------------------------------------
    # 3. Generate multi-step continuous labels (Percentage Returns)
    # ------------------------------------------------------------------
    print("\n>>> Generating continuous return labels for forecast horizon...")
    y_train = get_multistep_continuous_labels(train_loader, close_idx_x, close_idx_y, train_dataset)
    y_val   = get_multistep_continuous_labels(val_loader,   close_idx_x, close_idx_y, val_dataset)
    y_test  = get_multistep_continuous_labels(test_loader,  close_idx_x, close_idx_y, test_dataset)

    for name, y in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        print(f"  {name:5s}: {len(y):4d} samples -> Labels shape: {y.shape}")

    # ------------------------------------------------------------------
    # 4. Configure & build hybrid model
    # ------------------------------------------------------------------
    tm_config = TimeMixerConfig(
        historical_lookback_length=seq_len,
        forecast_horizon_length=pred_len,
        number_of_input_features=train_dataset.enc_in,
        number_of_output_features=train_dataset.c_out,
        model_embedding_dimension=32,
        number_of_pdm_blocks=1,
        dropout_probability=0.2,
    )

    hybrid_model = HybridTimeMixerLGBM(timemixer_config=tm_config)

    # ------------------------------------------------------------------
    # 5. Train end-to-end (with validation for early stopping)
    # ------------------------------------------------------------------
    hybrid_model.fit(
        train_loader,
        y_labels=y_train,
        val_loader=val_loader,
        y_val=y_val,
        tm_epochs=epochs,
    )

    # ------------------------------------------------------------------
    # 6. Evaluate multi-step forecast on test set
    # ------------------------------------------------------------------
    print("\n>>> Predicting on Test Data...")
    preds_return = hybrid_model.predict(test_loader)  # [N, pred_len]
    
    # Reconstruct actual prices
    test_current_close = get_current_close(test_loader, close_idx_x, test_dataset) # [N]
    
    # prices = current * (1 + return)
    pred_prices = test_current_close[:, None] * (1 + preds_return)
    actual_prices = test_current_close[:, None] * (1 + y_test)
    
    print(f"\n{'=' * 60}")
    print(f"  RESULTS: {ticker} – Multi-Step Forecasting Metrics")
    print(f"{'=' * 60}")
    
    # Calculate overall errors
    mse = mean_squared_error(actual_prices, pred_prices)
    mae = mean_absolute_error(actual_prices, pred_prices)
    
    print(f"  Overall Price MSE: {mse:.4f}")
    print(f"  Overall Price MAE: {mae:.4f}")
    
    # Print a sample comparison table
    print("\n>>> Sample Price Predictions (Day 5):")
    print(f"{'Today ($)':>12} | {'Actual ($)':>12} | {'Pred ($)':>12} | {'Diff ($)':>10}")
    print("-" * 55)
    for i in range(min(10, len(pred_prices))):
        today = test_current_close[i]
        act = actual_prices[i, -1]
        prd = pred_prices[i, -1]
        diff = prd - act
        print(f"{today:12.2f} | {act:12.2f} | {prd:12.2f} | {diff:+10.2f}")

    # Plot the first sample in the test set
    plot_forecast(ticker, actual_prices[0], pred_prices[0])

    # Calculate Directional Accuracy for the final day
    pred_classes_final = (preds_return[:, -1] > 0).astype(int)
    y_test_classes_final = (y_test[:, -1] > 0).astype(int)
    
    acc  = accuracy_score(y_test_classes_final, pred_classes_final)
    prec = precision_score(y_test_classes_final, pred_classes_final, zero_division=0)
    rec  = recall_score(y_test_classes_final, pred_classes_final, zero_division=0)
    f1   = f1_score(y_test_classes_final, pred_classes_final, zero_division=0)
    cm   = confusion_matrix(y_test_classes_final, pred_classes_final)

    print(f"\n  Final Day (Day {pred_len}) Directional Metrics:")
    print(f"  Accuracy:  {acc*100:.2f}%")
    print(f"  Precision: {prec*100:.2f}%")
    print(f"  Recall:    {rec*100:.2f}%")
    print(f"  F1 Score:  {f1*100:.2f}%")
    
    print("\n  Confusion Matrix (Final Day):")
    print(f"               Pred Down  Pred Up")
    if cm.shape == (2, 2):
        print(f"  Actual Down {cm[0, 0]:8d} {cm[0, 1]:8d}")
        print(f"  Actual Up   {cm[1, 0]:8d} {cm[1, 1]:8d}")
    else:
        print(f"  {cm}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
