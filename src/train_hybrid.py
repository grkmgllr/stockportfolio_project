import os, sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import random
import argparse
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error, mean_absolute_error, balanced_accuracy_score
from scipy.stats import pearsonr, spearmanr

from dataset import ParquetDataset
from models.HybridModel.HybridModel import HybridTimeMixerLGBM
from models.TimeMixer.TimeMixer import TimeMixerConfig

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100

def smape(y_true, y_pred):
    return np.mean(2.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100

def get_multistep_continuous_labels(data_loader, close_idx_x, close_idx_y, dataset):
    labels = []
    for batch_x, batch_y in data_loader:
        B, S, C_in = batch_x.shape
        _, P, C_out = batch_y.shape
        x_raw = dataset.inverse_transform_x(batch_x.reshape(-1, C_in).numpy()).reshape(B, S, C_in)
        y_raw = dataset.inverse_transform_y(batch_y.reshape(-1, C_out).numpy()).reshape(B, P, C_out)
        current_close = x_raw[:, -1, close_idx_x][:, None]
        future_closes = y_raw[:, :, close_idx_y]
        batch_returns = (future_closes - current_close) / (np.abs(current_close) + 1e-8)
        labels.append(batch_returns)
    return np.concatenate(labels, axis=0)

def get_current_close(data_loader, close_idx_x, dataset):
    current_closes = []
    for batch_x, _ in data_loader:
        B, S, C_in = batch_x.shape
        x_raw = dataset.inverse_transform_x(batch_x.reshape(-1, C_in).numpy()).reshape(B, S, C_in)
        current_closes.append(x_raw[:, -1, close_idx_x])
    return np.concatenate(current_closes)

def get_actual_future_closes(data_loader, close_idx_y, dataset):
    future_closes = []
    for _, batch_y in data_loader:
        _, P, C_out = batch_y.shape
        y_raw = dataset.inverse_transform_y(batch_y.reshape(-1, C_out).numpy()).reshape(-1, P, C_out)
        future_closes.append(y_raw[:, :, close_idx_y])
    return np.concatenate(future_closes, axis=0)

def get_past_5_day_returns(data_loader, close_idx_x, dataset):
    past_returns = []
    for batch_x, _ in data_loader:
        B, S, C_in = batch_x.shape
        x_raw = dataset.inverse_transform_x(batch_x.reshape(-1, C_in).numpy()).reshape(B, S, C_in)
        ret = (x_raw[:, -1, close_idx_x] - x_raw[:, -6, close_idx_x]) / (np.abs(x_raw[:, -6, close_idx_x]) + 1e-8)
        past_returns.append(ret)
    return np.concatenate(past_returns)

def check_scalers(train_dataset, val_dataset, test_dataset, verbose=False):
    if not verbose:
        return
    print("\n--- Diagnostic: Scaler Information ---")
    print("Train X Scaler ID:", id(train_dataset.scaler_x), "| Y Scaler ID:", id(train_dataset.scaler_y))
    print("Val X Scaler ID:  ", id(val_dataset.scaler_x), "| Y Scaler ID:", id(val_dataset.scaler_y))
    print("Test X Scaler ID: ", id(test_dataset.scaler_x), "| Y Scaler ID:", id(test_dataset.scaler_y))
    
    if hasattr(train_dataset.scaler_x, 'mean_'):
        print("\nTrain X Mean (first 5):", train_dataset.scaler_x.mean_[:5])
        print("Val X Mean (first 5):  ", val_dataset.scaler_x.mean_[:5])
        print("Test X Mean (first 5): ", test_dataset.scaler_x.mean_[:5])
        
        print("\nTrain Y Mean:", train_dataset.scaler_y.mean_)
        print("Val Y Mean:  ", val_dataset.scaler_y.mean_)
        print("Test Y Mean: ", test_dataset.scaler_y.mean_)
        
        same_x = np.allclose(train_dataset.scaler_x.mean_, test_dataset.scaler_x.mean_)
        same_y = np.allclose(train_dataset.scaler_y.mean_, test_dataset.scaler_y.mean_)
        print(f"\nTrain/Test X Scaler Means Match: {same_x}")
        print(f"Train/Test Y Scaler Means Match: {same_y}")
        if not (same_x and same_y):
             print("WARNING: Scalers might be fitted separately for splits! (Data Leakage or Distribution Shift issue)")
    else:
        print("WARNING: Scaler lacks mean_ attribute, might not be fitted yet.")

def check_label_alignment(test_loader, close_idx_x, close_idx_y, dataset, y_test, test_current_close, verbose=False):
    actual_future_closes = get_actual_future_closes(test_loader, close_idx_y, dataset)
    reconstructed_prices = test_current_close[:, None] * (1 + y_test)
    
    diff = np.abs(actual_future_closes - reconstructed_prices).max()
    if diff > 1e-3 or verbose:
        print("\n--- Diagnostic: Label/Feature Alignment ---")
        print(f"Max diff between actual Close and reconstructed Close: {diff:.6f}")
        if diff > 1e-3:
            print("WARNING: Label alignment mismatch detected!")
            
    if verbose:
        print("\nFirst 5 samples alignment:")
        for i in range(min(5, len(test_current_close))):
            print(f" Sample {i+1}:")
            print(f"  Current Close: {test_current_close[i]:.4f}")
            print(f"  Actual Future: {actual_future_closes[i]}")
            print(f"  y_test Ret:    {y_test[i]}")
            print(f"  Reconstructed: {reconstructed_prices[i]}")

def build_baseline_predictions(test_current_close, train_returns, past_5_day_returns, pred_len):
    N = len(test_current_close)
    # Persistence
    persistence_prices = np.tile(test_current_close[:, None], (1, pred_len))
    persistence_returns = np.zeros_like(persistence_prices) # 0 return
    
    # Drift
    avg_train_daily_return = np.mean(train_returns)
    drift_prices = np.zeros((N, pred_len))
    drift_returns = np.zeros((N, pred_len))
    for h in range(pred_len):
        drift_prices[:, h] = test_current_close * ((1 + avg_train_daily_return) ** (h + 1))
        drift_returns[:, h] = ((1 + avg_train_daily_return) ** (h + 1)) - 1
        
    # Momentum
    momentum_prices = np.zeros((N, pred_len))
    momentum_returns = np.zeros((N, pred_len))
    avg_daily_momentum = past_5_day_returns / 5.0
    for h in range(pred_len):
        momentum_prices[:, h] = test_current_close * ((1 + avg_daily_momentum) ** (h + 1))
        momentum_returns[:, h] = ((1 + avg_daily_momentum) ** (h + 1)) - 1
        
    return persistence_prices, drift_prices, momentum_prices, persistence_returns, drift_returns, momentum_returns

def evaluate_price_forecasts(actual_prices, pred_prices, actual_returns, pred_returns, pred_len, prefix):
    metrics = {}
    metrics[f'{prefix}_MAE'] = mean_absolute_error(actual_prices, pred_prices)
    metrics[f'{prefix}_RMSE'] = np.sqrt(mean_squared_error(actual_prices, pred_prices))
    metrics[f'{prefix}_MAPE'] = mape(actual_prices, pred_prices)
    metrics[f'{prefix}_sMAPE'] = smape(actual_prices, pred_prices)
    
    for h in range(pred_len):
        metrics[f'{prefix}_MAE_day_{h+1}'] = mean_absolute_error(actual_prices[:, h], pred_prices[:, h])
        metrics[f'{prefix}_RMSE_day_{h+1}'] = np.sqrt(mean_squared_error(actual_prices[:, h], pred_prices[:, h]))
        
        act_up = (actual_returns[:, h] > 0).astype(int)
        prd_up = (pred_returns[:, h] > 0).astype(int)
        metrics[f'{prefix}_DirAcc_day_{h+1}'] = accuracy_score(act_up, prd_up)
        
        metrics[f'{prefix}_Mean_ActRet_day_{h+1}'] = np.mean(actual_returns[:, h])
        metrics[f'{prefix}_Mean_PrdRet_day_{h+1}'] = np.mean(pred_returns[:, h])
        
        if np.std(pred_returns[:, h]) > 0 and np.std(actual_returns[:, h]) > 0:
            metrics[f'{prefix}_Corr_day_{h+1}'] = np.corrcoef(actual_returns[:, h], pred_returns[:, h])[0, 1]
        else:
            metrics[f'{prefix}_Corr_day_{h+1}'] = 0.0
        
    # Final day specific
    act_up = (actual_returns[:, -1] > 0).astype(int)
    prd_up = (pred_returns[:, -1] > 0).astype(int)
    metrics[f'{prefix}_Final_Precision'] = precision_score(act_up, prd_up, zero_division=0)
    metrics[f'{prefix}_Final_Recall'] = recall_score(act_up, prd_up, zero_division=0)
    metrics[f'{prefix}_Final_F1'] = f1_score(act_up, prd_up, zero_division=0)
    metrics[f'{prefix}_Final_PredUpRate'] = np.mean(prd_up)
    metrics[f'{prefix}_Final_ActUpRate'] = np.mean(act_up)
    
    return metrics

def calibrate_returns_on_validation(y_val, val_preds_return, ticker_dir):
    pred_len = y_val.shape[1]
    params_list = []
    alphas = np.linspace(0.0, 1.0, 11)
    
    for h in range(pred_len):
        act = y_val[:, h]
        prd = val_preds_return[:, h]
        
        bias = np.mean(prd - act)
        val_mae_before = mean_absolute_error(act, prd)
        val_pred_up_rate_before = np.mean(prd > 0)
        val_actual_up_rate = np.mean(act > 0)
        
        candidates = []
        
        # RAW_NO_CALIBRATION
        candidates.append({
            'method': 'RAW_NO_CALIBRATION',
            'alpha': 1.0,
            'bias': 0.0,
            'pred': prd
        })
        
        # Method A: Bias correction
        candidates.append({
            'method': 'A_bias_only',
            'alpha': 1.0,
            'bias': bias,
            'pred': prd - bias
        })
        
        # Method B: Shrinkage
        for alpha in alphas:
            candidates.append({
                'method': 'B_shrinkage_only',
                'alpha': alpha,
                'bias': 0.0,
                'pred': alpha * prd
            })
            
        # Method C: Bias + Shrinkage
        for alpha in alphas:
            candidates.append({
                'method': 'C_bias_and_shrinkage',
                'alpha': alpha,
                'bias': bias,
                'pred': alpha * (prd - bias)
            })
            
        best_mae = float('inf')
        best_candidate = None
        
        for cand in candidates:
            c_pred = cand['pred']
            c_mae = mean_absolute_error(act, c_pred)
            c_up_rate = np.mean(c_pred > 0)
            
            # 1. Direction collapse check
            is_collapsed = (c_up_rate <= 0.05 or c_up_rate >= 0.95)
            if is_collapsed:
                if abs(c_up_rate - val_actual_up_rate) <= 0.30:
                    is_collapsed = False
            
            cand['direction_collapse'] = is_collapsed
            cand['mae'] = c_mae
            cand['up_rate'] = c_up_rate
            cand['mae_improvement_pct'] = (val_mae_before - c_mae) / val_mae_before * 100 if val_mae_before > 0 else 0
            
            # 2. Rejection criteria
            if cand['method'] != 'RAW_NO_CALIBRATION':
                if is_collapsed:
                    cand['rejected_reason'] = "Direction Collapse"
                    cand['selected'] = False
                    continue
                if c_mae > val_mae_before * 0.995:
                    cand['rejected_reason'] = "Insufficient MAE Improvement"
                    cand['selected'] = False
                    continue
            
            cand['rejected_reason'] = "None"
            cand['selected'] = True
            
            # Select best among valid ones
            if c_mae < best_mae:
                best_mae = c_mae
                best_candidate = cand

        if best_candidate is None:
            best_candidate = candidates[0]
            best_candidate['rejected_reason'] = "Fallback"
            best_candidate['selected'] = True

        params_list.append({
            'horizon': h + 1,
            'method': best_candidate['method'],
            'bias': best_candidate['bias'],
            'alpha': best_candidate['alpha'],
            'selected': best_candidate['selected'],
            'rejected_reason': best_candidate['rejected_reason'],
            'direction_collapse': best_candidate['direction_collapse'],
            'val_mae_improvement_pct': best_candidate['mae_improvement_pct'],
            'val_mae_before': val_mae_before,
            'val_mae_after': best_candidate['mae'],
            'val_pred_up_rate_before': val_pred_up_rate_before,
            'val_pred_up_rate_after': best_candidate['up_rate'],
            'val_actual_up_rate': val_actual_up_rate
        })
        
    df_params = pd.DataFrame(params_list)
    df_params.to_csv(os.path.join(ticker_dir, 'calibration_params.csv'), index=False)
    return params_list

def apply_calibration(preds_return_raw, calibration_params):
    calibrated = np.zeros_like(preds_return_raw)
    for i, params in enumerate(calibration_params):
        if params['method'] == 'RAW_NO_CALIBRATION':
            calibrated[:, i] = preds_return_raw[:, i]
        elif params['method'] == 'A_bias_only':
            calibrated[:, i] = preds_return_raw[:, i] - params['bias']
        elif params['method'] == 'B_shrinkage_only':
            calibrated[:, i] = params['alpha'] * preds_return_raw[:, i]
        elif params['method'] == 'C_bias_and_shrinkage':
            calibrated[:, i] = params['alpha'] * (preds_return_raw[:, i] - params['bias'])
    return calibrated

def select_direction_thresholds_on_validation(y_val, val_direction_proba, ticker_dir):
    pred_len = y_val.shape[1]
    thresholds_to_search = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
    results = []
    
    for h in range(pred_len):
        act_dir = (y_val[:, h] > 0).astype(int)
        proba = val_direction_proba[:, h]
        
        best_bal_acc = -1.0
        best_threshold = 0.50
        
        if len(np.unique(act_dir)) < 2:
            best_threshold = 0.50
        else:
            best_candidates = []
            for th in thresholds_to_search:
                preds = (proba >= th).astype(int)
                bal_acc = balanced_accuracy_score(act_dir, preds)
                if bal_acc > best_bal_acc:
                    best_bal_acc = bal_acc
                    best_candidates = [th]
                elif bal_acc == best_bal_acc:
                    best_candidates.append(th)
                    
            if len(best_candidates) > 0:
                best_threshold = sorted(best_candidates, key=lambda x: abs(x - 0.50))[0]
                
        preds = (proba >= best_threshold).astype(int)
        results.append({
            'horizon': h + 1,
            'threshold': best_threshold,
            'val_balanced_accuracy': balanced_accuracy_score(act_dir, preds) if len(np.unique(act_dir)) > 1 else 0.0,
            'val_accuracy': accuracy_score(act_dir, preds),
            'val_pred_up_rate': np.mean(preds),
            'val_actual_up_rate': np.mean(act_dir),
            'val_precision': precision_score(act_dir, preds, zero_division=0),
            'val_recall': recall_score(act_dir, preds, zero_division=0),
            'val_f1': f1_score(act_dir, preds, zero_division=0)
        })
        
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(ticker_dir, 'direction_thresholds.csv'), index=False)
    return results

def plot_diagnostics(ticker, out_dir, actual_prices, hybrid_prices_raw, hybrid_prices_calibrated, persistence_prices, actual_returns, hybrid_returns_raw, hybrid_returns_calibrated, pred_len, hybrid_prices_dir_gated=None, pred_prices_final_selected=None):
    plot_dir = os.path.join(out_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    # 1. Actual vs Predicted Final-Day Prices
    plt.figure(figsize=(10, 5))
    plt.plot(actual_prices[:, -1], label='Actual', color='blue', alpha=0.7)
    plt.plot(hybrid_prices_raw[:, -1], label='Hybrid Raw', color='orange', alpha=0.7)
    plt.plot(hybrid_prices_calibrated[:, -1], label='Hybrid Calibrated', color='red', alpha=0.7)
    if hybrid_prices_dir_gated is not None:
        plt.plot(hybrid_prices_dir_gated[:, -1], label='Hybrid Gated', color='purple', alpha=0.7)
    if pred_prices_final_selected is not None:
        plt.plot(pred_prices_final_selected[:, -1], label='Final Selected Blend', color='cyan', alpha=0.9, linestyle='--')
    plt.plot(persistence_prices[:, -1], label='Persistence', color='green', linestyle='--', alpha=0.7)
    plt.title(f"{ticker} Final Day Forecasts")
    plt.legend()
    plt.savefig(os.path.join(plot_dir, f"{ticker}_final_day_prices.png"))
    plt.close()
    
    # 2. Raw vs Calibrated final-day predicted returns
    plt.figure(figsize=(6, 6))
    plt.scatter(hybrid_returns_raw[:, -1], hybrid_returns_calibrated[:, -1], alpha=0.5)
    plt.plot([-1, 1], [-1, 1], color='k', linestyle='--', alpha=0.3)
    plt.xlabel('Raw Predicted Return')
    plt.ylabel('Calibrated Predicted Return')
    plt.title(f"{ticker} Day {pred_len} Returns: Raw vs Calibrated")
    plt.savefig(os.path.join(plot_dir, f"{ticker}_scatter_day_{pred_len}_raw_vs_calibrated.png"))
    plt.close()
    
    # 3. Confusion Matrix Heatmap for Final-Day Direction
    act_up = (actual_returns[:, -1] > 0).astype(int)
    prd_up = (hybrid_returns_calibrated[:, -1] > 0).astype(int)
    cm = confusion_matrix(act_up, prd_up)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
    plt.xlabel('Calibrated Predicted')
    plt.ylabel('Actual')
    plt.title(f"{ticker} Final Day Calibrated Direction CM")
    plt.savefig(os.path.join(plot_dir, f"{ticker}_confusion_matrix_calibrated.png"))
    plt.close()

def save_diagnostics(metrics, out_dir, ticker):
    # Summary CSV
    df = pd.DataFrame([metrics])
    df.to_csv(os.path.join(out_dir, 'metrics_summary.csv'), index=False)
    
    # Per horizon CSV
    horizon_data = []
    pred_len = 5
    for k in metrics.keys():
        if 'Persistence_MAE_day_' in k:
            pred_len = max(pred_len, int(k.split('_')[-1]))
            
    for h in range(1, pred_len + 1):
        row = {'Day': h}
        for prefix in ['Hybrid_Raw', 'Hybrid_Calibrated', 'Hybrid_DirectionGated', 'Hybrid_DirectionGated_Conservative', 'Persistence', 'Drift', 'Momentum']:
            for m in ['MAE', 'RMSE', 'DirAcc', 'Mean_ActRet', 'Mean_PrdRet', 'Corr']:
                key = f'{prefix}_{m}_day_{h}'
                if key in metrics:
                    row[f'{prefix}_{m}'] = metrics[key]
        horizon_data.append(row)
    pd.DataFrame(horizon_data).to_csv(os.path.join(out_dir, 'per_horizon_metrics.csv'), index=False)

def select_final_blend_on_validation(y_val, val_current_close, candidate_return_predictions, pred_len, ticker_dir, min_improvement_pct=1.0, min_blocks_beating_baseline=2, worst_block_tolerance_pct=-2.0):
    alphas = np.linspace(0.0, 1.0, 11)
    all_candidates_results = []
    selected_variants_results = []
    
    priority_map = {
        'Persistence': 1, 'Drift': 2, 'Momentum': 3,
        'Hybrid_Calibrated': 4, 'Hybrid_Raw': 5,
        'Blend_HybridCalibrated_Persistence': 6, 'Blend_HybridCalibrated_Drift': 7,
        'Blend_HybridRaw_Persistence': 8, 'Blend_HybridRaw_Drift': 9,
    }
    
    for h in range(pred_len):
        act_ret = y_val[:, h]
        act_price = val_current_close * (1 + act_ret)
        val_actual_up_rate = np.mean(act_ret > 0)
        
        N = len(y_val)
        block_size = N // 3
        
        horizon_candidates = []
        for name, preds in candidate_return_predictions.items():
            if preds is None: continue
            pred_ret = preds[:, h]
            pred_price = val_current_close * (1 + pred_ret)
            
            c_price_mae = np.mean(np.abs(act_price - pred_price))
            c_ret_mae = np.mean(np.abs(act_ret - pred_ret))
            c_pred_up = (pred_ret > 0).astype(int)
            c_actual_up = (act_ret > 0).astype(int)
            c_dir_acc = np.mean(c_pred_up == c_actual_up)
            c_pred_up_rate = np.mean(c_pred_up)
            
            b_maes = []
            for i in range(3):
                start = i * block_size
                end = (i + 1) * block_size if i < 2 else N
                b_maes.append(np.mean(np.abs(act_price[start:end] - pred_price[start:end])))
            
            horizon_candidates.append({
                'horizon': h + 1, 'candidate_name': name,
                'hybrid_source': name if 'Hybrid' in name else 'None',
                'baseline_source': name if 'Hybrid' not in name else 'None',
                'alpha': 1.0 if 'Hybrid' in name else 0.0,
                'val_price_mae': c_price_mae, 'val_return_mae': c_ret_mae,
                'val_directional_accuracy': c_dir_acc, 'val_pred_up_rate': c_pred_up_rate,
                'val_actual_up_rate': val_actual_up_rate,
                'block_1_price_mae': b_maes[0], 'block_2_price_mae': b_maes[1], 'block_3_price_mae': b_maes[2],
                'priority': priority_map.get(name, 99)
            })
            
        hybrid_sources = ['Hybrid_Calibrated', 'Hybrid_Raw']
        baseline_sources = ['Persistence', 'Drift']
        for hs in hybrid_sources:
            if candidate_return_predictions.get(hs) is None: continue
            for bs in baseline_sources:
                if candidate_return_predictions.get(bs) is None: continue
                cand_name = f"Blend_{hs}_{bs}"
                for alpha in alphas:
                    if alpha == 0.0 or alpha == 1.0: continue
                    pred_ret = alpha * candidate_return_predictions[hs][:, h] + (1 - alpha) * candidate_return_predictions[bs][:, h]
                    pred_price = val_current_close * (1 + pred_ret)
                    c_price_mae = np.mean(np.abs(act_price - pred_price))
                    c_ret_mae = np.mean(np.abs(act_ret - pred_ret))
                    c_pred_up = (pred_ret > 0).astype(int)
                    c_actual_up = (act_ret > 0).astype(int)
                    c_dir_acc = np.mean(c_pred_up == c_actual_up)
                    c_pred_up_rate = np.mean(c_pred_up)
                    
                    b_maes = []
                    for i in range(3):
                        start = i * block_size
                        end = (i + 1) * block_size if i < 2 else N
                        b_maes.append(np.mean(np.abs(act_price[start:end] - pred_price[start:end])))
                        
                    horizon_candidates.append({
                        'horizon': h + 1, 'candidate_name': cand_name,
                        'hybrid_source': hs, 'baseline_source': bs, 'alpha': alpha,
                        'val_price_mae': c_price_mae, 'val_return_mae': c_ret_mae,
                        'val_directional_accuracy': c_dir_acc, 'val_pred_up_rate': c_pred_up_rate,
                        'val_actual_up_rate': val_actual_up_rate,
                        'block_1_price_mae': b_maes[0], 'block_2_price_mae': b_maes[1], 'block_3_price_mae': b_maes[2],
                        'priority': priority_map.get(cand_name, 99)
                    })
                    
        simple_baselines = [c for c in horizon_candidates if c['candidate_name'] in ['Persistence', 'Drift', 'Momentum']]
        best_sb = min(simple_baselines, key=lambda x: x['val_price_mae'])
        best_sb_mae = best_sb['val_price_mae']
        best_sb_dir = best_sb['val_directional_accuracy']
        best_sb_up_diff = abs(best_sb['val_pred_up_rate'] - val_actual_up_rate)
        
        valid_hybrid_candidates = []
        for cand in horizon_candidates:
            cand['is_simple_baseline'] = cand['candidate_name'] in ['Persistence', 'Drift', 'Momentum']
            cand['is_hybrid_based'] = not cand['is_simple_baseline']
            cand['best_simple_baseline_name'] = best_sb['candidate_name']
            cand['best_simple_baseline_val_price_mae'] = best_sb_mae
            
            cand['block_1_baseline_price_mae'] = best_sb['block_1_price_mae']
            cand['block_2_baseline_price_mae'] = best_sb['block_2_price_mae']
            cand['block_3_baseline_price_mae'] = best_sb['block_3_price_mae']
            
            imp_pct = (best_sb_mae - cand['val_price_mae']) / best_sb_mae * 100 if best_sb_mae > 0 else 0
            cand['improvement_vs_best_simple_baseline_pct'] = imp_pct
            
            b1_imp = (best_sb['block_1_price_mae'] - cand['block_1_price_mae']) / best_sb['block_1_price_mae'] * 100 if best_sb['block_1_price_mae'] > 0 else 0
            b2_imp = (best_sb['block_2_price_mae'] - cand['block_2_price_mae']) / best_sb['block_2_price_mae'] * 100 if best_sb['block_2_price_mae'] > 0 else 0
            b3_imp = (best_sb['block_3_price_mae'] - cand['block_3_price_mae']) / best_sb['block_3_price_mae'] * 100 if best_sb['block_3_price_mae'] > 0 else 0
            
            cand['block_1_improvement_pct_vs_best_baseline'] = b1_imp
            cand['block_2_improvement_pct_vs_best_baseline'] = b2_imp
            cand['block_3_improvement_pct_vs_best_baseline'] = b3_imp
            
            cand['num_blocks_beating_baseline'] = sum(1 for i in [b1_imp, b2_imp, b3_imp] if i > 0)
            cand['worst_block_improvement_pct'] = min([b1_imp, b2_imp, b3_imp])
            cand['mean_block_improvement_pct'] = np.mean([b1_imp, b2_imp, b3_imp])
            
            if cand['is_hybrid_based']:
                passes_mae = imp_pct >= min_improvement_pct
                passes_dir = (cand['val_directional_accuracy'] >= best_sb_dir) or \
                             (abs(cand['val_pred_up_rate'] - val_actual_up_rate) <= best_sb_up_diff)
                passes_blocks = cand['num_blocks_beating_baseline'] >= min_blocks_beating_baseline
                passes_worst = cand['worst_block_improvement_pct'] >= worst_block_tolerance_pct
                
                cand['passes_stability_guard'] = passes_blocks and passes_worst
                cand['passes_mae_guard'] = passes_mae
                cand['passes_direction_guard'] = passes_dir
                
                if not passes_mae:
                    cand['rejected_reason'] = f"MAE Imp < {min_improvement_pct}%"
                elif not passes_dir:
                    cand['rejected_reason'] = "Fails Direction Guard"
                elif not cand['passes_stability_guard']:
                    if not passes_blocks:
                        cand['rejected_reason'] = f"Num Blocks < {min_blocks_beating_baseline}"
                    else:
                        cand['rejected_reason'] = f"Worst Block < {worst_block_tolerance_pct}%"
                else:
                    cand['rejected_reason'] = "None"
                    valid_hybrid_candidates.append(cand)
            else:
                cand['passes_mae_guard'] = True
                cand['passes_direction_guard'] = True
                cand['passes_stability_guard'] = True
                cand['rejected_reason'] = "None"
                
        if len(valid_hybrid_candidates) > 0:
            best_price_mae = float('inf')
            best_meta = None
            for cand in valid_hybrid_candidates:
                is_best = False
                if cand['val_price_mae'] < best_price_mae - 1e-4:
                    is_best = True
                elif abs(cand['val_price_mae'] - best_price_mae) <= 1e-4:
                    if cand['priority'] < best_meta['priority']:
                        is_best = True
                if is_best or best_meta is None:
                    best_price_mae = cand['val_price_mae']
                    best_meta = cand
            best_meta['selected_by_guardrail'] = True
            best_meta['fallback_to_baseline'] = False
        else:
            best_meta = best_sb
            best_meta['selected_by_guardrail'] = True
            best_meta['fallback_to_baseline'] = True
            
        for cand in horizon_candidates:
            cand['selected'] = (cand['candidate_name'] == best_meta['candidate_name'] and cand.get('alpha') == best_meta.get('alpha'))
            cand.pop('priority', None)
            all_candidates_results.append(cand)
        selected_variants_results.append(best_meta)
        
    pd.DataFrame(all_candidates_results).to_csv(os.path.join(ticker_dir, "final_blend_candidates.csv"), index=False)
    pd.DataFrame(selected_variants_results).to_csv(os.path.join(ticker_dir, "final_blend_selection.csv"), index=False)
    return selected_variants_results

def select_final_safe_on_validation(y_val, val_current_close, candidate_return_predictions, pred_len, ticker_dir, min_improvement_pct=3.0, required_blocks=3):
    alphas = np.linspace(0.0, 1.0, 11)
    selected_variants_results = []
    
    for h in range(pred_len):
        act_ret = y_val[:, h]
        act_price = val_current_close * (1 + act_ret)
        
        N = len(y_val)
        block_size = N // 3
        
        horizon_candidates = []
        for name, preds in candidate_return_predictions.items():
            if preds is None: continue
            pred_ret = preds[:, h]
            pred_price = val_current_close * (1 + pred_ret)
            
            c_price_mae = np.mean(np.abs(act_price - pred_price))
            c_pred_up = (pred_ret > 0).astype(int)
            c_actual_up = (act_ret > 0).astype(int)
            c_dir_acc = np.mean(c_pred_up == c_actual_up)
            
            b_maes = []
            for i in range(3):
                start = i * block_size
                end = (i + 1) * block_size if i < 2 else N
                b_maes.append(np.mean(np.abs(act_price[start:end] - pred_price[start:end])))
            
            horizon_candidates.append({
                'candidate_name': name,
                'hybrid_source': name if 'Hybrid' in name else 'None',
                'baseline_source': name if 'Hybrid' not in name else 'None',
                'alpha': 1.0 if 'Hybrid' in name else 0.0,
                'val_price_mae': c_price_mae,
                'val_directional_accuracy': c_dir_acc,
                'block_1_price_mae': b_maes[0], 'block_2_price_mae': b_maes[1], 'block_3_price_mae': b_maes[2]
            })
            
        hybrid_sources = ['Hybrid_Calibrated', 'Hybrid_Raw']
        baseline_sources = ['Persistence', 'Drift']
        for hs in hybrid_sources:
            if candidate_return_predictions.get(hs) is None: continue
            for bs in baseline_sources:
                if candidate_return_predictions.get(bs) is None: continue
                cand_name = f"Blend_{hs}_{bs}"
                for alpha in alphas:
                    if alpha == 0.0 or alpha == 1.0: continue
                    pred_ret = alpha * candidate_return_predictions[hs][:, h] + (1 - alpha) * candidate_return_predictions[bs][:, h]
                    pred_price = val_current_close * (1 + pred_ret)
                    c_price_mae = np.mean(np.abs(act_price - pred_price))
                    c_pred_up = (pred_ret > 0).astype(int)
                    c_actual_up = (act_ret > 0).astype(int)
                    c_dir_acc = np.mean(c_pred_up == c_actual_up)
                    
                    b_maes = []
                    for i in range(3):
                        start = i * block_size
                        end = (i + 1) * block_size if i < 2 else N
                        b_maes.append(np.mean(np.abs(act_price[start:end] - pred_price[start:end])))
                        
                    horizon_candidates.append({
                        'candidate_name': cand_name,
                        'hybrid_source': hs, 'baseline_source': bs, 'alpha': alpha,
                        'val_price_mae': c_price_mae,
                        'val_directional_accuracy': c_dir_acc,
                        'block_1_price_mae': b_maes[0], 'block_2_price_mae': b_maes[1], 'block_3_price_mae': b_maes[2]
                    })
                    
        simple_baselines = [c for c in horizon_candidates if c['candidate_name'] in ['Persistence', 'Drift', 'Momentum']]
        best_sb = min(simple_baselines, key=lambda x: x['val_price_mae'])
        best_sb_mae = best_sb['val_price_mae']
        best_sb_dir = best_sb['val_directional_accuracy']
        
        hybrid_cands = [c for c in horizon_candidates if c['candidate_name'] not in ['Persistence', 'Drift', 'Momentum']]
        if not hybrid_cands:
            continue
        best_hc = min(hybrid_cands, key=lambda x: x['val_price_mae'])
        
        imp_pct = (best_sb_mae - best_hc['val_price_mae']) / best_sb_mae * 100 if best_sb_mae > 0 else 0
        b1_imp = (best_sb['block_1_price_mae'] - best_hc['block_1_price_mae']) / best_sb['block_1_price_mae'] * 100 if best_sb['block_1_price_mae'] > 0 else 0
        b2_imp = (best_sb['block_2_price_mae'] - best_hc['block_2_price_mae']) / best_sb['block_2_price_mae'] * 100 if best_sb['block_2_price_mae'] > 0 else 0
        b3_imp = (best_sb['block_3_price_mae'] - best_hc['block_3_price_mae']) / best_sb['block_3_price_mae'] * 100 if best_sb['block_3_price_mae'] > 0 else 0
        
        num_blocks_beating = sum(1 for i in [b1_imp, b2_imp, b3_imp] if i > 0)
        worst_block = min([b1_imp, b2_imp, b3_imp])
        
        passes_mae = imp_pct >= min_improvement_pct
        passes_blocks = num_blocks_beating >= required_blocks
        passes_dir = best_hc['val_directional_accuracy'] >= best_sb_dir
        
        if passes_mae and passes_blocks and passes_dir:
            selected = best_hc
            fallback = False
            reason = "None"
        else:
            selected = best_sb
            fallback = True
            if not passes_mae:
                reason = f"MAE Imp < {min_improvement_pct}%"
            elif not passes_blocks:
                reason = f"Blocks Beating < {required_blocks}"
            else:
                reason = "Fails Direction Guard"
                
        selected_variants_results.append({
            'horizon': h + 1,
            'selected_candidate': selected['candidate_name'],
            'hybrid_source': selected['hybrid_source'],
            'baseline_source': selected['baseline_source'],
            'alpha': selected['alpha'],
            'selected_is_hybrid_based': not fallback,
            'best_simple_baseline_name': best_sb['candidate_name'],
            'best_simple_baseline_val_price_mae': best_sb_mae,
            'best_hybrid_candidate_name': best_hc['candidate_name'],
            'best_hybrid_val_price_mae': best_hc['val_price_mae'],
            'full_val_improvement_pct': imp_pct,
            'num_blocks_beating_baseline': num_blocks_beating,
            'worst_block_improvement_pct': worst_block,
            'val_directional_accuracy': selected['val_directional_accuracy'],
            'best_simple_baseline_val_directional_accuracy': best_sb_dir,
            'fallback_to_baseline': fallback,
            'rejected_reason': reason
        })
        
    pd.DataFrame(selected_variants_results).to_csv(os.path.join(ticker_dir, "final_safe_selection.csv"), index=False)
    return selected_variants_results


def run_single_ticker(args, ticker):
    print(f"\n{'=' * 60}")
    print(f"  STARTING DIAGNOSTICS FOR {ticker}")
    print(f"{'=' * 60}")
    
    ticker_dir = os.path.join(args.output_dir, ticker)
    os.makedirs(ticker_dir, exist_ok=True)
    
    ohlcv_features = ['Open', 'High', 'Low', 'Close', 'Volume']
    ma_targets = ['EMA_20', 'SMA_50']
    
    ds_kwargs = dict(
        ticker=ticker, root_path=args.data_root,
        seq_len=args.seq_len, pred_len=args.pred_len,
        input_features=ohlcv_features,
        ma_targets=ma_targets,
    )
    
    train_dataset = ParquetDataset(flag='train', **ds_kwargs)
    val_dataset   = ParquetDataset(flag='val',   **ds_kwargs)
    test_dataset  = ParquetDataset(flag='test',  **ds_kwargs)
    
    check_scalers(train_dataset, val_dataset, test_dataset, verbose=getattr(args, 'verbose', False))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False)
    
    input_features = train_dataset.input_features
    target_features = train_dataset.target_features
    close_idx_x = input_features.index("Close")
    close_idx_y = target_features.index("Close")
    
    y_train = get_multistep_continuous_labels(train_loader, close_idx_x, close_idx_y, train_dataset)
    y_val   = get_multistep_continuous_labels(val_loader,   close_idx_x, close_idx_y, val_dataset)
    y_test  = get_multistep_continuous_labels(test_loader,  close_idx_x, close_idx_y, test_dataset)
    
    test_current_close = get_current_close(test_loader, close_idx_x, test_dataset)
    check_label_alignment(test_loader, close_idx_x, close_idx_y, test_dataset, y_test, test_current_close, verbose=getattr(args, 'verbose', False))
    
    tm_config = TimeMixerConfig(
        historical_lookback_length=args.seq_len,
        forecast_horizon_length=args.pred_len,
        number_of_input_features=train_dataset.enc_in,
        number_of_output_features=train_dataset.c_out,
        model_embedding_dimension=32,
        number_of_pdm_blocks=1,
        dropout_probability=0.2,
    )
    
    hybrid_model = HybridTimeMixerLGBM(timemixer_config=tm_config)
    hybrid_model.fit(
        train_loader, y_train,
        val_loader=val_loader, y_val=y_val,
        tm_epochs=args.epochs,
        output_dir=ticker_dir,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        use_raw_stats=args.use_raw_stats,
        use_direction_classifier=args.use_direction_classifier,
        use_regression_variant_search=args.use_regression_variant_search,
        verbose=getattr(args, 'verbose', False)
    )
    
    if args.calibrate_predictions:
        print("\n>>> Calibrating Predictions on Validation Data...")
        val_preds_return = hybrid_model.predict(
            val_loader,
            output_dir=ticker_dir,
            y_test=y_val,
            test_dataset=val_dataset,
            use_raw_stats=args.use_raw_stats,
            split_name="val_pred",
            verbose=getattr(args, 'verbose', False)
        )
        calibration_params = calibrate_returns_on_validation(y_val, val_preds_return, ticker_dir)
        num_raw_horizons = sum(1 for p in calibration_params if p['method'] == 'RAW_NO_CALIBRATION')
        num_calibrated_horizons = len(calibration_params) - num_raw_horizons
        
        if getattr(args, 'verbose', False):
            print("\n  --- Calibration Method Breakdown ---")
            method_counts = {}
            for p in calibration_params:
                m = p['method']
                method_counts[m] = method_counts.get(m, 0) + 1
            for m, count in method_counts.items():
                print(f"  {m}: {count} horizons")
            
    else:
        calibration_params = None
        num_raw_horizons = args.pred_len
        num_calibrated_horizons = 0
        
    if args.use_direction_classifier:
        print("\n>>> Selecting Direction Thresholds on Validation Data...")
        val_direction_proba = hybrid_model.predict_direction_proba(
            val_loader, test_dataset=val_dataset, use_raw_stats=args.use_raw_stats, verbose=getattr(args, 'verbose', False)
        )
        direction_thresholds = select_direction_thresholds_on_validation(y_val, val_direction_proba, ticker_dir)
    else:
        direction_thresholds = None
        
    if args.use_final_blend_selection:
        print("\n>>> Selecting Final Blend on Validation Data...")
        val_current_close = get_current_close(val_loader, close_idx_x, val_dataset)
        val_past_5_day_returns = get_past_5_day_returns(val_loader, close_idx_x, val_dataset)
        val_pers_p, val_drift_p, val_mom_p, val_pers_r, val_drift_r, val_mom_r = build_baseline_predictions(
            val_current_close, y_train[:, 0], val_past_5_day_returns, args.pred_len
        )
        
        if not args.calibrate_predictions:
            val_preds_return = hybrid_model.predict(
                val_loader, output_dir=ticker_dir, y_test=y_val, test_dataset=val_dataset,
                use_raw_stats=args.use_raw_stats, split_name="val_pred", verbose=getattr(args, 'verbose', False)
            )
            val_preds_return_calibrated = None
        else:
            val_preds_return_calibrated = apply_calibration(val_preds_return, calibration_params)
            
        candidate_returns = {
            'Hybrid_Raw': val_preds_return,
            'Hybrid_Calibrated': val_preds_return_calibrated,
            'Persistence': val_pers_r,
            'Drift': val_drift_r,
            'Momentum': val_mom_r
        }
        
        final_blend_selection = select_final_blend_on_validation(
            y_val, val_current_close, candidate_returns, args.pred_len, ticker_dir, 
            args.final_blend_min_improvement_pct, args.final_blend_min_blocks_beating_baseline, args.final_blend_worst_block_tolerance_pct
        )
    else:
        final_blend_selection = None
        
    if getattr(args, 'use_final_safe_selection', False):
        if 'val_current_close' not in locals():
            val_current_close = get_current_close(val_loader, close_idx_x, val_dataset)
            val_past_5_day_returns = get_past_5_day_returns(val_loader, close_idx_x, val_dataset)
            val_pers_p, val_drift_p, val_mom_p, val_pers_r, val_drift_r, val_mom_r = build_baseline_predictions(
                val_current_close, y_train[:, 0], val_past_5_day_returns, args.pred_len
            )
            
            if not args.calibrate_predictions:
                val_preds_return = hybrid_model.predict(
                    val_loader, output_dir=ticker_dir, y_test=y_val, test_dataset=val_dataset,
                    use_raw_stats=args.use_raw_stats, split_name="val_pred"
                )
                val_preds_return_calibrated = None
            else:
                val_preds_return_calibrated = apply_calibration(val_preds_return, calibration_params)
                
            candidate_returns = {
                'Hybrid_Raw': val_preds_return,
                'Hybrid_Calibrated': val_preds_return_calibrated,
                'Persistence': val_pers_r,
                'Drift': val_drift_r,
                'Momentum': val_mom_r
            }
            
        final_safe_selection = select_final_safe_on_validation(
            y_val, val_current_close, candidate_returns, args.pred_len, ticker_dir,
            args.final_safe_min_improvement_pct, args.final_safe_required_blocks_beating_baseline
        )
    else:
        final_safe_selection = None
    
    print("\n>>> Predicting on Test Data...")
    preds_return_raw = hybrid_model.predict(
        test_loader, 
        output_dir=ticker_dir, 
        y_test=y_test,
        test_dataset=test_dataset,
        use_raw_stats=args.use_raw_stats,
        verbose=getattr(args, 'verbose', False)
    )
    
    if args.calibrate_predictions and calibration_params:
        preds_return_calibrated = apply_calibration(preds_return_raw, calibration_params)
    else:
        preds_return_calibrated = preds_return_raw

    pred_prices_raw = test_current_close[:, None] * (1 + preds_return_raw)
    pred_prices_calibrated = test_current_close[:, None] * (1 + preds_return_calibrated)
    
    if args.use_direction_classifier and direction_thresholds:
        test_direction_proba = hybrid_model.predict_direction_proba(
            test_loader, test_dataset=test_dataset, use_raw_stats=args.use_raw_stats, verbose=getattr(args, 'verbose', False)
        )
        preds_return_dir_gated = np.zeros_like(preds_return_raw)
        preds_return_dir_gated_cons = np.zeros_like(preds_return_raw)
        
        for i, dt in enumerate(direction_thresholds):
            th = dt['threshold']
            preds_up = (test_direction_proba[:, i] >= th).astype(int)
            direction_sign = np.where(preds_up == 1, 1.0, -1.0)
            
            preds_return_dir_gated[:, i] = direction_sign * np.abs(preds_return_raw[:, i])
            preds_return_dir_gated_cons[:, i] = direction_sign * np.abs(preds_return_raw[:, i]) * 0.5
            
        pred_prices_dir_gated = test_current_close[:, None] * (1 + preds_return_dir_gated)
        pred_prices_dir_gated_cons = test_current_close[:, None] * (1 + preds_return_dir_gated_cons)
    else:
        preds_return_dir_gated = preds_return_raw
        preds_return_dir_gated_cons = preds_return_raw
        pred_prices_dir_gated = pred_prices_raw
        pred_prices_dir_gated_cons = pred_prices_raw
        test_direction_proba = None
        
    actual_prices = test_current_close[:, None] * (1 + y_test)
    
    # Baselines
    past_5_day_returns = get_past_5_day_returns(test_loader, close_idx_x, test_dataset)
    pers_p, drift_p, mom_p, pers_r, drift_r, mom_r = build_baseline_predictions(
        test_current_close, y_train[:, 0], past_5_day_returns, args.pred_len
    )
    
    if args.use_final_blend_selection and final_blend_selection:
        test_candidate_returns = {
            'Hybrid_Raw': preds_return_raw,
            'Hybrid_Calibrated': preds_return_calibrated if args.calibrate_predictions else None,
            'Persistence': pers_r,
            'Drift': drift_r,
            'Momentum': mom_r
        }
        
        preds_return_final_selected = np.zeros_like(preds_return_raw)
        for cand in final_blend_selection:
            h_idx = cand['horizon'] - 1
            alpha = cand['alpha']
            hs = cand['hybrid_source']
            bs = cand['baseline_source']
            
            if hs != 'None' and bs != 'None':
                pred_h = alpha * test_candidate_returns[hs][:, h_idx] + (1 - alpha) * test_candidate_returns[bs][:, h_idx]
            elif hs != 'None':
                pred_h = test_candidate_returns[hs][:, h_idx]
            else:
                pred_h = test_candidate_returns[bs][:, h_idx]
                
            preds_return_final_selected[:, h_idx] = pred_h
            
        pred_prices_final_selected = test_current_close[:, None] * (1 + preds_return_final_selected)
    else:
        preds_return_final_selected = preds_return_calibrated if args.calibrate_predictions else preds_return_raw
        pred_prices_final_selected = pred_prices_calibrated if args.calibrate_predictions else pred_prices_raw

    if getattr(args, 'use_final_safe_selection', False) and final_safe_selection:
        if 'test_candidate_returns' not in locals():
            test_candidate_returns = {
                'Hybrid_Raw': preds_return_raw,
                'Hybrid_Calibrated': preds_return_calibrated if args.calibrate_predictions else None,
                'Persistence': pers_r,
                'Drift': drift_r,
                'Momentum': mom_r
            }
            
        preds_return_final_safe = np.zeros_like(preds_return_raw)
        for cand in final_safe_selection:
            h_idx = cand['horizon'] - 1
            alpha = cand['alpha']
            hs = cand['hybrid_source']
            bs = cand['baseline_source']
            
            if hs != 'None' and bs != 'None':
                pred_h = alpha * test_candidate_returns[hs][:, h_idx] + (1 - alpha) * test_candidate_returns[bs][:, h_idx]
            elif hs != 'None':
                pred_h = test_candidate_returns[hs][:, h_idx]
            else:
                pred_h = test_candidate_returns[bs][:, h_idx]
                
            preds_return_final_safe[:, h_idx] = pred_h
            
        pred_prices_final_safe = test_current_close[:, None] * (1 + preds_return_final_safe)
    else:
        preds_return_final_safe = preds_return_final_selected
        pred_prices_final_safe = pred_prices_final_selected
    
    # Evaluate
    all_metrics = {}
    all_metrics.update(evaluate_price_forecasts(actual_prices, pred_prices_raw, y_test, preds_return_raw, args.pred_len, "Hybrid_Raw"))
    if args.calibrate_predictions:
        all_metrics.update(evaluate_price_forecasts(actual_prices, pred_prices_calibrated, y_test, preds_return_calibrated, args.pred_len, "Hybrid_Calibrated"))
    else:
        all_metrics.update(evaluate_price_forecasts(actual_prices, pred_prices_raw, y_test, preds_return_raw, args.pred_len, "Hybrid_Calibrated")) # Fallback if disabled

    if args.use_direction_classifier:
        all_metrics.update(evaluate_price_forecasts(actual_prices, pred_prices_dir_gated, y_test, preds_return_dir_gated, args.pred_len, "Hybrid_DirectionGated"))
        all_metrics.update(evaluate_price_forecasts(actual_prices, pred_prices_dir_gated_cons, y_test, preds_return_dir_gated_cons, args.pred_len, "Hybrid_DirectionGated_Conservative"))
        
        dir_results = []
        for i, dt in enumerate(direction_thresholds):
            th = dt['threshold']
            preds = (test_direction_proba[:, i] >= th).astype(int)
            act = (y_test[:, i] > 0).astype(int)
            dir_results.append({
                'horizon': i + 1,
                'threshold': th,
                'test_balanced_accuracy': balanced_accuracy_score(act, preds) if len(np.unique(act)) > 1 else 0.0,
                'test_accuracy': accuracy_score(act, preds),
                'test_pred_up_rate': np.mean(preds),
                'test_actual_up_rate': np.mean(act)
            })
        pd.DataFrame(dir_results).to_csv(os.path.join(ticker_dir, 'direction_classifier_metrics.csv'), index=False)
        all_metrics['DirectionClassifier_Final_BalancedAcc'] = dir_results[-1]['test_balanced_accuracy']
    else:
        all_metrics.update(evaluate_price_forecasts(actual_prices, pred_prices_raw, y_test, preds_return_raw, args.pred_len, "Hybrid_DirectionGated"))
        all_metrics.update(evaluate_price_forecasts(actual_prices, pred_prices_raw, y_test, preds_return_raw, args.pred_len, "Hybrid_DirectionGated_Conservative"))
        all_metrics['DirectionClassifier_Final_BalancedAcc'] = 0.0

    all_metrics.update(evaluate_price_forecasts(actual_prices, pers_p, y_test, pers_r, args.pred_len, "Persistence"))
    all_metrics.update(evaluate_price_forecasts(actual_prices, drift_p, y_test, drift_r, args.pred_len, "Drift"))
    all_metrics.update(evaluate_price_forecasts(actual_prices, mom_p, y_test, mom_r, args.pred_len, "Momentum"))
    
    if args.use_final_blend_selection:
        all_metrics.update(evaluate_price_forecasts(actual_prices, pred_prices_final_selected, y_test, preds_return_final_selected, args.pred_len, "Final_Selected"))
    else:
        all_metrics.update(evaluate_price_forecasts(actual_prices, pred_prices_final_selected, y_test, preds_return_final_selected, args.pred_len, "Final_Selected"))
        
    if getattr(args, 'use_final_safe_selection', False):
        all_metrics.update(evaluate_price_forecasts(actual_prices, pred_prices_final_safe, y_test, preds_return_final_safe, args.pred_len, "Final_Safe"))
    else:
        all_metrics.update(evaluate_price_forecasts(actual_prices, pred_prices_final_safe, y_test, preds_return_final_safe, args.pred_len, "Final_Safe"))
    
    save_diagnostics(all_metrics, ticker_dir, ticker)
    
    plot_diagnostics(ticker, ticker_dir, actual_prices, pred_prices_raw, pred_prices_calibrated, pers_p, y_test, preds_return_raw, preds_return_calibrated, args.pred_len, hybrid_prices_dir_gated=pred_prices_dir_gated, pred_prices_final_selected=pred_prices_final_selected)
    
    # Save sample predictions
    sample_df = pd.DataFrame({
        'Today_Price': test_current_close[:50],
        'Actual_Final_Price': actual_prices[:50, -1],
        'Hybrid_Raw_Final': pred_prices_raw[:50, -1],
        'Hybrid_Calibrated_Final': pred_prices_calibrated[:50, -1],
        'Final_Selected_Final': pred_prices_final_selected[:50, -1],
        'Final_Safe_Final': pred_prices_final_safe[:50, -1],
        'Persistence_Final': pers_p[:50, -1],
        'Drift_Final': drift_p[:50, -1]
    })
    sample_df.to_csv(os.path.join(ticker_dir, 'sample_predictions_final_selected.csv'), index=False)
    
    if getattr(args, 'use_final_safe_selection', False):
        sample_df.to_csv(os.path.join(ticker_dir, 'sample_predictions_final_safe.csv'), index=False)
    
    summary_dict = {
        'ticker': ticker,
        'hybrid_raw_MAE': all_metrics.get('Hybrid_Raw_MAE', 0),
        'hybrid_calibrated_MAE': all_metrics.get('Hybrid_Calibrated_MAE', 0),
        'final_selected_MAE': all_metrics.get('Final_Selected_MAE', 0),
        'persistence_MAE': all_metrics['Persistence_MAE'],
        'drift_MAE': all_metrics['Drift_MAE'],
        'momentum_MAE': all_metrics['Momentum_MAE'],
        'final_selected_vs_persistence_%': (all_metrics.get('Final_Selected_MAE', 0) - all_metrics['Persistence_MAE']) / all_metrics['Persistence_MAE'] * 100 if all_metrics['Persistence_MAE'] > 0 else 0,
        'raw_pred_up_rate': all_metrics.get('Hybrid_Raw_Final_PredUpRate', 0),
        'final_selected_pred_up_rate': all_metrics.get('Final_Selected_Final_PredUpRate', 0),
        'actual_up_rate': all_metrics.get('Hybrid_Raw_Final_ActUpRate', 0),
        'raw_final_day_dir_acc': all_metrics.get(f'Hybrid_Raw_DirAcc_day_{args.pred_len}', 0),
        'final_selected_final_day_dir_acc': all_metrics.get(f'Final_Selected_DirAcc_day_{args.pred_len}', 0),
        'final_selected_candidate_day5': final_blend_selection[-1]['candidate_name'] if final_blend_selection else "N/A",
        'final_selected_alpha_day5': final_blend_selection[-1]['alpha'] if final_blend_selection else 0.0,
        'final_selected_fallback_day5': final_blend_selection[-1].get('fallback_to_baseline', False) if final_blend_selection else False,
        'final_selected_day5_num_blocks_beating_baseline': final_blend_selection[-1].get('num_blocks_beating_baseline', 0) if final_blend_selection else 0,
        'final_selected_day5_worst_block_improvement_pct': final_blend_selection[-1].get('worst_block_improvement_pct', 0.0) if final_blend_selection else 0.0,
        'final_selected_vs_drift_%': (all_metrics.get('Final_Selected_MAE', 0) - all_metrics['Drift_MAE']) / all_metrics['Drift_MAE'] * 100 if all_metrics['Drift_MAE'] > 0 else 0,
        'final_safe_MAE': all_metrics.get('Final_Safe_MAE', 0),
        'final_safe_vs_persistence_%': (all_metrics.get('Final_Safe_MAE', 0) - all_metrics['Persistence_MAE']) / all_metrics['Persistence_MAE'] * 100 if all_metrics['Persistence_MAE'] > 0 else 0,
        'final_safe_vs_drift_%': (all_metrics.get('Final_Safe_MAE', 0) - all_metrics['Drift_MAE']) / all_metrics['Drift_MAE'] * 100 if all_metrics['Drift_MAE'] > 0 else 0,
        'final_safe_candidate_day5': final_safe_selection[-1]['selected_candidate'] if (getattr(args, 'use_final_safe_selection', False) and final_safe_selection) else "N/A",
        'final_safe_alpha_day5': final_safe_selection[-1]['alpha'] if (getattr(args, 'use_final_safe_selection', False) and final_safe_selection) else 0.0,
        'final_safe_fallback_day5': final_safe_selection[-1].get('fallback_to_baseline', False) if (getattr(args, 'use_final_safe_selection', False) and final_safe_selection) else False,
        'final_safe_day5_num_blocks_beating_baseline': final_safe_selection[-1].get('num_blocks_beating_baseline', 0) if (getattr(args, 'use_final_safe_selection', False) and final_safe_selection) else 0,
        'final_safe_day5_worst_block_improvement_pct': final_safe_selection[-1].get('worst_block_improvement_pct', 0.0) if (getattr(args, 'use_final_safe_selection', False) and final_safe_selection) else 0.0,
        'direction_classifier_final_day_balanced_acc': all_metrics.get('DirectionClassifier_Final_BalancedAcc', 0),
        'best_variant_by_val_if_available': "N/A"
    }
    
    prediction_arrays = {
        'actual_returns': y_test,
        'Hybrid_Raw': preds_return_raw,
        'Final_Selected': preds_return_final_selected,
        'Final_Safe': preds_return_final_safe,
        'Persistence': pers_r,
        'Drift': drift_r
    }
    
    return summary_dict, prediction_arrays

def run_all_tickers(args):
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    summaries = []
    all_prediction_arrays = {}
    for t in args.tickers:
        summary, pred_arrays = run_single_ticker(args, t)
        summaries.append(summary)
        all_prediction_arrays[t] = pred_arrays
        
    df_summary = pd.DataFrame(summaries)
    df_summary.to_csv(os.path.join(args.output_dir, 'all_tickers_summary.csv'), index=False)
    
    print(f"\n{'=' * 80}")
    print("  FINAL SUMMARY")
    print(f"{'=' * 80}")
    
    compact_cols = [
        'ticker', 'hybrid_raw_MAE', 'final_selected_MAE', 'final_safe_MAE',
        'persistence_MAE', 'drift_MAE', 'final_safe_vs_persistence_%',
        'final_safe_vs_drift_%', 'final_safe_candidate_day5', 'final_safe_fallback_day5'
    ]
    
    df_compact = df_summary[compact_cols].copy()
    
    formatters = {}
    for col in df_compact.columns:
        if 'MAE' in col:
            formatters[col] = lambda x: f"{x:.4f}"
        elif '%' in col:
            formatters[col] = lambda x: f"{x:.2f}"
            
    print(df_compact.to_string(index=False, formatters=formatters))
    
    # ---------------------------------------------------------
    # Cross-Sectional IC / RIC Evaluation
    # ---------------------------------------------------------
    if len(args.tickers) >= 3:
        common_test_len = min(len(arrays['actual_returns']) for arrays in all_prediction_arrays.values())
        if getattr(args, 'verbose', False):
            print(f"\n[IC/RIC] Aligning test predictions to minimum common test length: {common_test_len} samples.")
            
        ic_ric_results = []
        ic_ric_summary_results = []
        pred_families = ['Hybrid_Raw', 'Final_Selected', 'Final_Safe', 'Persistence', 'Drift']
        
        for family in pred_families:
            family_ic_all = []
            family_ric_all = []
            for h in range(args.pred_len):
                horizon_ics = []
                horizon_rics = []
                for t in range(common_test_len):
                    actual_xs = []
                    pred_xs = []
                    for ticker in args.tickers:
                        actual_xs.append(all_prediction_arrays[ticker]['actual_returns'][t, h])
                        pred_xs.append(all_prediction_arrays[ticker][family][t, h])
                    
                    actual_xs = np.array(actual_xs)
                    pred_xs = np.array(pred_xs)
                    
                    if np.var(actual_xs) > 1e-8 and np.var(pred_xs) > 1e-8:
                        ic, _ = pearsonr(pred_xs, actual_xs)
                        ric, _ = spearmanr(pred_xs, actual_xs)
                        if not np.isnan(ic): horizon_ics.append(ic)
                        if not np.isnan(ric): horizon_rics.append(ric)
                
                mean_ic = np.mean(horizon_ics) if horizon_ics else 0.0
                std_ic = np.std(horizon_ics) if horizon_ics else 0.0
                mean_ric = np.mean(horizon_rics) if horizon_rics else 0.0
                std_ric = np.std(horizon_rics) if horizon_rics else 0.0
                
                ic_ric_results.append({
                    'prediction_family': family,
                    'horizon': h + 1,
                    'mean_IC': mean_ic,
                    'std_IC': std_ic,
                    'mean_RIC': mean_ric,
                    'std_RIC': std_ric,
                    'valid_cross_sections': len(horizon_ics)
                })
                family_ic_all.extend(horizon_ics)
                family_ric_all.extend(horizon_rics)
                
            ic_ric_summary_results.append({
                'prediction_family': family,
                'mean_IC_all_horizons': np.mean(family_ic_all) if family_ic_all else 0.0,
                'mean_RIC_all_horizons': np.mean(family_ric_all) if family_ric_all else 0.0
            })
            
        df_ic_ric = pd.DataFrame(ic_ric_results)
        df_ic_ric.to_csv(os.path.join(args.output_dir, 'cross_sectional_ic_ric.csv'), index=False)
        
        df_ic_ric_summary = pd.DataFrame(ic_ric_summary_results)
        df_ic_ric_summary.to_csv(os.path.join(args.output_dir, 'cross_sectional_ic_ric_summary.csv'), index=False)
        
        print(f"\n{'=' * 80}")
        print("  CROSS-SECTIONAL IC/RIC SUMMARY")
        print(f"{'=' * 80}")
        df_ic_ric_compact = df_ic_ric_summary[['prediction_family', 'mean_IC_all_horizons', 'mean_RIC_all_horizons']].copy()
        df_ic_ric_compact.rename(columns={'mean_IC_all_horizons': 'mean_IC', 'mean_RIC_all_horizons': 'mean_RIC'}, inplace=True)
        ic_formatters = {'mean_IC': lambda x: f"{x:.4f}", 'mean_RIC': lambda x: f"{x:.4f}"}
        print(df_ic_ric_compact.to_string(index=False, formatters=ic_formatters))
    
    print(f"\nSaved full summary to: {os.path.join(args.output_dir, 'all_tickers_summary.csv')}")
    print(f"Saved compact final report outputs to: {args.output_dir}")
    if not args.verbose:
        print("Note: Run with --verbose for full diagnostic output during training.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Final hybrid stock forecasting runner. By default runs the final report configuration for all five tickers.")
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "GOOGL", "META", "MSFT", "NVDA"], help="List of tickers to run")
    parser.add_argument("--data_root", type=str, default="data/raw", help="Root directory for parquet/csv data")
    parser.add_argument("--seq_len", type=int, default=30, help="Input sequence length")
    parser.add_argument("--pred_len", type=int, default=5, help="Prediction horizon length")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="hybrid_diagnostics_final_report", help="Output directory for diagnostics")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False, help="Enable detailed diagnostic console outputs")
    parser.add_argument("--use_raw_stats", action=argparse.BooleanOptionalAction, default=True, help="Use raw statistical features instead of normalized")
    parser.add_argument("--calibrate_predictions", action=argparse.BooleanOptionalAction, default=True, help="Calibrate test predictions using validation predictions")
    parser.add_argument("--use_direction_classifier", action=argparse.BooleanOptionalAction, default=True, help="Train a directional classifier per horizon")
    parser.add_argument("--use_regression_variant_search", action=argparse.BooleanOptionalAction, default=True, help="Use validation-based regression variant search")
    parser.add_argument("--use_final_blend_selection", action=argparse.BooleanOptionalAction, default=True, help="Use validation-based final blend selection")
    parser.add_argument("--final_blend_min_improvement_pct", type=float, default=1.0, help="Minimum validation MAE improvement percentage for hybrid blend to be selected")
    parser.add_argument("--final_blend_min_blocks_beating_baseline", type=int, default=2, help="Minimum number of validation blocks where hybrid beats baseline")
    parser.add_argument("--final_blend_worst_block_tolerance_pct", type=float, default=-2.0, help="Maximum allowed MAE percentage loss in any single block")
    parser.add_argument("--use_final_safe_selection", action=argparse.BooleanOptionalAction, default=True, help="Use extremely conservative validation-based final safe selection")
    parser.add_argument("--final_safe_min_improvement_pct", type=float, default=3.0, help="Minimum validation MAE improvement percentage for safe selection")
    parser.add_argument("--final_safe_required_blocks_beating_baseline", type=int, default=3, help="Required number of validation blocks where hybrid beats baseline for safe selection")
    
    args = parser.parse_args()
    run_all_tickers(args)
