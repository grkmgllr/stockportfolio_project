"""
Testing/Evaluation script for TimeMixer/TimeMixer++ models.
Usage: python test.py
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os

from configs.config import get_config
from dataset import ETTh1Dataset
from utils import (
    get_model, 
    load_checkpoint, 
    calculate_metrics,
    print_config
)


def evaluate(model, test_loader, criterion, config):
    """
    Evaluate model on test set.
    
    Returns:
        metrics: Dictionary with MSE, MAE, RMSE
        all_preds: All predictions
        all_trues: All ground truths
    """
    model.eval()
    
    all_preds = []
    all_trues = []
    test_loss = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(config.device)
            batch_y = batch_y.to(config.device)
            
            # Forward pass
            outputs = model(batch_x, None)
            
            # Get predictions
            pred = outputs[:, -config.model.pred_len:, :]
            true = batch_y[:, -config.model.pred_len:, :]
            
            loss = criterion(pred, true)
            test_loss.append(loss.item())
            
            # Store predictions
            all_preds.append(pred.cpu().numpy())
            all_trues.append(true.cpu().numpy())
    
    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0)
    all_trues = np.concatenate(all_trues, axis=0)
    
    # Calculate metrics
    metrics = calculate_metrics(all_preds, all_trues)
    
    return metrics, all_preds, all_trues


def main():
    # Load configuration
    config = get_config()
    print_config(config)
    
    # Data loading
    print("Loading Test Data...")
    size = [config.model.seq_len, 0, config.model.pred_len]
    
    test_dataset = ETTh1Dataset(
        root_path=config.data.root_path, 
        flag='test', 
        size=size, 
        features=config.data.features
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.training.batch_size, 
        shuffle=False, 
        drop_last=False
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Model initialization
    model = get_model(config.model_name, config.get_model_params(), config.device)
    
    # Load checkpoint
    checkpoint_path = os.path.join(
        config.checkpoint_dir, 
        f"{config.model_name.replace('+', 'p')}_best.pt"
    )
    
    if not os.path.exists(checkpoint_path):
        print(f"\nError: Checkpoint not found at {checkpoint_path}")
        print("Please run train.py first to train the model.")
        return
    
    load_checkpoint(model, checkpoint_path, config.device)
    
    # Evaluation
    print("\nEvaluating on Test Set...")
    print("-" * 50)
    
    criterion = nn.MSELoss()
    metrics, all_preds, all_trues = evaluate(model, test_loader, criterion, config)
    
    # Print results
    print("\nTest Results:")
    print("=" * 50)
    print(f"  MSE:  {metrics['MSE']:.6f}")
    print(f"  MAE:  {metrics['MAE']:.6f}")
    print(f"  RMSE: {metrics['RMSE']:.6f}")
    print("=" * 50)
    
    # Optional: Save predictions
    save_predictions = False  # Set to True to save
    if save_predictions:
        output_dir = "results"
        os.makedirs(output_dir, exist_ok=True)
        
        np.save(os.path.join(output_dir, "predictions.npy"), all_preds)
        np.save(os.path.join(output_dir, "ground_truth.npy"), all_trues)
        print(f"\nPredictions saved to {output_dir}/")
    
    return metrics


if __name__ == "__main__":
    main()
