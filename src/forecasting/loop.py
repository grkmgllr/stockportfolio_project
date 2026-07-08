"""Train / val / test loops for PyTorch forecasters.

Pure computation — no printing, no file IO.  Reporting lives in reporting.py.
"""

import numpy as np
import torch

from utils import calculate_metrics, calculate_return_metrics


def train_epoch(model, train_loader, criterion, optimizer, device, grad_clip):
    """Run one training epoch. Returns mean loss."""
    model.train()
    losses = []

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        # second arg is time-features (unused by our models)
        outputs = model(batch_x, None)

        loss = criterion(outputs, batch_y)
        losses.append(loss.item())

        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

    return float(np.average(losses))


def validate_epoch(model, val_loader, criterion, device):
    """Run one validation pass (no grad). Returns mean val loss."""
    model.eval()
    losses = []

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_x, None)
            loss = criterion(outputs, batch_y)
            losses.append(loss.item())

    return float(np.average(losses))


def evaluate(model, test_loader, criterion, device, dataset):
    """
    Run test-set inference and compute metrics.

    Returns (results_dict, preds, trues) where preds/trues are in the
    original dollar-price scale with shape [N, pred_len, n_targets].
    """
    model.eval()
    all_preds = []
    all_trues = []
    test_loss = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_x, None)
            loss = criterion(outputs, batch_y)
            test_loss.append(loss.item())

            all_preds.append(outputs.cpu().numpy())
            all_trues.append(batch_y.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_trues = np.concatenate(all_trues, axis=0)

    n_samples, pred_len, n_features = all_preds.shape

    if dataset.return_targets:
        # convert % returns back to absolute prices: price = anchor * (1 + return)
        anchors = dataset.get_anchors()
        a = anchors[:, None, None]
        preds_original = a * (1.0 + all_preds)
        trues_original = a * (1.0 + all_trues)
    else:
        preds_flat = all_preds.reshape(-1, n_features)
        trues_flat = all_trues.reshape(-1, n_features)
        preds_original = dataset.inverse_transform_y(preds_flat).reshape(
            n_samples, pred_len, n_features)
        trues_original = dataset.inverse_transform_y(trues_flat).reshape(
            n_samples, pred_len, n_features)

    results = {
        "overall": calculate_metrics(preds_original, trues_original),
        "test_loss": float(np.average(test_loss)),
        "overall_returns": calculate_return_metrics(all_preds, all_trues),
    }
    for i, name in enumerate(dataset.target_features):
        results[name] = calculate_metrics(
            preds_original[:, :, i], trues_original[:, :, i],
        )
        results[f"{name}_returns"] = calculate_return_metrics(
            all_preds[:, :, i], all_trues[:, :, i],
        )

    return results, preds_original, trues_original
