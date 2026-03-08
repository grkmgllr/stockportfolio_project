"""
Academic evaluation metrics for the meta-labeling pipeline.

Provides functions to measure the quality of the primary model's signals
both *before* and *after* the LightGBM meta-classifier filters them.
This allows a rigorous, side-by-side comparison that demonstrates whether
the secondary model adds value.

Metrics implemented
-------------------
Classification metrics (per scikit-learn conventions):
    Precision  -- Of all trades the model says "take", what fraction
                  were actually profitable?
    Recall     -- Of all trades that *were* profitable, what fraction
                  did the model correctly identify?
    F1-Score   -- Harmonic mean of Precision and Recall.

Risk-adjusted metric (per Lopez de Prado):
    Probabilistic Sharpe Ratio (PSR)
               -- The probability that the observed Sharpe Ratio exceeds
                  a benchmark, accounting for the skewness and kurtosis
                  of the return distribution.

Design goals
------------
- Produce publication-ready evaluation tables that can go directly into
  a thesis or research paper.
- Keep all metric calculations in pure NumPy / SciPy so there is no
  dependency on PyTorch for evaluation.
- Accept both the raw meta-label DataFrame and explicit arrays, making
  the functions usable from scripts or notebooks.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Optional


def _precision_recall_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute binary Precision, Recall, and F1-Score.

    This is a self-contained implementation that avoids importing
    scikit-learn at evaluation time (keeping the trading_logic package
    lightweight).

    Args:
        y_true (np.ndarray):
            Ground-truth binary labels of shape ``[N]``.
        y_pred (np.ndarray):
            Predicted binary labels of shape ``[N]``.

    Returns:
        Dict[str, float]:
            Dictionary with keys ``'precision'``, ``'recall'``,
            ``'f1_score'``.  Values are 0.0 when the denominator
            is zero (e.g. no positive predictions).
    """
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": int(np.sum((y_pred == 0) & (y_true == 0))),
    }


def evaluate_baseline(
    y_true: np.ndarray,
) -> Dict[str, float]:
    """
    Evaluate the primary model's signals WITHOUT any filtering.

    In the baseline scenario, the primary model generates a "buy"
    signal on every bar and all signals are acted upon.  This is
    equivalent to setting ``y_pred = np.ones_like(y_true)`` -- the model
    predicts positive (profitable) for every observation.

    The resulting Precision equals the *base rate* of profitable trades
    in the dataset, which is the number the meta-classifier must beat.

    Args:
        y_true (np.ndarray):
            Ground-truth binary labels from the Triple Barrier Method.
            Shape ``[N]``, values in {0, 1}.

    Returns:
        Dict[str, float]:
            Classification metrics for the unfiltered baseline.
            Keys: ``'precision'``, ``'recall'``, ``'f1_score'``,
            ``'tp'``, ``'fp'``, ``'fn'``, ``'tn'``,
            ``'n_trades'``, ``'base_rate'``.

    Notes
    -----
    - Recall is always 1.0 in the baseline because all positive events
      are "captured" (no signal is skipped).
    - Precision equals ``y_true.mean()`` because every observation is
      predicted as positive.
    """
    y_pred = np.ones_like(y_true)
    metrics = _precision_recall_f1(y_true, y_pred)
    metrics["n_trades"] = int(len(y_true))
    metrics["base_rate"] = float(y_true.mean())
    return metrics


def evaluate_filtered(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Evaluate the meta-classifier's filtered signals.

    After the LightGBM model filters the primary model's signals, only
    a subset of trades are taken.  This function computes classification
    metrics on that filtered subset and compares against the baseline.

    Args:
        y_true (np.ndarray):
            Ground-truth binary labels from the Triple Barrier Method.
            Shape ``[N]``, values in {0, 1}.
        y_pred (np.ndarray):
            Binary predictions from the meta-classifier.  Shape ``[N]``,
            values in {0, 1}.  A value of 1 means "take this trade".
        y_proba (np.ndarray, optional):
            Probability scores from ``predict_proba()``.  Shape ``[N]``,
            values in [0, 1].  If provided, the mean confidence of
            accepted and rejected trades is included in the output.

    Returns:
        Dict[str, float]:
            Classification metrics for the filtered predictions.
            Keys: ``'precision'``, ``'recall'``, ``'f1_score'``,
            ``'tp'``, ``'fp'``, ``'fn'``, ``'tn'``,
            ``'n_trades_taken'``, ``'n_trades_skipped'``,
            ``'filter_rate'``.
            If ``y_proba`` is provided, also includes
            ``'mean_confidence_taken'`` and ``'mean_confidence_skipped'``.
    """
    metrics = _precision_recall_f1(y_true, y_pred)

    n_taken = int(y_pred.sum())
    n_skipped = int(len(y_pred) - n_taken)

    metrics["n_trades_taken"] = n_taken
    metrics["n_trades_skipped"] = n_skipped
    metrics["filter_rate"] = n_skipped / len(y_pred) if len(y_pred) > 0 else 0.0

    if y_proba is not None:
        taken_mask = y_pred == 1
        skipped_mask = y_pred == 0
        metrics["mean_confidence_taken"] = (
            float(y_proba[taken_mask].mean()) if taken_mask.any() else 0.0
        )
        metrics["mean_confidence_skipped"] = (
            float(y_proba[skipped_mask].mean()) if skipped_mask.any() else 0.0
        )

    return metrics


def probabilistic_sharpe_ratio(
    returns: np.ndarray,
    sr_benchmark: float = 0.0,
    annualization_factor: float = 252.0,
) -> Dict[str, float]:
    """
    Compute the Probabilistic Sharpe Ratio (PSR).

    The PSR, introduced by Lopez de Prado (2014), answers the question:
    "What is the probability that the observed Sharpe Ratio is greater
    than a given benchmark, after adjusting for the higher moments
    (skewness and kurtosis) of the return distribution?"

    A high PSR (e.g. > 0.95) indicates that the strategy's performance
    is unlikely to be due to chance alone.

    Formula
    -------
    .. math::

        PSR = \\Phi\\left(
            \\frac{(\\hat{SR} - SR^{*}) \\sqrt{T-1}}
                 {\\sqrt{1 - \\hat{\\gamma}_3 \\hat{SR}
                        + \\frac{\\hat{\\gamma}_4 - 1}{4} \\hat{SR}^2}}
        \\right)

    where:
        :math:`\\hat{SR}` = observed (sample) Sharpe Ratio
        :math:`SR^{*}`    = benchmark Sharpe Ratio
        :math:`T`         = number of return observations
        :math:`\\hat{\\gamma}_3` = skewness of returns
        :math:`\\hat{\\gamma}_4` = kurtosis of returns (excess)
        :math:`\\Phi`     = CDF of the standard normal distribution

    Args:
        returns (np.ndarray):
            Array of per-trade or per-period returns.  Shape ``[T]``.
        sr_benchmark (float):
            Benchmark Sharpe Ratio to test against (default 0.0, i.e.
            "is the strategy better than flat?").
        annualization_factor (float):
            Factor for annualising the Sharpe Ratio.  Use 252 for daily
            returns, 52 for weekly, 12 for monthly (default 252).

    Returns:
        Dict[str, float]:
            Dictionary with keys:
            ``'sharpe_ratio'``       -- annualised sample SR
            ``'psr'``                -- Probabilistic Sharpe Ratio [0, 1]
            ``'skewness'``           -- sample skewness of returns
            ``'kurtosis'``           -- sample excess kurtosis of returns
            ``'n_observations'``     -- number of return observations
            ``'sr_benchmark'``       -- the benchmark used

    Notes
    -----
    - If there are fewer than 3 observations or zero standard deviation,
      PSR is returned as NaN because the higher-moment estimates are
      unreliable.
    - The SR is computed in *per-period* terms for the PSR formula, then
      annualised for reporting.  The PSR itself is scale-invariant.

    References
    ----------
    Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio: Correcting
    for Selection Bias, Backtest Overfitting, and Non-Normality."
    *Journal of Portfolio Management*, 40(5), 94-107.
    """
    returns = np.asarray(returns, dtype=np.float64)
    T = len(returns)

    result = {
        "sharpe_ratio": np.nan,
        "psr": np.nan,
        "skewness": np.nan,
        "kurtosis": np.nan,
        "n_observations": T,
        "sr_benchmark": sr_benchmark,
    }

    if T < 3:
        return result

    mu = returns.mean()
    sigma = returns.std(ddof=1)

    if sigma < 1e-15:
        return result

    sr_per_period = mu / sigma
    sr_annualised = sr_per_period * np.sqrt(annualization_factor)

    gamma3 = float(stats.skew(returns, bias=False))
    gamma4 = float(stats.kurtosis(returns, bias=False))

    # PSR formula (using per-period SR, not annualised)
    denom_sq = 1.0 - gamma3 * sr_per_period + (gamma4 / 4.0) * sr_per_period ** 2

    if denom_sq <= 0:
        return result

    z = (sr_per_period - sr_benchmark) * np.sqrt(T - 1) / np.sqrt(denom_sq)
    psr = float(stats.norm.cdf(z))

    result["sharpe_ratio"] = float(sr_annualised)
    result["psr"] = psr
    result["skewness"] = gamma3
    result["kurtosis"] = gamma4

    return result


def compute_trade_returns(
    df: pd.DataFrame,
    close_col: str = "Close",
    pred_high_col: str = "pred_high",
    meta_label_col: str = "meta_label",
    exit_type_col: str = "exit_type",
) -> np.ndarray:
    """
    Compute per-trade simple returns for Sharpe Ratio calculation.

    For profitable trades (meta_label=1), the return is the distance
    to the predicted high.  For losing trades, the return depends on
    the exit type: stop-loss returns are estimated from the lower
    barrier, and timeouts are assumed to break even (0% return).

    Args:
        df (pd.DataFrame):
            Meta-label DataFrame containing price, prediction, and
            label columns.
        close_col (str):
            Entry price column.
        pred_high_col (str):
            Predicted high (take-profit level) column.
        meta_label_col (str):
            Binary label column (1=win, 0=loss).
        exit_type_col (str):
            Exit type column (``'take_profit'``, ``'stop_loss'``,
            ``'timeout'``).

    Returns:
        np.ndarray:
            Per-trade return array of shape ``[N]``.
    """
    entry = df[close_col].values
    pred_high = df[pred_high_col].values
    labels = df[meta_label_col].values
    exits = df[exit_type_col].values

    returns = np.zeros(len(df), dtype=np.float64)

    for i in range(len(df)):
        if labels[i] == 1:
            returns[i] = (pred_high[i] / entry[i]) - 1.0
        elif exits[i] == "stop_loss" and "lower_barrier" in df.columns:
            returns[i] = (df["lower_barrier"].iloc[i] / entry[i]) - 1.0
        else:
            returns[i] = 0.0

    return returns


def full_evaluation(
    df: pd.DataFrame,
    threshold: float = 0.5,
    sr_benchmark: float = 0.0,
) -> Dict[str, Dict[str, float]]:
    """
    Run the complete evaluation suite: baseline, filtered, and PSR.

    This is the top-level function intended for use by evaluation
    scripts and notebooks.  It expects the DataFrame produced by
    ``train_meta.py`` (which contains ``meta_label``, ``meta_proba``,
    and ``meta_pred`` columns).

    Evaluation procedure
    --------------------
    1. **Baseline** -- All primary model signals are taken.  Precision
       equals the base rate of profitable trades.
    2. **Filtered** -- Only signals where the meta-classifier's
       confidence exceeds ``threshold`` are taken.  Precision should
       increase at the cost of some Recall.
    3. **PSR (baseline)** -- Probabilistic Sharpe Ratio of taking all
       trades.
    4. **PSR (filtered)** -- Probabilistic Sharpe Ratio of taking only
       the filtered trades.

    Args:
        df (pd.DataFrame):
            Output of ``train_meta.py`` with columns ``meta_label``,
            ``meta_proba``, ``meta_pred``, ``Close``, ``pred_high``,
            ``exit_type``, and optionally ``lower_barrier``.
        threshold (float):
            Classification threshold for the meta-classifier.
        sr_benchmark (float):
            Benchmark Sharpe Ratio for the PSR calculation.

    Returns:
        Dict[str, Dict[str, float]]:
            Nested dictionary with keys ``'baseline'``, ``'filtered'``,
            ``'psr_baseline'``, ``'psr_filtered'``.
    """
    y_true = df["meta_label"].values.astype(int)
    y_proba = df["meta_proba"].values
    y_pred = (y_proba >= threshold).astype(int)

    baseline = evaluate_baseline(y_true)
    filtered = evaluate_filtered(y_true, y_pred, y_proba)

    # Trade returns
    all_returns = compute_trade_returns(df)
    psr_baseline = probabilistic_sharpe_ratio(all_returns, sr_benchmark)

    taken_mask = y_pred == 1
    if taken_mask.any():
        filtered_returns = all_returns[taken_mask]
        psr_filtered = probabilistic_sharpe_ratio(filtered_returns, sr_benchmark)
    else:
        psr_filtered = probabilistic_sharpe_ratio(np.array([]), sr_benchmark)

    return {
        "baseline": baseline,
        "filtered": filtered,
        "psr_baseline": psr_baseline,
        "psr_filtered": psr_filtered,
    }


def print_evaluation_report(
    results: Dict[str, Dict[str, float]],
    threshold: float = 0.5,
) -> None:
    """
    Print a formatted evaluation report to stdout.

    Args:
        results (Dict[str, Dict[str, float]]):
            Output of ``full_evaluation()``.
        threshold (float):
            The threshold used (displayed in the header).
    """
    b = results["baseline"]
    f = results["filtered"]
    pb = results["psr_baseline"]
    pf = results["psr_filtered"]

    print("\n" + "=" * 65)
    print("META-LABELING EVALUATION REPORT")
    print("=" * 65)

    print(f"\n{'Metric':<25s} {'Baseline':>12s} {'Filtered':>12s} {'Delta':>10s}")
    print("-" * 65)

    for key in ["precision", "recall", "f1_score"]:
        bv = b[key]
        fv = f[key]
        delta = fv - bv
        sign = "+" if delta >= 0 else ""
        print(f"  {key.replace('_', ' ').title():<23s} {bv:>12.4f} {fv:>12.4f} {sign}{delta:>9.4f}")

    print("-" * 65)
    print(f"  {'Trades taken':<23s} {b['n_trades']:>12d} {f['n_trades_taken']:>12d}"
          f" {f['n_trades_taken'] - b['n_trades']:>+10d}")
    print(f"  {'Filter rate':<23s} {'0.0%':>12s} {100*f['filter_rate']:>11.1f}%")
    print(f"  {'Threshold':<23s} {'N/A':>12s} {threshold:>12.2f}")

    if "mean_confidence_taken" in f:
        print(f"  {'Avg conf (taken)':<23s} {'':>12s} {f['mean_confidence_taken']:>12.4f}")
        print(f"  {'Avg conf (skipped)':<23s} {'':>12s} {f['mean_confidence_skipped']:>12.4f}")

    print(f"\n{'Confusion Matrix (Filtered)':}")
    print(f"  TP={f['tp']:>5d}  FP={f['fp']:>5d}")
    print(f"  FN={f['fn']:>5d}  TN={f['tn']:>5d}")

    print(f"\n{'Risk-Adjusted Performance':}")
    print(f"  {'Metric':<25s} {'Baseline':>12s} {'Filtered':>12s}")
    print(f"  {'-'*55}")
    print(f"  {'Sharpe Ratio (ann.)':<25s} {pb['sharpe_ratio']:>12.4f} {pf['sharpe_ratio']:>12.4f}")
    print(f"  {'PSR':<25s} {pb['psr']:>12.4f} {pf['psr']:>12.4f}")
    print(f"  {'Skewness':<25s} {pb['skewness']:>12.4f} {pf['skewness']:>12.4f}")
    print(f"  {'Kurtosis':<25s} {pb['kurtosis']:>12.4f} {pf['kurtosis']:>12.4f}")
    print(f"  {'Observations':<25s} {pb['n_observations']:>12d} {pf['n_observations']:>12d}")

    print("=" * 65 + "\n")
