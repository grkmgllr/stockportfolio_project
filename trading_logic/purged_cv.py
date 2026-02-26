"""
Purged K-Fold cross-validation for financial time series.

Implements the Purged and Embargo cross-validation framework described in
Marcos Lopez de Prado's *Advances in Financial Machine Learning* (Ch. 7).

Standard K-Fold cross-validation assumes i.i.d. observations.  In financial
time series this assumption is violated in two ways:

    1. **Overlapping event windows** -- a training observation whose trade
       is still "alive" (its barrier has not yet been hit) during the test
       period leaks future information into the training set.
    2. **Serial correlation / market memory** -- even after removing
       overlapping events, observations *immediately* following the test
       set may still carry lingering market reactions from events inside
       the test period.

This module solves both problems:

    Purging  -- removes training observations whose event window
                [t_start, t_end] overlaps with any test observation's
                event window.
    Embargo  -- after each test fold, an additional time gap of
                ``n_embargo`` bars is enforced: training observations
                that fall within this gap are also removed.

Design goals
------------
- Provide a scikit-learn compatible cross-validator that can be passed
  directly to ``lightgbm.cv()`` or used manually in a training loop.
- Accept event spans as a separate array (produced by
  ``triple_barrier.get_event_spans``) so the CV logic is decoupled from
  the labeling logic.
- Remain purely time-based: folds are contiguous blocks of time, never
  shuffled, preserving the temporal ordering of the data.
"""

import numpy as np
from typing import List, Tuple, Optional


class PurgedKFold:
    """
    Purged K-Fold cross-validator for overlapping financial events.

    Splits the dataset into ``n_splits`` contiguous time-ordered folds.
    For each fold used as a test set, training indices are purged of any
    observation whose event window overlaps with the test period, and an
    optional embargo buffer is applied after the test set.

    This prevents the two main sources of data leakage in financial
    time-series cross-validation: look-ahead bias from overlapping events
    and information seepage from serial correlation.

    Attributes
    ----------
    n_splits : int
        Number of folds.
    n_embargo : int
        Number of bars to embargo after each test fold.
    t_start : np.ndarray
        Per-observation event start indices (integer positions).
    t_end : np.ndarray
        Per-observation event end indices (integer positions).

    Shape contract
    --------------
    Inputs
        X : array-like of shape [N, F]
            Feature matrix (only its length is used for splitting).
        y : ignored
            Present for scikit-learn API compatibility.
        groups : ignored
            Present for scikit-learn API compatibility.

    Yields
        train_indices : np.ndarray
            Purged + embargoed training indices for this fold.
        test_indices : np.ndarray
            Contiguous test indices for this fold.

    Notes
    -----
    - Folds are always contiguous blocks ordered by time.  They are never
      shuffled.
    - The purging logic checks for *any* overlap between a training
      observation's event window and the test period's time range.  An
      observation is purged if its event started before the test period
      ends *and* its event ends after the test period starts (standard
      interval overlap test).
    - The embargo is applied *after* purging.  It removes training
      observations in the range [test_end, test_end + n_embargo).
    """

    def __init__(
        self,
        n_splits: int = 5,
        n_embargo: int = 0,
        t_start: Optional[np.ndarray] = None,
        t_end: Optional[np.ndarray] = None,
    ):
        """
        Initialize the purged cross-validator.

        Args:
            n_splits (int):
                Number of folds (must be >= 2).
            n_embargo (int):
                Number of bars to embargo after each test fold.  A value
                of 0 disables the embargo (purging still applies).  A
                typical choice is 1-2% of the dataset length, or the
                expected half-life of the autocorrelation function.
            t_start (np.ndarray, optional):
                Integer array of event start positions, one per observation.
                If None, each observation's event is assumed to span only
                its own index (no overlap possible beyond adjacency).
            t_end (np.ndarray, optional):
                Integer array of event end positions, one per observation.
                Must be provided together with ``t_start``.

        Raises:
            ValueError:
                If ``n_splits < 2`` or if ``t_start`` and ``t_end``
                have mismatched lengths.
        """
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")
        self.n_splits = n_splits
        self.n_embargo = n_embargo

        if t_start is not None and t_end is not None:
            t_start = np.asarray(t_start)
            t_end = np.asarray(t_end)
            if len(t_start) != len(t_end):
                raise ValueError(
                    f"t_start length ({len(t_start)}) != "
                    f"t_end length ({len(t_end)})"
                )
        self.t_start = t_start
        self.t_end = t_end

    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate purged train/test index pairs.

        The dataset is divided into ``n_splits`` contiguous folds ordered
        by time.  Each fold is used exactly once as the test set.  The
        remaining folds form the candidate training set, which is then
        purged of overlapping events and embargoed.

        Args:
            X (np.ndarray):
                Feature matrix of shape [N, F].  Only ``N`` is used.
            y (np.ndarray, optional):
                Ignored.  Present for scikit-learn compatibility.
            groups (np.ndarray, optional):
                Ignored.  Present for scikit-learn compatibility.

        Returns:
            List[Tuple[np.ndarray, np.ndarray]]:
                List of ``(train_indices, test_indices)`` tuples, one per
                fold.  Indices are integer positions into the original
                array.
        """
        n_samples = len(X)
        indices = np.arange(n_samples)

        if self.t_start is None or self.t_end is None:
            t_start = indices.copy()
            t_end = indices.copy()
        else:
            t_start = self.t_start
            t_end = self.t_end

        fold_boundaries = np.array_split(indices, self.n_splits)
        splits = []

        for fold_idx in range(self.n_splits):
            test_indices = fold_boundaries[fold_idx]
            if len(test_indices) == 0:
                continue

            test_start = test_indices[0]
            test_end = test_indices[-1]

            # Candidate training indices = everything outside this fold
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[test_indices] = False

            # --- Purging ---
            # Remove training observations whose event window overlaps
            # with the test period.  Two intervals [a, b] and [c, d]
            # overlap iff a <= d and c <= b.
            for i in range(n_samples):
                if not train_mask[i]:
                    continue
                if t_start[i] <= test_end and t_end[i] >= test_start:
                    train_mask[i] = False

            # --- Embargo ---
            # Remove training observations in [test_end+1, test_end+n_embargo]
            if self.n_embargo > 0:
                embargo_start = test_end + 1
                embargo_end = min(test_end + self.n_embargo, n_samples - 1)
                train_mask[embargo_start : embargo_end + 1] = False

            train_indices = indices[train_mask]
            splits.append((train_indices, test_indices))

        return splits

    def get_n_splits(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
    ) -> int:
        """
        Return the number of folds.

        This method exists for scikit-learn API compatibility.

        Returns:
            int:
                Number of splitting iterations (``n_splits``).
        """
        return self.n_splits
