"""Time-aware data splitting for trading models.

CRITICAL: never use a random train/test split on time-series financial data —
it leaks the future into the past. Use walk-forward / expanding windows.
"""
from __future__ import annotations

from typing import Iterator

import pandas as pd


def walk_forward_splits(
    df: pd.DataFrame, n_splits: int = 5, gap: int = 0
) -> Iterator[tuple[pd.Index, pd.Index]]:
    """Yield (train_index, test_index) pairs in chronological order.

    `gap` is an embargo (days) between train and test to avoid leakage.
    TODO: implement (consider sklearn.model_selection.TimeSeriesSplit).
    """
    raise NotImplementedError
