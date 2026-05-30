"""Time-aware data splitting.

The one rule that matters most for a trading model: the model must never see
the future while training. A random train/test split breaks that rule, so we
only ever split chronologically.
"""
from __future__ import annotations

from typing import Iterator

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


def walk_forward_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    gap: int = 0,
) -> Iterator[tuple[pd.Index, pd.Index]]:
    """Yield (train_index, test_index) pairs in chronological order.

    Each fold trains on everything up to a point and tests on the block right
    after it. `gap` inserts an embargo of N rows between the two so a
    multi-day-ahead label can't straddle the boundary.
    """
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    for train_pos, test_pos in splitter.split(df):
        yield df.index[train_pos], df.index[test_pos]


def single_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    gap: int = 0,
) -> tuple[pd.Index, pd.Index]:
    """One chronological cut: oldest rows train, newest rows test."""
    n = len(df)
    cut = n - int(n * test_size)
    train_index = df.index[: max(cut - gap, 0)]
    test_index = df.index[cut:]
    return train_index, test_index


def make_splits(df: pd.DataFrame, config: dict) -> list[tuple[pd.Index, pd.Index]]:
    """Build splits according to the `split` section of config.yaml."""
    cfg = config["split"]
    if cfg.get("scheme", "walk_forward") == "single":
        return [single_split(df, cfg.get("test_size", 0.2), cfg.get("gap", 0))]
    return list(walk_forward_splits(df, cfg.get("n_splits", 5), cfg.get("gap", 0)))
