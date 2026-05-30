"""Greed indicator: strong upward momentum backed by heavy volume."""
from __future__ import annotations

import pandas as pd


def compute_greed(
    df: pd.DataFrame,
    momentum_window: int = 10,
    volume_window: int = 20,
) -> pd.Series:
    """Upward momentum weighted by how busy trading is versus normal.

    Rising prices alone aren't greed; rising prices that everyone is piling
    into are. We keep only positive momentum and scale it by volume relative to
    its recent average.
    """
    momentum = df["Close"].pct_change(momentum_window).clip(lower=0)
    relative_volume = df["Volume"] / df["Volume"].rolling(volume_window).mean()
    greed = momentum * relative_volume
    greed.name = "greed"
    return greed
