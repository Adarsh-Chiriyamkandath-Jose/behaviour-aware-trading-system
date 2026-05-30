"""Herd-behavior indicator: volume spikes + large price moves."""
from __future__ import annotations

import pandas as pd


def compute_herd(df: pd.DataFrame, volume_spike_z: float = 2.0, price_move_z: float = 2.0) -> pd.Series:
    """Higher when abnormal volume coincides with large price moves. TODO: implement."""
    raise NotImplementedError
