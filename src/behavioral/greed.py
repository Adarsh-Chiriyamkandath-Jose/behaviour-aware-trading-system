"""Greed indicator: strong momentum + high volume."""
from __future__ import annotations

import pandas as pd


def compute_greed(df: pd.DataFrame, momentum_window: int = 10, volume_window: int = 20) -> pd.Series:
    """Higher when momentum is strong AND volume is elevated. TODO: implement."""
    raise NotImplementedError
