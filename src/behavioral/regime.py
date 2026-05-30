"""Market regime detection (STRETCH GOAL).

Label periods as e.g. bull / bear / high-volatility so results can be analysed
per regime. Keep only if time allows.
"""
from __future__ import annotations

import pandas as pd


def detect_regime(df: pd.DataFrame) -> pd.Series:
    """Assign a regime label per row. TODO: implement (stretch)."""
    raise NotImplementedError
