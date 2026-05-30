"""Define the prediction target (up/down direction)."""
from __future__ import annotations

import pandas as pd


def make_direction_label(df: pd.DataFrame, horizon: int = 1, threshold: float = 0.0) -> pd.Series:
    """Binary label: 1 if forward `horizon`-day return > threshold else 0.

    NOTE: the label looks into the FUTURE; make sure no future info leaks into
    the features. TODO: implement.
    """
    raise NotImplementedError
