"""Fear indicator: high volatility + negative returns."""
from __future__ import annotations

import pandas as pd


def compute_fear(df: pd.DataFrame, volatility_window: int = 20) -> pd.Series:
    """Higher when volatility is high AND returns are negative. TODO: implement."""
    raise NotImplementedError
