"""Fear indicator: the market is fearful when it's falling *and* volatile."""
from __future__ import annotations

import pandas as pd


def compute_fear(df: pd.DataFrame, volatility_window: int = 20) -> pd.Series:
    """Downside volatility - the dispersion of losing days over a window.

    Plain volatility treats sharp rallies and sharp sell-offs alike. Fear is
    one-sided: it lives in the downside. So we measure the spread of negative
    returns only. Higher means bigger, more frequent drops.
    """
    returns = df["Close"].pct_change()
    downside = returns.where(returns < 0, 0.0)
    fear = downside.rolling(volatility_window).std()
    fear.name = "fear"
    return fear
