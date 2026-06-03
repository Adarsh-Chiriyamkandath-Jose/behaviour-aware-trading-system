"""Fear indicator: the market is fearful when it's falling *and* volatile."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_fear(df: pd.DataFrame, volatility_window: int = 20) -> pd.Series:
    """Downside deviation - volatility measured on losing days only.

    The proposal frames fear as "high volatility + negative returns". We capture
    both in one quantity: the root-mean-square of negative returns over a window.
    It is ordinary volatility (dispersion) restricted to the downside, so a calm
    market and a market that only rises both read as low fear, while a stretch of
    large, frequent drops reads as high fear. This is the semi-deviation behind
    the Sortino ratio, so it's a recognised risk measure rather than an ad-hoc one.
    """
    returns = df["Close"].pct_change()
    losses = returns.clip(upper=0.0)
    fear = np.sqrt(losses.pow(2).rolling(volatility_window).mean())
    fear.name = "fear"
    return fear
