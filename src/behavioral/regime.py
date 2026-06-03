"""Market regime labelling (stretch goal).

A coarse description of the backdrop each day sits in, so results can later be
broken down by regime. Kept simple and transparent on purpose.
"""
from __future__ import annotations

import pandas as pd


def detect_regime(
    df: pd.DataFrame,
    trend_window: int = 200,
    vol_window: int = 20,
) -> pd.Series:
    """Label each day as bull/bear crossed with calm/volatile.

    Trend comes from price versus its long moving average; volatility is "high"
    when it sits above its own running median. Both use past data only.
    """
    close = df["Close"]
    trend_ma = close.rolling(trend_window).mean()

    trend = pd.Series(index=df.index, dtype="object")
    trend[close >= trend_ma] = "bull"
    trend[close < trend_ma] = "bear"

    vol = close.pct_change().rolling(vol_window).std()
    high_vol = vol > vol.expanding(min_periods=vol_window).median()
    vol_label = high_vol.map({True: "volatile", False: "calm"})

    regime = trend.str.cat(vol_label, sep="_")
    regime.name = "regime"
    return regime
