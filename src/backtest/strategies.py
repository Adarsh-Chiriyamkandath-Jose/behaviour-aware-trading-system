"""Rule-based strategies mapping model output to buy/sell/hold signals.

A *signal* here is the **target position** for a day: +1 fully long, 0 in cash,
-1 short. The backtest engine takes this series and turns it into an equity
curve. Keeping the mapping in its own module means we can compare "act on the
hard class label" against "only act when the model is confident" without touching
the engine.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def signal_from_predictions(predictions, index=None, allow_short: bool = False) -> pd.Series:
    """Map hard class predictions (1 = up, 0 = down) to target positions.

    Long on an "up" call; in cash on a "down" call, or short if `allow_short`.
    """
    sig = pd.Series(np.asarray(predictions), index=index, dtype=float)
    sig = sig.where(sig > 0, -1.0 if allow_short else 0.0)
    sig.name = "signal"
    return sig


def signal_from_proba(
    proba,
    index=None,
    upper: float = 0.55,
    lower: float = 0.45,
    allow_short: bool = False,
) -> pd.Series:
    """Confidence-banded positions from positive-class probabilities.

    Go long only when the model is clearly bullish (p >= `upper`), short/flat when
    clearly bearish (p <= `lower`), and stay in cash in the uncertain middle. A
    neutral band cuts churn — and the transaction costs that come with it.
    """
    p = pd.Series(np.asarray(proba), index=index, dtype=float)
    sig = pd.Series(0.0, index=p.index)
    sig[p >= upper] = 1.0
    sig[p <= lower] = -1.0 if allow_short else 0.0
    sig.name = "signal"
    return sig
