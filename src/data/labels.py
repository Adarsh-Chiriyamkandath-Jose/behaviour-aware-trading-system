"""Define the prediction target: future price direction (up/down)."""
from __future__ import annotations

import pandas as pd


def make_direction_label(
    df: pd.DataFrame,
    horizon: int = 1,
    threshold: float = 0.0,
) -> pd.Series:
    """Binary target: 1 if the forward `horizon`-day return exceeds `threshold`.

    This is built from FUTURE prices on purpose - it's what we're trying to
    predict. The forward shift leaves the final `horizon` rows with no answer
    (NaN); drop those before training.
    """
    forward_return = df["Close"].shift(-horizon) / df["Close"] - 1.0
    label = (forward_return > threshold).astype("float")
    label = label.mask(forward_return.isna())
    label.name = "target"
    return label
