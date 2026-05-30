"""Rule-based strategies mapping model output to buy/sell/hold signals."""
from __future__ import annotations

import pandas as pd


def signal_from_predictions(predictions: pd.Series) -> pd.Series:
    """Map class predictions (up/down) to position signals. TODO: implement."""
    raise NotImplementedError
