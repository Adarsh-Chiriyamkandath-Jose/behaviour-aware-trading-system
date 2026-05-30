"""Validate that behavioral proxies are meaningful, not noise.

E.g. correlation with forward returns, separation of outcomes across
high/low values of each indicator.
"""
from __future__ import annotations

import pandas as pd


def validate_indicator(indicator: pd.Series, forward_returns: pd.Series) -> dict:
    """Return diagnostics for a single behavioral indicator. TODO: implement."""
    raise NotImplementedError
