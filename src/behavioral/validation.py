"""Sanity-check behavioural indicators against future returns.

Before trusting an indicator we ask one question: do higher readings line up
with different forward returns? If not, it's probably noise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def validate_indicator(
    indicator: pd.Series,
    forward_returns: pd.Series,
    n_buckets: int = 5,
) -> dict:
    """Return basic diagnostics for one indicator.

    - ic: Spearman rank correlation with forward returns (the "information
      coefficient") - scale-free and robust to outliers.
    - coverage: fraction of rows that actually carry a value.
    - top/bottom bucket: mean forward return in the highest vs lowest readings,
      i.e. does the signal separate outcomes at all.
    """
    data = pd.concat([indicator, forward_returns], axis=1).dropna()
    data.columns = ["indicator", "fwd_return"]

    if data.empty:
        return {"ic": np.nan, "coverage": 0.0, "top_bucket": np.nan, "bottom_bucket": np.nan}

    ic = data["indicator"].corr(data["fwd_return"], method="spearman")

    try:
        buckets = pd.qcut(data["indicator"], n_buckets, labels=False, duplicates="drop")
        grouped = data["fwd_return"].groupby(buckets).mean()
        bottom_bucket, top_bucket = grouped.iloc[0], grouped.iloc[-1]
    except ValueError:
        # Too few distinct values to bucket (common for the gated herd signal).
        top_bucket = bottom_bucket = np.nan

    return {
        "ic": ic,
        "coverage": len(data) / len(indicator),
        "top_bucket": top_bucket,
        "bottom_bucket": bottom_bucket,
    }
