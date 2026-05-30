"""Rolling distributional features of daily returns."""
from __future__ import annotations

import pandas as pd


def add_rolling_stats(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Add rolling mean, std, skew and kurtosis of returns over each window."""
    df = df.copy()
    if "return" not in df.columns:
        df["return"] = df["Close"].pct_change()

    for w in windows:
        r = df["return"].rolling(w)
        df[f"roll_mean_{w}"] = r.mean()
        df[f"roll_std_{w}"] = r.std()
        df[f"roll_skew_{w}"] = r.skew()
        df[f"roll_kurt_{w}"] = r.kurt()
    return df
