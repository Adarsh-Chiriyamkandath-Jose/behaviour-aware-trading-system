"""Herd-behaviour indicator: the crowd moving all at once.

Herding shows up as an unusual volume spike coinciding with an unusually large
price move. We measure "unusual" in standard deviations and only register a
signal when both are extreme on the same day.
"""
from __future__ import annotations

import pandas as pd


def compute_herd(
    df: pd.DataFrame,
    volume_spike_z: float = 2.0,
    price_move_z: float = 2.0,
    window: int = 20,
) -> pd.Series:
    returns = df["Close"].pct_change()
    abs_move = returns.abs()

    vol_mean = df["Volume"].rolling(window).mean()
    vol_std = df["Volume"].rolling(window).std()
    volume_z = (df["Volume"] - vol_mean) / vol_std

    move_z = (abs_move - abs_move.rolling(window).mean()) / abs_move.rolling(window).std()

    both_extreme = (volume_z > volume_spike_z) & (move_z > price_move_z)
    herd = (volume_z * move_z).where(both_extreme, 0.0)
    herd.name = "herd"
    return herd
