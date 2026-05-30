"""Shared helpers: logging, reproducibility, paths, and normalization."""
from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    return logging.getLogger(name)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def expanding_zscore(series: pd.Series, min_periods: int = 20) -> pd.Series:
    """Standardize a series using only past-and-present values.

    A full-sample mean/std would leak the future into every row, so we use an
    expanding window instead: each point is scaled by what was knowable up to
    that day.
    """
    mean = series.expanding(min_periods=min_periods).mean()
    std = series.expanding(min_periods=min_periods).std()
    return (series - mean) / std
