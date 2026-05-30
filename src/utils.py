"""Shared helpers: logging, reproducibility, paths."""
from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set seeds for reproducibility across random / numpy."""
    random.seed(seed)
    np.random.seed(seed)


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    return logging.getLogger(name)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
