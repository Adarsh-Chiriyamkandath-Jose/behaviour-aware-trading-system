"""Shared model interface so every model is interchangeable in the pipeline."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseModel(ABC):
    """Common interface for all classifiers."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseModel": ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...
