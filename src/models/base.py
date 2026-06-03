"""Shared model interface so every model is interchangeable in the pipeline.

Every classifier in this package speaks the same three-method language
(`fit` / `predict` / `predict_proba`) plus an optional `feature_importances`
hook. That uniformity is what lets the ablation loop swap models without caring
which one it's holding.

Convention: `predict_proba` returns the probability of the **positive class**
(label 1 = "up") as a 1-D array, not the sklearn-style two-column matrix. That's
the only number downstream code (ROC-AUC, the backtest, position sizing) ever
wants, so we standardise on it here.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseModel(ABC):
    """Common interface for all classifiers."""

    name: str = "base"

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseModel": ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Probability of the positive class (1 = up), as a 1-D array."""

    def feature_importances(self) -> pd.Series | None:
        """Per-feature importance keyed by column name, or None if undefined.

        Tree models expose impurity importances; the linear model exposes the
        magnitude of its (standardised) coefficients. Models that offer nothing
        meaningful return None. Member C's SHAP analysis is the richer story;
        this is the cheap built-in one.
        """
        return None
