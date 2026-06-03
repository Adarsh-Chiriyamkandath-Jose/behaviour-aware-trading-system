"""Logistic Regression model.

Logistic regression is scale-sensitive, so we standardise features inside the
model (a StandardScaler fit on the training fold only) rather than asking every
caller to remember to do it. Wrapping the scaler with the estimator also means
no training-set statistics leak into the test fold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.base import BaseModel


class LogisticModel(BaseModel):
    name = "logistic"

    def __init__(self, C: float = 1.0, max_iter: int = 1000, random_state: int = 42):
        self.params = {"C": C, "max_iter": max_iter, "random_state": random_state}
        self.model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=C, max_iter=max_iter, random_state=random_state)),
            ]
        )
        self._columns: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LogisticModel":
        self._columns = list(X.columns)
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def feature_importances(self) -> pd.Series | None:
        if self._columns is None:
            return None
        coefs = self.model.named_steps["clf"].coef_[0]
        return pd.Series(np.abs(coefs), index=self._columns).sort_values(ascending=False)
