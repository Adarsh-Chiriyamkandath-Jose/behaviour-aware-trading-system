"""Random Forest model.

Trees are scale-invariant, so no standardisation here. `class_weight="balanced"`
guards against the mild up/down imbalance in the direction label.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.models.base import BaseModel


class RandomForestModel(BaseModel):
    name = "random_forest"

    def __init__(self, n_estimators: int = 300, max_depth: int = 8, random_state: int = 42):
        self.params = {"n_estimators": n_estimators, "max_depth": max_depth, "random_state": random_state}
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
        )
        self._columns: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestModel":
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
        return pd.Series(self.model.feature_importances_, index=self._columns).sort_values(ascending=False)
