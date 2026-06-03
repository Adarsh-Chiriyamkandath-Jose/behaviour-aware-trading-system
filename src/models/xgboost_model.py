"""XGBoost model.

Gradient-boosted trees, our most expressive baseline. Like the random forest it
needs no scaling. We keep the booster deterministic via `random_state` so folds
are reproducible.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.models.base import BaseModel


class XGBoostModel(BaseModel):
    name = "xgboost"

    def __init__(
        self,
        n_estimators: int = 400,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        random_state: int = 42,
    ):
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "random_state": random_state,
        }
        # Imported lazily so the rest of src.models works even if xgboost isn't
        # installed; you only need the library when you actually build this model.
        from xgboost import XGBClassifier

        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
        )
        self._columns: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBoostModel":
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
