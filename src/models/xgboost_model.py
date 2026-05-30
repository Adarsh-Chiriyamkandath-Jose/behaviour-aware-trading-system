"""XGBoost model."""
from __future__ import annotations

from src.models.base import BaseModel


class XGBoostModel(BaseModel):
    def __init__(self, n_estimators: int = 400, max_depth: int = 5, learning_rate: float = 0.05, random_state: int = 42):
        self.params = {
            "n_estimators": n_estimators, "max_depth": max_depth,
            "learning_rate": learning_rate, "random_state": random_state,
        }
        self.model = None  # TODO: xgboost.XGBClassifier

    def fit(self, X, y): raise NotImplementedError
    def predict(self, X): raise NotImplementedError
    def predict_proba(self, X): raise NotImplementedError
