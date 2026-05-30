"""Random Forest model."""
from __future__ import annotations

from src.models.base import BaseModel


class RandomForestModel(BaseModel):
    def __init__(self, n_estimators: int = 300, max_depth: int = 8, random_state: int = 42):
        self.params = {"n_estimators": n_estimators, "max_depth": max_depth, "random_state": random_state}
        self.model = None  # TODO: sklearn RandomForestClassifier

    def fit(self, X, y): raise NotImplementedError
    def predict(self, X): raise NotImplementedError
    def predict_proba(self, X): raise NotImplementedError
