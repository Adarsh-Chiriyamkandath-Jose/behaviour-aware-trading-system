"""Logistic Regression model."""
from __future__ import annotations

from src.models.base import BaseModel


class LogisticModel(BaseModel):
    def __init__(self, C: float = 1.0, max_iter: int = 1000):
        self.params = {"C": C, "max_iter": max_iter}
        self.model = None  # TODO: sklearn LogisticRegression

    def fit(self, X, y): raise NotImplementedError
    def predict(self, X): raise NotImplementedError
    def predict_proba(self, X): raise NotImplementedError
