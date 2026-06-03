"""Tests for the model classes and the factory."""
import numpy as np
import pandas as pd
import pytest

from src.models import (
    LogisticModel,
    RandomForestModel,
    XGBoostModel,
    build_active_models,
    build_model,
)

MODELS = [LogisticModel, RandomForestModel, XGBoostModel]


@pytest.fixture
def xy():
    """A separable-ish binary problem so models reach non-trivial accuracy."""
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame(
        {"a": rng.normal(0, 1, n), "b": rng.normal(0, 1, n), "c": rng.normal(0, 1, n)}
    )
    y = (X["a"] + 0.5 * X["b"] + rng.normal(0, 0.3, n) > 0).astype(int)
    return X, y


@pytest.mark.parametrize("cls", MODELS)
def test_fit_predict_shapes_and_range(cls, xy):
    X, y = xy
    model = cls().fit(X, y)

    pred = model.predict(X)
    proba = model.predict_proba(X)
    assert len(pred) == len(X)
    assert set(np.unique(pred)).issubset({0, 1})
    # predict_proba is the 1-D positive-class probability, per BaseModel contract.
    assert proba.shape == (len(X),)
    assert ((proba >= 0) & (proba <= 1)).all()


@pytest.mark.parametrize("cls", MODELS)
def test_learns_better_than_chance(cls, xy):
    X, y = xy
    model = cls().fit(X, y)
    assert (model.predict(X) == y.to_numpy()).mean() > 0.6


@pytest.mark.parametrize("cls", MODELS)
def test_feature_importances_align_with_columns(cls, xy):
    X, y = xy
    model = cls().fit(X, y)
    imp = model.feature_importances()
    assert imp is not None
    assert set(imp.index) == set(X.columns)


def test_factory_builds_from_config(sample_config):
    model = build_model("random_forest", sample_config)
    assert isinstance(model, RandomForestModel)
    assert model.params["n_estimators"] == 50  # pulled from sample_config

    active = build_active_models(sample_config)
    assert [m.name for m in active] == ["logistic", "random_forest"]


def test_factory_rejects_unknown_model():
    with pytest.raises(KeyError):
        build_model("transformer", {})
