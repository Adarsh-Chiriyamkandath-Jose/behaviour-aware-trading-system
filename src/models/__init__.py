"""Model package: a small factory so the rest of the code asks for models by
name and never imports a concrete class directly.

    model = build_model("xgboost", config)
    models = build_active_models(config)   # everything in config["models"]["active"]
"""
from __future__ import annotations

from typing import Any

from src.models.base import BaseModel
from src.models.logistic import LogisticModel
from src.models.random_forest import RandomForestModel
from src.models.xgboost_model import XGBoostModel

_REGISTRY = {
    "logistic": LogisticModel,
    "random_forest": RandomForestModel,
    "xgboost": XGBoostModel,
}


def build_model(name: str, config: dict[str, Any] | None = None) -> BaseModel:
    """Construct one model by name, pulling its hyper-parameters from config.

    The model section of config.yaml carries a global `random_state` plus a
    per-model block (e.g. `logistic: {C: 1.0, max_iter: 1000}`); we merge the
    two and hand only the kwargs each constructor accepts.
    """
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Known models: {sorted(_REGISTRY)}")

    cls = _REGISTRY[name]
    models_cfg = (config or {}).get("models", {})
    params = dict(models_cfg.get(name, {}))

    # Share the global seed with any model whose constructor accepts it.
    if "random_state" in cls.__init__.__code__.co_varnames:
        params.setdefault("random_state", models_cfg.get("random_state", 42))

    return cls(**params)


def build_active_models(config: dict[str, Any]) -> list[BaseModel]:
    """Build every model listed in config['models']['active'], in order."""
    active = config.get("models", {}).get("active", list(_REGISTRY))
    return [build_model(name, config) for name in active]


__all__ = [
    "BaseModel",
    "LogisticModel",
    "RandomForestModel",
    "XGBoostModel",
    "build_model",
    "build_active_models",
]
