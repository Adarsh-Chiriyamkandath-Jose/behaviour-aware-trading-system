"""Optional on-disk cache for processed feature frames and trained models.

These artifacts are a best-effort *speed-up*, never a hard dependency. A saved
file is used only when it exists, matches the current config (via a fingerprint),
and loads cleanly. If any of those fail — missing file, changed config, or a
library-version mismatch on another machine (e.g. a cloud host on a different
Python) — callers fall back to rebuilding from source. That keeps the app working
everywhere while making `data/processed/` and `results/models/` concrete instead
of empty scaffolding.

Layout written:
  data/processed/<TICKER>.csv          one model-ready frame per ticker
  data/processed/_manifest.json        {ticker: config-fingerprint}
  results/models/<TICKER>_<model>_<featureset>.joblib   a trained model + its fingerprint
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.utils import ensure_dir, get_logger

logger = get_logger(__name__)

# Only the parts of config that actually change features/labels/models. If any of
# these change, cached artifacts are considered stale and are rebuilt.
_FINGERPRINT_KEYS = ("data", "labels", "features", "behavioral", "split", "models", "seed")


def config_fingerprint(config: dict[str, Any]) -> str:
    """A short, stable hash of the config parts that determine the artifacts."""
    subset = {k: config.get(k) for k in _FINGERPRINT_KEYS}
    blob = json.dumps(subset, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def _processed_dir(config: dict[str, Any]) -> Path:
    return Path(config.get("data", {}).get("processed_dir", "data/processed"))


def _models_dir(config: dict[str, Any]) -> Path:
    return Path(config.get("paths", {}).get("results_models", "results/models"))


# ---- processed feature frames ----------------------------------------------

def _manifest_path(config: dict[str, Any]) -> Path:
    return _processed_dir(config) / "_manifest.json"


def _read_manifest(config: dict[str, Any]) -> dict[str, str]:
    path = _manifest_path(config)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}


def save_processed_frame(frame: pd.DataFrame, ticker: str, config: dict[str, Any]) -> Path:
    """Write one ticker's model-ready frame to data/processed/<TICKER>.csv."""
    ensure_dir(_processed_dir(config))
    path = _processed_dir(config) / f"{ticker}.csv"
    frame.to_csv(path)

    manifest = _read_manifest(config)
    manifest[ticker] = config_fingerprint(config)
    _manifest_path(config).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("Cached processed frame: %s (%d rows)", path, len(frame))
    return path


def load_processed_frame(ticker: str, config: dict[str, Any]) -> pd.DataFrame | None:
    """Return the cached frame for `ticker`, or None if absent/stale/unreadable."""
    path = _processed_dir(config) / f"{ticker}.csv"
    if not path.exists():
        return None
    if _read_manifest(config).get(ticker) != config_fingerprint(config):
        logger.info("Processed cache for %s is stale (config changed); rebuilding.", ticker)
        return None
    try:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Could not read processed cache %s (%s); rebuilding.", path, exc)
        return None


# ---- trained models ---------------------------------------------------------

def save_model(key: str, model, config: dict[str, Any]) -> Path:
    """Persist a trained model (with its config fingerprint) to results/models/."""
    ensure_dir(_models_dir(config))
    path = _models_dir(config) / f"{key}.joblib"
    joblib.dump({"model": model, "fingerprint": config_fingerprint(config)}, path, compress=3)
    logger.info("Cached model: %s", path)
    return path


def load_model(key: str, config: dict[str, Any]):
    """Return the cached model for `key`, or None if absent/stale/unloadable.

    Unloadable covers the realistic cloud case: a pickle written under one library
    version failing to deserialise under another. We swallow it and let the caller
    retrain — a slower cold start, never a crash.
    """
    path = _models_dir(config) / f"{key}.joblib"
    if not path.exists():
        return None
    try:
        bundle = joblib.load(path)
    except Exception as exc:  # pragma: no cover - version-mismatch path
        logger.warning("Could not load cached model %s (%s); retraining.", path, exc)
        return None
    if not isinstance(bundle, dict) or bundle.get("fingerprint") != config_fingerprint(config):
        logger.info("Cached model %s is stale (config changed); retraining.", key)
        return None
    return bundle["model"]
