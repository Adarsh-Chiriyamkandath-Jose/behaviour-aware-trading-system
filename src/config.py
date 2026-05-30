"""Load and validate the project configuration (config.yaml).

Every module reads its settings from here, so the whole pipeline can be retuned
in one place without editing code.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REQUIRED_SECTIONS = ["data", "labels", "features", "behavioral", "split", "models"]


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    missing = [s for s in REQUIRED_SECTIONS if s not in config]
    if missing:
        raise KeyError(f"config.yaml is missing required sections: {missing}")

    return config
