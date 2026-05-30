"""Load and validate the project configuration (config.yaml)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    """Read the YAML config into a dict.

    TODO: add schema validation (required keys, value ranges).
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
