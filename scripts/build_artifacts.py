"""Pre-build the on-disk artifacts the dashboard reads.

Writes one processed-feature CSV per ticker to ``data/processed/`` and one trained
model per (ticker, model, feature-set) to ``results/models/``. The Streamlit app
loads these when present (and config-compatible) for a faster cold start, falling
back to rebuilding if anything is missing or incompatible.

Run from the project root:

    python scripts/build_artifacts.py

Re-run it whenever you change config.yaml so the cache stays in sync (the app also
detects a changed config via a fingerprint and rebuilds automatically).
"""
from __future__ import annotations

import os
import sys

# Make `import src...` work when run as a plain script from the project root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import persistence as P
from src.config import load_config
from src.data.loader import load_ohlcv
from src.data.splitter import single_split
from src.experiments.ablation import FEATURE_SETS, build_model_frame, feature_columns
from src.models import build_model
from src.utils import get_logger, set_seed

logger = get_logger("build_artifacts")


def main(config_path: str = "config.yaml") -> None:
    config = load_config(config_path)
    set_seed(config.get("seed", 42))

    tickers = config["data"]["tickers"]
    models = config["models"].get("active", ["logistic", "random_forest", "xgboost"])
    test_size = config["split"].get("test_size", 0.2)

    n_frames = n_models = 0
    for ticker in tickers:
        raw = load_ohlcv(
            ticker, config["data"]["start_date"], config["data"]["end_date"], config["data"]["raw_dir"]
        )
        frame = build_model_frame(raw, config)
        P.save_processed_frame(frame, ticker, config)
        n_frames += 1

        train_idx, _ = single_split(frame, test_size)
        for model_name in models:
            for feature_set in FEATURE_SETS:
                cols = feature_columns(frame, feature_set)
                model = build_model(model_name, config).fit(
                    frame.loc[train_idx, cols], frame.loc[train_idx, "target"]
                )
                P.save_model(f"{ticker}_{model_name}_{feature_set}", model, config)
                n_models += 1

    logger.info("Done: wrote %d processed frames and %d models.", n_frames, n_models)
    print(f"\nWrote {n_frames} processed CSVs to data/processed/ and {n_models} models to results/models/.")


if __name__ == "__main__":
    main()
