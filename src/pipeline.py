"""End-to-end orchestration: data -> features -> model -> backtest -> report.

This is the glue that ties every module together so notebooks stay thin.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.experiments.ablation import run_ablation
from src.utils import ensure_dir, get_logger, set_seed

logger = get_logger(__name__)


def run_pipeline(config: dict[str, Any]) -> dict[str, Any]:
    """Run the full experiment described by `config`.

    Flow (steps 1-7 below are folded into `run_ablation`, which walks the data ->
    features -> labels -> split -> train -> evaluate path for every ticker, model
    and feature set):

      1. load + clean OHLCV               (src.data.loader / cleaner)
      2. build quant + behavioral features (src.features / src.behavioral)
      3. create labels                     (src.data.labels)
      4. time-aware split                  (src.data.splitter)
      5. train models                      (src.models)
      6. evaluate (classification + financial) (src.evaluation)
      7. ablation tables + save results    (src.experiments)

    The realistic backtest and the report figures (src.backtest / src.visualize)
    are Member C's; they consume the `per_fold` predictions/returns saved here.
    Returns the ablation result tables and writes them to `paths.results_tables`.
    """
    set_seed(config.get("seed", 42))
    logger.info("Pipeline starting with tickers=%s", config["data"]["tickers"])

    results = run_ablation(config)

    tables_dir = ensure_dir(config.get("paths", {}).get("results_tables", "results/tables"))
    for name, table in results.items():
        out = Path(tables_dir) / f"ablation_{name}.csv"
        table.to_csv(out, index=False)
        logger.info("Wrote %s (%d rows)", out, len(table))

    return results
