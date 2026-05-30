"""End-to-end orchestration: data -> features -> model -> backtest -> report.

This is the glue that ties every module together so notebooks stay thin.
"""
from __future__ import annotations

from typing import Any

from src.utils import get_logger, set_seed

logger = get_logger(__name__)


def run_pipeline(config: dict[str, Any]) -> None:
    """Run the full experiment described by `config`.

    Intended flow:
      1. load + clean OHLCV               (src.data.loader / cleaner)
      2. build quant + behavioral features (src.features / src.behavioral)
      3. create labels                     (src.data.labels)
      4. time-aware split                  (src.data.splitter)
      5. train models                      (src.models)
      6. evaluate + backtest               (src.evaluation / src.backtest)
      7. run ablation + save results       (src.experiments / src.visualize)

    TODO: implement once the underlying modules are ready.
    """
    set_seed(config.get("seed", 42))
    logger.info("Pipeline starting with tickers=%s", config["data"]["tickers"])
    raise NotImplementedError("Wire up modules as they are implemented.")
