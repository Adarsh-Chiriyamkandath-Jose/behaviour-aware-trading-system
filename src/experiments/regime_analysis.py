"""Per-regime performance analysis (STRETCH GOAL).

Does the behaviour-aware model earn its keep mainly in turbulent regimes (where
fear/greed/herd should matter most) and add little in calm bull markets? We
answer it by slicing one out-of-sample test window by the `regime` label Member A
produced and scoring each slice.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.data.splitter import single_split
from src.evaluation.classification import classification_metrics
from src.evaluation.financial import equity_curve, financial_metrics, strategy_returns
from src.experiments.ablation import build_model_frame, feature_columns
from src.models import build_model
from src.utils import get_logger, set_seed

logger = get_logger(__name__)


def regime_breakdown(
    y_true: pd.Series,
    y_pred,
    fwd_returns: pd.Series,
    regimes: pd.Series,
    transaction_cost: float = 0.0005,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """Classification + financial metrics computed within each regime label.

    All inputs share the test-window index. Returns one row per regime (plus an
    "ALL" row) so a regime can be compared against the blended baseline.
    """
    pred = pd.Series(y_pred, index=y_true.index)
    rows: list[dict] = []
    for label, idx in regimes.groupby(regimes).groups.items():
        idx = y_true.index.intersection(idx)
        if len(idx) == 0:
            continue
        cls = classification_metrics(y_true.loc[idx], pred.loc[idx])
        rets = strategy_returns(pred.loc[idx], fwd_returns.loc[idx], transaction_cost)
        fin = financial_metrics(equity_curve(rets), periods_per_year)
        rows.append({"regime": label, "n": len(idx), **cls, **fin})

    cls_all = classification_metrics(y_true, pred)
    rets_all = strategy_returns(pred, fwd_returns, transaction_cost)
    fin_all = financial_metrics(equity_curve(rets_all), periods_per_year)
    rows.append({"regime": "ALL", "n": len(y_true), **cls_all, **fin_all})
    return pd.DataFrame(rows)


def run_regime_analysis(
    config: dict[str, Any],
    raw_data: dict[str, pd.DataFrame] | None = None,
    model_name: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Train the behaviour-aware model on a single chronological split and break
    its out-of-sample performance down by regime, per ticker.
    """
    set_seed(config.get("seed", 42))
    name = model_name or config["models"].get("active", ["random_forest"])[0]
    from src.data.loader import load_ohlcv  # local import keeps offline tests import-light

    out: dict[str, pd.DataFrame] = {}
    for ticker in config["data"]["tickers"]:
        raw = raw_data[ticker] if raw_data is not None else load_ohlcv(
            ticker, config["data"]["start_date"], config["data"]["end_date"], config["data"]["raw_dir"]
        )
        frame = build_model_frame(raw, config)
        cols = feature_columns(frame, "quant_plus_behavioral")

        train_idx, test_idx = single_split(frame, config["split"].get("test_size", 0.2), config["split"].get("gap", 0))
        model = build_model(name, config).fit(frame.loc[train_idx, cols], frame.loc[train_idx, "target"])
        pred = model.predict(frame.loc[test_idx, cols])

        horizon = max(1, config["labels"]["horizon"])
        out[ticker] = regime_breakdown(
            frame.loc[test_idx, "target"],
            pred,
            frame.loc[test_idx, "fwd_return"],
            frame.loc[test_idx, "regime"],
            config.get("backtest", {}).get("transaction_cost", 0.0005),
            max(1, round(252 / horizon)),
        )
    return out
