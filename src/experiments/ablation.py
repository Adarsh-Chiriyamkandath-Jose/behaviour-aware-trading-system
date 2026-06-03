"""Ablation study — THE core research question.

Does adding the behavioral indicators (fear / greed / herd) to a quant-only
model actually improve prediction *and* trading performance? We answer it the
fair way:

  * the two feature sets are scored on the **same rows, same folds, same models**;
    the only thing that changes is whether fear/greed/herd are in X;
  * splits are time-aware (walk-forward), never random;
  * we report classification metrics *and* financial metrics, then a paired test
    across folds to say whether any gap is real.

The heavy lifting (data, features, behavioral) is Member A's; the backtest engine
is Member B's lightweight long/flat proxy here (Member C's `src/backtest/` is the
realistic one). This module just orchestrates and tabulates.
"""
from __future__ import annotations

import re
from typing import Any

import pandas as pd

from src.behavioral import build_behavioral_features
from src.data.cleaner import clean_ohlcv
from src.data.labels import make_direction_label
from src.data.loader import load_ohlcv
from src.data.splitter import make_splits
from src.evaluation.classification import classification_metrics
from src.evaluation.financial import equity_curve, financial_metrics, strategy_returns
from src.evaluation.statistical import paired_significance
from src.features import build_features
from src.models import build_model
from src.utils import get_logger, set_seed

logger = get_logger(__name__)

BEHAVIORAL_COLS = ["fear", "greed", "herd"]
# Columns that are never model inputs: raw OHLCV (non-stationary levels), the
# regime label (used to *slice* results, not predict), and the targets.
NON_FEATURE_COLS = {"Open", "High", "Low", "Close", "Volume", "regime", "target", "fwd_return"}
# Absolute moving-average levels are non-stationary; their stationary cousin
# `close_to_sma_*` already carries the signal, so we drop the raw levels.
_RAW_LEVEL_RE = re.compile(r"^(sma|ema)_\d+$")

FEATURE_SETS = ("quant_only", "quant_plus_behavioral")


def feature_columns(df: pd.DataFrame, feature_set: str) -> list[str]:
    """Pick the model-input columns for a feature set, in stable order.

    `quant_only` is every stationary quantitative feature; `quant_plus_behavioral`
    appends whichever of fear/greed/herd are present. Raw price levels, the regime
    label and the targets are always excluded.
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"feature_set must be one of {FEATURE_SETS}, got '{feature_set}'")

    quant = [
        c
        for c in df.columns
        if c not in NON_FEATURE_COLS
        and c not in BEHAVIORAL_COLS
        and not _RAW_LEVEL_RE.match(c)
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    if feature_set == "quant_only":
        return quant
    return quant + [c for c in BEHAVIORAL_COLS if c in df.columns]


def build_model_frame(raw_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Turn one ticker's raw OHLCV into a model-ready frame.

    Runs Member A's pipeline (clean -> quant features -> behavioral features ->
    label) and attaches a forward-return column for the financial leg. Rows are
    dropped where *any* candidate feature, the label, or the forward return is
    missing — so both feature sets later evaluate on an identical row set.
    """
    df = clean_ohlcv(raw_df)
    horizon = config["labels"]["horizon"]
    threshold = config["labels"]["threshold"]

    out = build_features(df, config)
    out = build_behavioral_features(out, config, normalize=True)
    out["target"] = make_direction_label(df, horizon, threshold)
    # Realised return earned over the period the prediction is about (no leak:
    # this is the same future window the label encodes).
    out["fwd_return"] = df["Close"].shift(-horizon) / df["Close"] - 1.0

    required = feature_columns(out, "quant_plus_behavioral") + ["target", "fwd_return"]
    return out.dropna(subset=required)


def _evaluate_splits(
    name: str,
    feature_set: str,
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> list[dict]:
    """Train/score one (model, feature_set) across every walk-forward fold."""
    cols = feature_columns(frame, feature_set)
    X, y, fwd = frame[cols], frame["target"], frame["fwd_return"]

    bt = config.get("backtest", {})
    cost = bt.get("transaction_cost", 0.0005)
    allow_short = bt.get("allow_short", False)
    capital = bt.get("initial_capital", 100_000.0)
    horizon = max(1, config["labels"]["horizon"])
    periods_per_year = max(1, round(252 / horizon))

    rows: list[dict] = []
    for fold, (train_idx, test_idx) in enumerate(make_splits(frame, config)):
        y_train = y.loc[train_idx]
        if y_train.nunique() < 2:
            logger.warning("Skipping fold %d for %s/%s: single-class train split", fold, name, feature_set)
            continue

        model = build_model(name, config).fit(X.loc[train_idx], y_train)
        y_test = y.loc[test_idx]
        pred = model.predict(X.loc[test_idx])
        proba = model.predict_proba(X.loc[test_idx])

        cls = classification_metrics(y_test, pred, proba)
        rets = strategy_returns(pred, fwd.loc[test_idx], cost, allow_short)
        fin = financial_metrics(equity_curve(rets, capital), periods_per_year)

        rows.append({"model": name, "feature_set": feature_set, "fold": fold, "n_test": len(test_idx), **cls, **fin})
    return rows


def run_ablation(
    config: dict[str, Any],
    raw_data: dict[str, pd.DataFrame] | None = None,
) -> dict[str, pd.DataFrame]:
    """Run the with/without-behavioral comparison across tickers and models.

    Pass `raw_data={ticker: ohlcv_df}` to run fully offline (used by tests and
    notebooks that already hold data); otherwise tickers are loaded/cached via
    `src.data.loader`. Returns three tables:

      - ``per_fold``: one row per (ticker, model, feature_set, fold)
      - ``summary``: per-fold metrics averaged within (ticker, model, feature_set)
      - ``significance``: paired test of quant+behavioral vs quant-only per
        (ticker, model), for accuracy and Sharpe
    """
    set_seed(config.get("seed", 42))
    tickers = config["data"]["tickers"]
    models = config["models"].get("active", ["logistic", "random_forest", "xgboost"])

    per_fold: list[dict] = []
    for ticker in tickers:
        raw = raw_data[ticker] if raw_data is not None else load_ohlcv(
            ticker, config["data"]["start_date"], config["data"]["end_date"], config["data"]["raw_dir"]
        )
        frame = build_model_frame(raw, config)
        logger.info("Ablation on %s: %d rows after warmup/label trimming", ticker, len(frame))

        for name in models:
            for feature_set in FEATURE_SETS:
                for row in _evaluate_splits(name, feature_set, frame, config):
                    per_fold.append({"ticker": ticker, **row})

    per_fold_df = pd.DataFrame(per_fold)
    return {
        "per_fold": per_fold_df,
        "summary": _summarise(per_fold_df),
        "significance": _significance(per_fold_df),
    }


def _summarise(per_fold: pd.DataFrame) -> pd.DataFrame:
    """Average the per-fold metrics within each (ticker, model, feature_set)."""
    if per_fold.empty:
        return per_fold
    metric_cols = [c for c in per_fold.columns if c not in {"ticker", "model", "feature_set", "fold", "n_test"}]
    return (
        per_fold.groupby(["ticker", "model", "feature_set"])[metric_cols]
        .mean()
        .reset_index()
    )


def _significance(per_fold: pd.DataFrame, metrics=("accuracy", "sharpe")) -> pd.DataFrame:
    """Paired test (quant+behavioral vs quant-only) per (ticker, model)."""
    if per_fold.empty:
        return per_fold

    out: list[dict] = []
    for (ticker, model), grp in per_fold.groupby(["ticker", "model"]):
        base = grp[grp["feature_set"] == "quant_only"].sort_values("fold")
        cand = grp[grp["feature_set"] == "quant_plus_behavioral"].sort_values("fold")
        if base.empty or cand.empty or len(base) != len(cand):
            continue
        for metric in metrics:
            res = paired_significance(base[metric].to_numpy(), cand[metric].to_numpy())
            out.append({"ticker": ticker, "model": model, "metric": metric, **res})
    return pd.DataFrame(out)
