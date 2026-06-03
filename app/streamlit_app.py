"""Streamlit dashboard for the Behaviour-Aware Trading System.

Pick a ticker, train the behaviour-aware model on its history, and see the
out-of-sample equity curve, classification quality, and which features (including
fear/greed/herd) drove the calls.

Run with:  streamlit run app/streamlit_app.py
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

import streamlit as st

from src.backtest import buy_and_hold, run_backtest, signal_from_predictions
from src.config import load_config
from src.data.loader import load_ohlcv
from src.data.splitter import single_split
from src.evaluation import classification_metrics, financial_metrics
from src.experiments import build_model_frame, feature_columns
from src.models import build_model
from src.visualize import plot_equity_curve, plot_feature_importance


@st.cache_data(show_spinner=False)
def _prepare(ticker: str, _config: dict):
    raw = load_ohlcv(ticker, _config["data"]["start_date"], _config["data"]["end_date"], _config["data"]["raw_dir"])
    return build_model_frame(raw, _config)


def main() -> None:
    st.set_page_config(page_title="Behaviour-Aware Trading System", layout="wide")
    st.title("Behaviour-Aware Trading System")

    config = load_config(os.path.dirname(__file__) + "/../config.yaml")

    col = st.sidebar
    ticker = col.selectbox("Ticker", config["data"]["tickers"])
    model_name = col.selectbox("Model", config["models"]["active"], index=len(config["models"]["active"]) - 1)
    use_behavioral = col.checkbox("Include behavioral features (fear/greed/herd)", value=True)

    frame = _prepare(ticker, config)
    feature_set = "quant_plus_behavioral" if use_behavioral else "quant_only"
    cols = feature_columns(frame, feature_set)

    train_idx, test_idx = single_split(frame, config["split"].get("test_size", 0.2))
    model = build_model(model_name, config).fit(frame.loc[train_idx, cols], frame.loc[train_idx, "target"])
    pred = model.predict(frame.loc[test_idx, cols])

    prices = frame.loc[test_idx, "Close"]
    signals = signal_from_predictions(pred, index=test_idx, allow_short=config["backtest"].get("allow_short", False))
    bt = run_backtest(prices, signals, config["backtest"]["initial_capital"], config["backtest"]["transaction_cost"])

    cls = classification_metrics(frame.loc[test_idx, "target"], pred)
    fin = financial_metrics(bt["equity"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{cls['accuracy']:.1%}")
    c2.metric("Sharpe", f"{fin['sharpe']:.2f}")
    c3.metric("Total return", f"{fin['total_return']:.1%}")
    c4.metric("Max drawdown", f"{fin['max_drawdown']:.1%}")

    left, right = st.columns(2)
    left.subheader("Equity vs buy & hold")
    left.pyplot(plot_equity_curve(bt["equity"], buy_and_hold(prices, config["backtest"]["initial_capital"])))

    imp = model.feature_importances()
    if imp is not None:
        right.subheader("What drove the calls")
        right.pyplot(plot_feature_importance(imp, top_n=12))

    st.caption(f"Feature set: {feature_set} ({len(cols)} features) · out-of-sample test rows: {len(test_idx)}")


if __name__ == "__main__":
    main()
