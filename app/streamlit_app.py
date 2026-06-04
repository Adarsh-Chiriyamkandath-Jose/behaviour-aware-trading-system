"""Streamlit dashboard for the Behaviour-Aware Trading System.

Pick a ticker and model, decide whether to feed the model behavioural signals
(fear / greed / herd), and see — side by side — how well it calls direction and
whether acting on those calls would have beaten buy-and-hold out of sample.

Run with:  streamlit run app/streamlit_app.py
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.backtest import buy_and_hold, run_backtest, signal_from_predictions
from src.config import load_config
from src.data.loader import load_ohlcv
from src.data.splitter import single_split
from src.evaluation import classification_metrics, financial_metrics
from src.experiments import build_model_frame, feature_columns
from src.models import build_model
from src.visualize import plot_feature_importance
from src.visualize.plots import BEHAVIORAL, plot_confusion

# A clean, consistent look for the matplotlib figures we still embed.
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"axes.titlesize": 11, "axes.titleweight": "bold", "figure.autolayout": True})

ACCENT = "#2f9e6f"


@st.cache_data(show_spinner=False)
def _prepare(ticker: str, _config: dict) -> pd.DataFrame:
    raw = load_ohlcv(ticker, _config["data"]["start_date"], _config["data"]["end_date"], _config["data"]["raw_dir"])
    return build_model_frame(raw, _config)


@st.cache_data(show_spinner=False)
def _evaluate(ticker: str, model_name: str, feature_set: str, _config: dict) -> dict:
    """Train on the in-sample split, score out of sample, and package everything
    the dashboard needs into one picklable dict (so the whole run is cached)."""
    frame = _prepare(ticker, _config)
    cols = feature_columns(frame, feature_set)

    train_idx, test_idx = single_split(frame, _config["split"].get("test_size", 0.2))
    model = build_model(model_name, _config).fit(frame.loc[train_idx, cols], frame.loc[train_idx, "target"])

    y_true = frame.loc[test_idx, "target"]
    pred = model.predict(frame.loc[test_idx, cols])
    proba = model.predict_proba(frame.loc[test_idx, cols])

    prices = frame.loc[test_idx, "Close"]
    signals = signal_from_predictions(pred, index=test_idx, allow_short=_config["backtest"].get("allow_short", False))
    bt = run_backtest(prices, signals, _config["backtest"]["initial_capital"], _config["backtest"]["transaction_cost"])
    bench = buy_and_hold(prices, _config["backtest"]["initial_capital"])

    importances = model.feature_importances()

    return {
        "n_features": len(cols),
        "n_test": len(test_idx),
        "train_span": (str(frame.loc[train_idx].index[0].date()), str(frame.loc[train_idx].index[-1].date())),
        "test_span": (str(test_idx[0].date()), str(test_idx[-1].date())),
        "y_true": y_true,
        "pred": pred,
        "proba": proba,
        "equity": bt["equity"],
        "benchmark": bench,
        "drawdown": bt["drawdown"],
        "n_trades": int((bt["turnover"] > 0).sum()),
        "cls": classification_metrics(y_true, pred, proba),
        "fin": financial_metrics(bt["equity"]),
        "bench_fin": financial_metrics(bench),
        "importances": importances,
    }


def _kpi_row(res: dict) -> None:
    cls, fin, bench = res["cls"], res["fin"], res["bench_fin"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{cls['accuracy']:.1%}", f"{cls['accuracy'] - 0.5:+.1%} vs coin flip")
    c2.metric("Sharpe", f"{fin['sharpe']:.2f}")
    c3.metric(
        "Total return",
        f"{fin['total_return']:.1%}",
        f"{fin['total_return'] - bench['total_return']:+.1%} vs buy & hold",
    )
    c4.metric("Max drawdown", f"{fin['max_drawdown']:.1%}", f"{fin['max_drawdown'] - bench['max_drawdown']:+.1%}", delta_color="inverse")
    c5.metric("Win rate", f"{fin['win_rate']:.1%}")


def _performance_tab(res: dict, config: dict) -> None:
    cap = config["backtest"]["initial_capital"]
    equity = pd.DataFrame({"Strategy": res["equity"], "Buy & hold": res["benchmark"]})
    st.markdown("**Portfolio value — strategy vs buy & hold**")
    st.line_chart(equity, height=320, color=[ACCENT, "#9aa5b1"])

    st.markdown("**Drawdown** — how far below the prior peak the strategy sat")
    st.area_chart(res["drawdown"], height=180, color="#d64545")

    fin, bench = res["fin"], res["bench_fin"]
    table = pd.DataFrame(
        {
            "Strategy": [fin["total_return"], fin["annual_return"], fin["annual_volatility"], fin["sharpe"], fin["max_drawdown"]],
            "Buy & hold": [bench["total_return"], bench["annual_return"], bench["annual_volatility"], bench["sharpe"], bench["max_drawdown"]],
        },
        index=["Total return", "Annual return", "Annual volatility", "Sharpe", "Max drawdown"],
    )
    fmt = table.astype(object)
    for row in ["Total return", "Annual return", "Annual volatility", "Max drawdown"]:
        fmt.loc[row] = table.loc[row].map(lambda v: f"{v:.1%}")
    fmt.loc["Sharpe"] = table.loc["Sharpe"].map(lambda v: f"{v:.2f}")
    st.dataframe(fmt, use_container_width=True)
    st.caption(f"Started from ${cap:,.0f} of capital · {res['n_trades']} position changes over the test window.")


def _quality_tab(res: dict) -> None:
    cls = res["cls"]
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("**Direction call quality**")
        q = pd.DataFrame(
            {"Score": [cls["accuracy"], cls["precision"], cls["recall"], cls["f1"], cls.get("roc_auc", float("nan"))]},
            index=["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
        )
        st.dataframe(q.style.format({"Score": "{:.3f}"}), use_container_width=True)
        st.caption("Precision/recall are for the **up** class (label 1).")
    with c2:
        st.pyplot(plot_confusion(res["y_true"], res["pred"]))


def _drivers_tab(res: dict, feature_set: str) -> None:
    imp = res["importances"]
    if imp is None:
        st.info("This model doesn't expose feature importances.")
        return

    if feature_set == "quant_plus_behavioral":
        behav_share = imp[imp.index.isin(BEHAVIORAL)].sum() / imp.sum() if imp.sum() else 0.0
        st.metric("Behavioural share of importance", f"{behav_share:.1%}",
                  help="Combined weight the model put on fear / greed / herd — the question this whole project asks.")
    st.pyplot(plot_feature_importance(imp, top_n=12))


def main() -> None:
    st.set_page_config(page_title="Behaviour-Aware Trading System", page_icon="📈", layout="wide")
    config = load_config(os.path.dirname(__file__) + "/../config.yaml")

    # ---- Sidebar: controls -------------------------------------------------
    sb = st.sidebar
    sb.title("Controls")
    sb.caption("Configure a run, then read the results on the right.")

    sb.subheader("Data")
    ticker = sb.selectbox("Ticker", config["data"]["tickers"], help="Symbol to train and backtest on.")

    sb.subheader("Model")
    model_name = sb.selectbox("Algorithm", config["models"]["active"],
                              index=len(config["models"]["active"]) - 1)
    use_behavioral = sb.toggle("Include behavioural features", value=True,
                               help="Feed the model the fear / greed / herd signals on top of the quant features.")
    feature_set = "quant_plus_behavioral" if use_behavioral else "quant_only"

    with sb.expander("About this dashboard"):
        st.write(
            "Trains the selected model on the early part of the history and scores it "
            "out of sample. Toggle the behavioural features to run the core ablation: "
            "do fear / greed / herd signals actually help?"
        )

    # ---- Header ------------------------------------------------------------
    st.title("Behaviour-Aware Trading System")
    st.caption("Can crowd-behaviour signals — fear, greed, herding — improve a machine-learned stock-direction model?")

    with st.spinner(f"Training {model_name} on {ticker}…"):
        res = _evaluate(ticker, model_name, feature_set, config)

    badge = "Quant + behavioural" if use_behavioral else "Quant only"
    st.markdown(
        f"**{ticker}** · **{model_name}** · feature set: `{feature_set}` "
        f"({res['n_features']} features) · test window {res['test_span'][0]} → {res['test_span'][1]}"
    )
    st.divider()

    _kpi_row(res)
    st.write("")

    tab_perf, tab_quality, tab_drivers = st.tabs(["Performance", "Call quality", "What drove it"])
    with tab_perf:
        _performance_tab(res, config)
    with tab_quality:
        _quality_tab(res)
    with tab_drivers:
        _drivers_tab(res, feature_set)

    st.divider()
    st.caption(
        f"Trained on {res['train_span'][0]} → {res['train_span'][1]} · "
        f"evaluated on {res['n_test']} out-of-sample rows. "
        "Past performance is not indicative of future results — research tool, not investment advice."
    )


if __name__ == "__main__":
    main()
