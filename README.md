# Behaviour-Aware Trading System using Machine Learning

Predict stock price direction (up/down) by combining traditional **quantitative**
features with **behavioral-finance** proxies (fear, greed, herd behavior), and test
whether the behavioral signals actually improve prediction *and* trading performance
versus a quant-only baseline.

> The central research question is the **ablation**: model *with* behavioral features
> vs *without*. See [`src/experiments/ablation.py`](src/experiments/ablation.py).

**Team:** Adarsh, Lakshmi, Lucas

---

## Quick start

```bash
# 1. one-command setup: creates .venv and installs everything
python setup_env.py
#    Windows shortcut:    .\setup.ps1
#    macOS/Linux shortcut: source setup.sh

# 2. activate the environment
#    Windows:      .venv\Scripts\activate
#    macOS/Linux:  source .venv/bin/activate

# 3. run the full pipeline (config-driven) - available once modeling lands
python main.py --config config.yaml

# 4. (optional) launch the dashboard
streamlit run app/streamlit_app.py
```

To explore what's built so far, run the notebooks:
`notebooks/01_data_exploration.ipynb` and `notebooks/02_behavioral_validation.ipynb`.

## Project structure

```
src/
├── config.py / utils.py / pipeline.py   orchestration + helpers
├── data/        loader, cleaner, labels, splitter (time-series CV)
├── features/    technical indicators, rolling stats
├── behavioral/  fear, greed, herd, regime, validation
├── models/      logistic, random_forest, xgboost (shared base)
├── evaluation/  classification, financial, statistical
├── backtest/    engine, strategies
├── experiments/ ablation (core), regime_analysis
└── visualize/   plots, shap_analysis
```

## How the team can split work

| Area | Owner | Modules |
|------|-------|---------|
| Data + features | TBD | `src/data/`, `src/features/` |
| Behavioral signals | TBD | `src/behavioral/` |
| Models + evaluation | TBD | `src/models/`, `src/evaluation/` |
| Backtest + experiments | TBD | `src/backtest/`, `src/experiments/` |

> ⚠️ **Important:** Use **time-aware** splitting (`src/data/splitter.py`), never a
> random train/test split — a random split leaks the future into the past and makes
> the accuracy numbers meaningless for a trading model.

## Section A output — feature reference (for modeling)

Section A turns OHLCV into a model-ready DataFrame indexed by date. After
`build_features` + `build_behavioral_features` + `make_direction_label`, then
`dropna()`, these columns are available (with the default `config.yaml`):

| Group | Columns |
|-------|---------|
| Raw | `Open, High, Low, Close, Volume` |
| Returns | `return, log_return` |
| Moving avgs | `sma_10, sma_20, sma_50, close_to_sma_10/20/50, ema_12, ema_26` |
| Momentum/RSI | `momentum_10, rsi_14` |
| Volatility | `volatility_20` |
| Rolling stats | `roll_mean_5/20, roll_std_5/20, roll_skew_5/20, roll_kurt_5/20` |
| Behavioral | `fear, greed, herd` (expanding z-scored), `regime` (categorical) |
| Target | `target` (1 = up over the next `horizon` days) |

The **ablation** compares the quant columns alone vs quant **+** `fear/greed/herd`.
Exclude `regime` from model inputs (it's for slicing results, not a feature).

## Workflow

- Branch per feature: `git checkout -b feature/<area>`
- Open a PR into `main`; don't push directly to `main`.
- Keep raw data out of git (it's gitignored) — only code is versioned.
