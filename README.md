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
# 1. create + activate a virtual environment
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

# 2. install dependencies
pip install -r requirements.txt

# 3. run the full pipeline (config-driven)
python main.py --config config.yaml

# 4. (optional) launch the dashboard
streamlit run app/streamlit_app.py
```

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

## Workflow

- Branch per feature: `git checkout -b feature/<area>`
- Open a PR into `main`; don't push directly to `main`.
- Keep raw data out of git (it's gitignored) — only code is versioned.
