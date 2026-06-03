# Results — Behaviour-Aware Trading System

Empirical findings from the full pipeline on **AAPL, MSFT, SPY** (daily OHLCV,
2015-01-01 → 2024-12-31), evaluated with 5-fold **walk-forward** splits and a
cost-aware backtest. Reproduce with `python main.py --config config.yaml`
(writes `results/tables/`) and `notebooks/04_final_results.ipynb` (figures).

## Headline

> **Behavioral indicators (fear / greed / herd) did not produce a statistically
> significant, consistent improvement over a quant-only baseline for predicting
> stock direction.** Next-day direction is predictable at ~51% (near chance);
> a weekly horizon is easier (~54%). The behavioral proxies *are* used by the
> models (greed ranks as the single most important feature for MSFT under SHAP),
> but that usage does not translate into a reliable out-of-sample edge —
> consistent with weak-form market efficiency.

A ~51% accuracy is the **expected, credible** outcome for daily equity direction,
not a failure: the literature and industry operate in the 51–53% range, and a
much higher number on this setup would more likely indicate look-ahead leakage
than skill. The contribution of this project is the **rigorous, leak-free test**
of the behavioral hypothesis — and the honest answer it returns.

## 1. Prediction accuracy (next-day direction, horizon = 1)

Mean accuracy across walk-forward folds, averaged over tickers:

| Model | quant_only | quant + behavioral | Δ (behavioral − quant) |
|-------|-----------:|-------------------:|----------------------:|
| logistic | 0.520 | 0.516 | −0.004 |
| random_forest | 0.503 | 0.507 | +0.004 |
| xgboost | 0.507 | 0.512 | +0.005 |

- **Overall mean accuracy: 0.511** (best single config 0.539 = MSFT / random_forest / +behavioral; worst 0.491).
- ROC-AUC sits at ~0.51 everywhere — essentially no rank ordering of up vs down days.
- Behavioral deltas are tiny and **flip sign across models** → no coherent effect.

## 2. Is the gap statistically real? No.

Paired tests (t-test + Wilcoxon) of quant+behavioral vs quant-only across folds,
18 comparisons (3 tickers × 3 models × {accuracy, sharpe}):

- **Only 1 of 18** reached p < 0.05 (MSFT / random_forest / accuracy, +2.6pp,
  t-p = 0.003) — but its **Wilcoxon p = 0.06** disagrees, and with 18 tests ~1
  false positive at α = 0.05 is expected by chance (0.05 × 18 ≈ 0.9).
- Conclusion: **no credible evidence** that behavioral features help at horizon 1.

## 3. Robustness — weekly horizon (horizon = 5)

Re-running the ablation predicting 5-day-ahead direction:

| Model | quant_only | quant + behavioral | Δ |
|-------|-----------:|-------------------:|----:|
| logistic | 0.557 | 0.555 | −0.002 |
| random_forest | 0.519 | 0.541 | +0.022 |
| xgboost | 0.529 | 0.541 | +0.012 |

- **Overall accuracy rises to 0.540** (vs 0.511 daily) — longer horizons are more
  predictable, as expected.
- Behavioral features lean slightly **positive for tree models** here (+1–2pp) but
  remain non-significant (min accuracy p = 0.048, again within multiple-comparison
  noise). The null result is therefore **robust across horizons**.

## 4. Trading performance (xgboost, quant + behavioral, out-of-sample split)

The directional strategy is profitable in absolute terms but **underperforms
simply holding the asset** — typical for a long/flat daily signal in a bull market:

| Ticker | Accuracy | Strategy return | Strategy Sharpe | Max DD | Buy & hold return | Buy & hold Sharpe |
|--------|--------:|---------------:|---------------:|------:|------------------:|------------------:|
| AAPL | 0.503 | +54% | 1.57 | −15% | +94% | 1.70 |
| MSFT | 0.519 | +50% | 1.34 | −13% | +84% | 1.50 |
| SPY  | 0.525 | +18% | 0.86 | −8%  | +54% | 1.81 |

Takeaway: accuracy ≠ profitability, but here neither beats the benchmark.

## 5. Interpretability (SHAP)

For MSFT (xgboost), mean |SHAP| ranks the behavioral features:

| Behavioral feature | Importance rank | mean \|SHAP\| |
|--------------------|----------------:|-------------:|
| greed | **1** (of all features) | 0.215 |
| herd  | 7 | 0.126 |
| fear  | 17 | 0.044 |

The model **does** rely on the greed proxy heavily — behavioral signals are not
ignored. The interesting tension: high feature usage **without** an out-of-sample
accuracy gain suggests the proxy captures structure the model fits to in-sample
but that does not generalize to a directional edge.

## Figures (`results/figures/`)

| File | Shows |
|------|-------|
| `ablation_accuracy.png`, `ablation_sharpe.png` | quant vs quant+behavioral per model |
| `<TICKER>_equity.png` | strategy equity vs buy-and-hold |
| `<TICKER>_drawdown.png` | underwater plot |
| `<TICKER>_confusion.png` | up/down confusion matrix |
| `<TICKER>_importance.png` | feature importances (behavioral highlighted) |
| `MSFT_shap_summary.png` | SHAP beeswarm |

## Bottom line for the report

1. **Prediction:** ~51% next-day / ~54% weekly accuracy — marginally above chance,
   methodologically sound (time-aware splits, no leakage).
2. **Hypothesis:** behavioral proxies derived from price/volume **do not** yield a
   statistically significant edge over technical features, robust across horizons.
3. **Insight:** models still weight behavioral signals (esp. greed) highly under
   SHAP — usage without generalizable edge.
4. **Framing:** an honest null result with correct methodology — the intended
   deliverable of the ablation, and a credible scientific finding.
