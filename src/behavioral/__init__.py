"""Behavioural feature engineering.

`build_behavioral_features` computes the fear / greed / herd proxies and returns
them normalized. Normalization uses an expanding window (see
src.utils.expanding_zscore) so no future information leaks backwards.
"""
from src.behavioral.fear import compute_fear
from src.behavioral.greed import compute_greed
from src.behavioral.herd import compute_herd
from src.behavioral.regime import detect_regime
from src.utils import expanding_zscore


def build_behavioral_features(df, config, normalize=True):
    cfg = config["behavioral"]

    fear = compute_fear(df, **cfg.get("fear", {}))
    greed = compute_greed(df, **cfg.get("greed", {}))
    herd = compute_herd(df, **cfg.get("herd", {}))

    out = df.copy()
    if normalize:
        out["fear"] = expanding_zscore(fear)
        out["greed"] = expanding_zscore(greed)
        out["herd"] = expanding_zscore(herd)
    else:
        out["fear"], out["greed"], out["herd"] = fear, greed, herd

    out["regime"] = detect_regime(df)
    return out


__all__ = [
    "build_behavioral_features",
    "compute_fear",
    "compute_greed",
    "compute_herd",
    "detect_regime",
]
