# src/portfolios.py
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def min_var_long_only(cov: pd.DataFrame) -> pd.Series:
    """
    Long-only minimum variance portfolio (robust):
        min_w w' Σ w
        s.t. sum(w)=1, w>=0

    Robust handling:
    - drops assets with NaN cov rows/cols
    - drops assets with non-positive/invalid variance
    - adds tiny ridge on diagonal for numerical stability
    - falls back to equal-weight if optimizer fails
    """
    cov = cov.copy()

    # 1) Drop assets with any NaN in covariance row/col
    ok = cov.notna().all(axis=0) & cov.notna().all(axis=1)
    cov = cov.loc[ok, ok]

    # 2) Drop assets with invalid variance
    if cov.shape[0] == 0:
        raise RuntimeError("MinVar: covariance empty after NaN cleaning.")

    var = np.diag(cov.values)
    ok_var = np.isfinite(var) & (var > 0)
    cov = cov.loc[cov.index[ok_var], cov.columns[ok_var]]

    assets = cov.index
    n = len(assets)

    if n == 0:
        raise RuntimeError("MinVar: no valid assets after variance cleaning.")
    if n == 1:
        return pd.Series([1.0], index=assets, name="weight")

    # 3) Ridge for numerical stability
    avg_var = float(np.mean(np.diag(cov.values)))
    ridge = 1e-8 * avg_var if np.isfinite(avg_var) and avg_var > 0 else 1e-8
    Sigma = cov.values + ridge * np.eye(n)

    # initial guess: equal weight
    x0 = np.ones(n) / n

    def obj(w):
        return float(w @ Sigma @ w)

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n

    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons)

    # 4) Fallback if optimizer fails
    if (not res.success) or (not np.isfinite(res.fun)):
        w = np.ones(n) / n
        return pd.Series(w, index=assets, name="weight")

    return pd.Series(res.x, index=assets, name="weight")


def value_weighted_weights(mv_row: pd.Series) -> pd.Series:
    """
    Value-weighted weights for a given month t:
        w_i,t = MV_i,t / sum_j MV_j,t
    mv_row: Series indexed by assets, values = market values at time t
    """
    mv = mv_row.copy()
    mv = pd.to_numeric(mv, errors="coerce")
    mv = mv.dropna()

    total = mv.sum()
    if total <= 0 or np.isnan(total) or len(mv) == 0:
        # fallback: equal weight if something weird happens
        n = len(mv)
        if n == 0:
            return pd.Series(dtype=float, name="weight")
        return pd.Series(np.ones(n) / n, index=mv.index, name="weight")

    w = mv / total
    w.name = "weight"
    return w
