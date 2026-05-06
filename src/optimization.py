# optimization.py — Minimum-variance optimization (Section 2.2)

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from config import REBALANCE_YEARS, MIN_MONTHS_DATA


def _covariance_matrix(ret: pd.DataFrame) -> pd.DataFrame:
    """
    Sample covariance matrix with 1/τ denominator (as in the project statement):
        Σ_Y = (1/τ) Σ_{k=0}^{τ-1} (R_{t-k} - μ̂_Y)'(R_{t-k} - μ̂_Y)
    Assets with fewer than MIN_MONTHS_DATA valid observations are dropped.
    """
    sigma = ret.T.cov(min_periods=MIN_MONTHS_DATA, ddof=0)
    valid = sigma.notna().all(axis=1)
    sigma = sigma.loc[valid, valid]
    return sigma


def _min_variance_weights(cov: np.ndarray) -> np.ndarray:
    """
    Long-only minimum-variance:
        min  w'Σw   s.t.  Σw_i = 1,  w_i ≥ 0
    """
    n  = cov.shape[0]
    w0 = np.ones(n) / n

    result = minimize(
        fun         = lambda w, S=cov: w @ S @ w,
        x0          = w0,
        method      = "SLSQP",
        bounds      = [(0.0, 1.0)] * n,
        constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}],
        options     = {"ftol": 1e-9, "maxiter": 500},
    )
    if not result.success:
        print(f"    ⚠ WARNING: MV solver did not converge — {result.message}")
    return result.x if result.success else w0


def run_optimization(invest_sets: dict, ret_windows: dict) -> dict:
    """
    For each year Y, estimate Σ_Y (1/τ denominator) and solve the min-variance problem.
    Rebalances annually from 2013 to 2024.

    Returns
    -------
    weights_dict : {year → pd.Series(weights, index=ISINs)}
    """
    print("=" * 55)
    print("MINIMUM-VARIANCE OPTIMIZATION")
    print("=" * 55)

    weights_dict = {}

    for Y in REBALANCE_YEARS:
        if Y not in invest_sets:
            continue

        isins_Y = invest_sets[Y]
        if len(isins_Y) < 2:
            print(f"  {Y}: only {len(isins_Y)} firm(s) — skipped.")
            continue

        ret_window = ret_windows[Y].loc[isins_Y]

        # Covariance matrix (1/τ denominator)
        sigma_df = _covariance_matrix(ret_window)
        if sigma_df.shape[0] < 2:
            print(f"  {Y}: insufficient data — skipped.")
            continue

        kept  = sigma_df.index.tolist()
        Sigma = sigma_df.values

        # Optimize
        w = _min_variance_weights(Sigma)
        weights_dict[Y] = pd.Series(w, index=kept)

        port_var = w @ Sigma @ w
        print(f"  {Y}: {len(kept):>4} firms | "
              f"port. variance = {port_var:.6f} | "
              f"ann. vol = {np.sqrt(port_var * 12) * 100:.2f}%")

    print(f"  ✓ Optimization done for {len(weights_dict)} years.\n")
    return weights_dict
