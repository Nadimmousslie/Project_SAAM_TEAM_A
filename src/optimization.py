# =============================================================
# optimization.py — Section 2.2: Minimum-Variance Optimization
# Uses 1/(T-1) denominator (unbiased sample covariance)
# =============================================================

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from config import REBALANCE_YEARS, MIN_MONTHS_DATA


# ─────────────────────────────────────────────────────────────
# COVARIANCE ESTIMATOR — 1/(T-1) unbiased (pandas standard)
# ─────────────────────────────────────────────────────────────

def _covariance_matrix(ret: pd.DataFrame) -> pd.DataFrame:
    """
    Sample covariance matrix with 1/(T-1) denominator (unbiased).
    Assets with fewer than MIN_MONTHS_DATA valid observations are dropped.
    Uses pandas .cov() with min_periods for robustness.
    """
    sigma = ret.T.cov(min_periods=MIN_MONTHS_DATA)
    valid = sigma.notna().all(axis=1)
    sigma = sigma.loc[valid, valid]
    return sigma


# ─────────────────────────────────────────────────────────────
# OPTIMIZER
# ─────────────────────────────────────────────────────────────

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
    return result.x if result.success else w0


# ─────────────────────────────────────────────────────────────
# ROLLING OPTIMIZATION
# ─────────────────────────────────────────────────────────────

def run_optimization(invest_sets: dict, ret_windows: dict) -> dict:
    """
    For each year Y, estimate Σ_Y (1/(T-1)) and solve the min-variance problem.
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

        # Covariance matrix (1/(T-1) unbiased)
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

