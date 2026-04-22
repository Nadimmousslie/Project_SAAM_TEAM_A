# =============================================================
# optimization.py — Section 2.2: Minimum-Variance Optimization
# Uses 1/τ covariance denominator as per project consignes
# ─────────────────────────────────────────────────────────────
# Consignes formula:
#   Σ_Y = (1/τ) Σ_{k=0}^{τ-1} (R_{t-k} - µ_Y)'(R_{t-k} - µ_Y)
# =============================================================

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from config import REBALANCE_YEARS, MIN_MONTHS_DATA


# ─────────────────────────────────────────────────────────────
# COVARIANCE ESTIMATOR — 1/τ denominator (per consignes)
# ─────────────────────────────────────────────────────────────

def _covariance_matrix(ret: pd.DataFrame) -> pd.DataFrame:
    """
    Sample covariance matrix with 1/τ denominator as per project consignes.
        Σ_Y = (1/τ) Σ (R - µ)'(R - µ)

    Assets with fewer than MIN_MONTHS_DATA valid observations are dropped.
    NaN values are filled with 0 before computation.
    """
    # Drop firms with too few observations
    n_valid = ret.notna().sum(axis=1)
    ret = ret[n_valid >= MIN_MONTHS_DATA]

    if ret.shape[0] < 2:
        return pd.DataFrame()

    X    = ret.fillna(0).values.T   # (T × N)
    T, N = X.shape

    # Demean
    mu = X.mean(axis=0)
    Xc = X - mu

    # Covariance with 1/τ denominator (biased — per consignes)
    S = (Xc.T @ Xc) / T

    # Drop firms with zero variance
    var_diag = np.diag(S)
    valid    = var_diag > 0
    if valid.sum() < 2:
        return pd.DataFrame()

    S_clean   = S[np.ix_(valid, valid)]
    kept      = [ret.index[i] for i in range(N) if valid[i]]

    return pd.DataFrame(S_clean, index=kept, columns=kept)


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
    For each year Y, estimate Σ_Y (1/τ) and solve the min-variance problem.
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

