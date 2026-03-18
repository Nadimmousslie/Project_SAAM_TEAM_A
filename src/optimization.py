# =============================================================
# optimization.py — Section 2.2: Minimum-Variance Optimization
# =============================================================

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from config import REBALANCE_YEARS, MIN_MONTHS_DATA


# ─────────────────────────────────────────────────────────────
# Core optimizer
# ─────────────────────────────────────────────────────────────

def _min_variance_weights(cov: np.ndarray) -> np.ndarray:
    """
    Solve the long-only minimum-variance problem:
        min  w' Σ w
        s.t. sum(w) = 1,  w_i >= 0  for all i

    Uses SLSQP. Falls back to equal weights if optimization fails.
    """
    n  = cov.shape[0]
    w0 = np.ones(n) / n

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds      = [(0.0, 1.0)] * n

    result = minimize(
        fun        = lambda w: w @ cov @ w,
        x0         = w0,
        method     = "SLSQP",
        bounds     = bounds,
        constraints= constraints,
        options    = {"ftol": 1e-12, "maxiter": 1000},
    )

    if result.success:
        return result.x
    else:
        # Fallback: equal weight (should be rare)
        return w0


def _covariance_matrix(ret_eligible: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate the covariance matrix from observed returns only.
    Assets with undefined pairwise covariances are dropped before optimization.
    """
    sigma = ret_eligible.T.cov(min_periods=MIN_MONTHS_DATA)
    valid_assets = sigma.notna().all(axis=1)
    sigma = sigma.loc[valid_assets, valid_assets]
    return sigma


# ─────────────────────────────────────────────────────────────
# Rolling optimization
# ─────────────────────────────────────────────────────────────

def run_optimization(invest_sets: dict, ret_windows: dict) -> dict:
    """
    For each year Y, estimate Σ_Y from the 10-year window and
    solve the minimum-variance problem over the eligible firms.

    Parameters
    ----------
    invest_sets : {year → list of eligible ISINs}
    ret_windows : {year → DataFrame (all firms × window months)}

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

        ret_window = ret_windows[Y]

        # Keep only eligible firms and estimate covariance from observed returns
        ret_eligible = ret_window.loc[isins_Y]

        # ── Covariance matrix Σ_Y ────────────────────────────
        Sigma_Y_df = _covariance_matrix(ret_eligible)
        kept_isins = Sigma_Y_df.index.tolist()

        if len(kept_isins) < 2:
            print(f"  {Y}: insufficient overlap after covariance cleaning — skipped.")
            continue

        Sigma_Y = Sigma_Y_df.values

        # ── Optimize ─────────────────────────────────────────
        w = _min_variance_weights(Sigma_Y)

        weights_dict[Y] = pd.Series(w, index=kept_isins)

        port_var = w @ Sigma_Y @ w
        print(f"  {Y}: {len(kept_isins):>4} firms | "
              f"port. variance = {port_var:.6f} | "
              f"ann. vol = {np.sqrt(port_var * 12) * 100:.2f}%")

    print(f"  ✓ Optimization done for {len(weights_dict)} years.\n")
    return weights_dict
