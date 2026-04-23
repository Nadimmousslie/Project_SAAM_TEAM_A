# =============================================================
# carbon_portfolio.py — Sections 3.2, 3.3, 4.1
# Carbon-constrained portfolio optimization (optimized)
# =============================================================

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from config import REBALANCE_YEARS
from carbon import compute_cf


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def _get_dec(df: pd.DataFrame, year: int):
    """Return December column LABEL (Timestamp) for a given year, or None."""
    cols = pd.DatetimeIndex(df.columns)
    dec  = cols[(cols.year == year) & (cols.month == 12)]
    return dec[-1] if len(dec) > 0 else None


def _cf_constraint_value(w, co2_arr, cap_arr):
    """Pure numpy CF — fast inside optimizer."""
    return float(np.dot(w, co2_arr / cap_arr))


def _prepare_cf_arrays(kept, co2_series, cap_series_m):
    """Pre-compute CF arrays once per year (no pandas in optimizer loop)."""
    co2_r = co2_series.reindex(kept).values
    cap_r = cap_series_m.reindex(kept).values
    valid = np.isfinite(co2_r) & np.isfinite(cap_r) & (cap_r > 0)
    co2_arr = np.where(valid, co2_r, 0.0)
    cap_arr = np.where(valid, cap_r, 1e12)
    return co2_arr, cap_arr


def _vw_weights_year(invest_sets, mv_y, year):
    """VW weights using yearly market cap."""
    isins_Y = invest_sets.get(year, [])
    col = _get_dec(mv_y, year)
    if col is None:
        return pd.Series(dtype=float)
    caps = mv_y[col].reindex(isins_Y).dropna()
    if caps.sum() == 0:
        return pd.Series(dtype=float)
    return caps / caps.sum()


# ─────────────────────────────────────────────────────────────
# PRE-COMPUTE ALL COVARIANCE MATRICES ONCE
# Uses 1/τ denominator as per consignes (not 1/(T-1))
# ─────────────────────────────────────────────────────────────

def precompute_sigmas(invest_sets: dict, ret_windows: dict) -> dict:
    """
    Compute covariance matrices once for all years.
    Uses 1/(T-1) unbiased estimator — same as optimization.py.
    Returns {year → (Sigma_np, kept_isins)}
    """
    print("  Pre-computing covariance matrices...", end=" ", flush=True)
    sigmas = {}
    for Y in REBALANCE_YEARS:
        if Y not in invest_sets:
            continue
        isins_Y    = invest_sets[Y]
        ret_window = ret_windows[Y].loc[isins_Y]

        # Same method as optimization.py — pandas .cov() with 1/(T-1)
        sigma_df = ret_window.T.cov(min_periods=36)
        valid    = sigma_df.notna().all(axis=1)
        sigma_df = sigma_df.loc[valid, valid]

        if sigma_df.shape[0] < 2:
            continue

        sigmas[Y] = (sigma_df.values, sigma_df.index.tolist())

    print(f"done ({len(sigmas)} years).")
    return sigmas


# ─────────────────────────────────────────────────────────────
# SECTION 3.2 — MV with CF constraint  P_oos^(mv)(0.5)
# ─────────────────────────────────────────────────────────────

def run_mv_carbon(invest_sets, ret_windows, weights_mv,
                  co2, mv_y, sigmas, reduction=0.5):
    """Min-variance with CF ≤ reduction × CF(P_mv)."""
    print("=" * 60)
    print(f"MV WITH CARBON CONSTRAINT  (CF ≤ {reduction:.0%} × CF_mv)")
    print("=" * 60)

    weights_mvc = {}

    for Y in REBALANCE_YEARS:
        if Y not in invest_sets or Y not in weights_mv or Y not in sigmas:
            continue

        Sigma, kept = sigmas[Y]
        n = len(kept)

        cf_mv = compute_cf(weights_mv[Y], co2, mv_y, Y)
        if np.isnan(cf_mv) or cf_mv <= 0:
            weights_mvc[Y] = weights_mv[Y]
            print(f"  {Y}: CF_mv unavailable — using unconstrained MV.")
            continue

        cf_target = reduction * cf_mv

        co2_col = _get_dec(co2, Y)
        cap_col = _get_dec(mv_y, Y)
        if co2_col is None or cap_col is None:
            weights_mvc[Y] = weights_mv[Y]
            continue

        co2_arr, cap_arr = _prepare_cf_arrays(
            kept, co2[co2_col], mv_y[cap_col] / 1000)

        # Warm start from MV weights
        w0 = weights_mv[Y].reindex(kept).fillna(0).values
        w0 = w0 / w0.sum() if w0.sum() > 0 else np.ones(n) / n

        result = minimize(
            fun         = lambda w, S=Sigma: w @ S @ w,
            x0          = w0,
            method      = "SLSQP",
            bounds      = [(0.0, 1.0)] * n,
            constraints = [
                {"type": "eq",   "fun": lambda w: w.sum() - 1},
                {"type": "ineq", "fun": lambda w, t=cf_target, c=co2_arr, p=cap_arr:
                 t - _cf_constraint_value(w, c, p)},
            ],
            options = {"ftol": 1e-9, "maxiter": 500},
        )

        w = result.x if result.success else w0
        weights_mvc[Y] = pd.Series(w, index=kept)

        cf_ach = _cf_constraint_value(w, co2_arr, cap_arr)
        vol    = np.sqrt(w @ Sigma @ w * 12) * 100
        print(f"  {Y}: {n:>4} firms | CF_mv={cf_mv:.1f} | "
              f"target={cf_target:.1f} | achieved={cf_ach:.1f} | vol={vol:.2f}%")

    print(f"  ✓ done for {len(weights_mvc)} years.\n")
    return weights_mvc


# ─────────────────────────────────────────────────────────────
# SECTION 3.3 — Tracking Error min with CF constraint
# ─────────────────────────────────────────────────────────────

def run_te_carbon(invest_sets, ret_windows, mv_y_bench,
                  co2, mv_y, sigmas, reduction=0.5, label="vw"):
    """Tracking-error min with CF ≤ reduction × CF(P_vw)."""
    print("=" * 60)
    print(f"TRACKING-ERROR MIN WITH CARBON CONSTRAINT  [{label}]")
    print("=" * 60)

    weights_te = {}

    for Y in REBALANCE_YEARS:
        if Y not in invest_sets or Y not in sigmas:
            continue

        Sigma, kept = sigmas[Y]
        n = len(kept)

        # VW benchmark weights
        w_vw_full = _vw_weights_year(invest_sets, mv_y_bench, Y)
        w_vw      = w_vw_full.reindex(kept).fillna(0)
        w_vw      = w_vw / w_vw.sum() if w_vw.sum() > 0 else w_vw
        w_vw_arr  = w_vw.values

        cf_vw = compute_cf(w_vw, co2, mv_y, Y)
        if np.isnan(cf_vw) or cf_vw <= 0:
            weights_te[Y] = pd.Series(w_vw_arr, index=kept)
            print(f"  {Y}: CF_vw unavailable — using VW.")
            continue

        red_Y     = reduction[Y] if isinstance(reduction, dict) else reduction
        cf_target = red_Y * cf_vw

        co2_col = _get_dec(co2, Y)
        cap_col = _get_dec(mv_y, Y)
        if co2_col is None or cap_col is None:
            weights_te[Y] = pd.Series(w_vw_arr, index=kept)
            continue

        co2_arr, cap_arr = _prepare_cf_arrays(
            kept, co2[co2_col], mv_y[cap_col] / 1000)

        result = minimize(
            fun         = lambda w, S=Sigma, b=w_vw_arr: (w-b) @ S @ (w-b),
            x0          = w_vw_arr.copy(),
            method      = "SLSQP",
            bounds      = [(0.0, 1.0)] * n,
            constraints = [
                {"type": "eq",   "fun": lambda w: w.sum() - 1},
                {"type": "ineq", "fun": lambda w, t=cf_target, c=co2_arr, p=cap_arr:
                 t - _cf_constraint_value(w, c, p)},
            ],
            options = {"ftol": 1e-9, "maxiter": 500},
        )

        w = result.x if result.success else w_vw_arr
        weights_te[Y] = pd.Series(w, index=kept)

        cf_ach = _cf_constraint_value(w, co2_arr, cap_arr)

        # --- MINIMAL FIX: clamp numerical negatives before sqrt ---
        quad = float((w - w_vw_arr) @ Sigma @ (w - w_vw_arr))
        te   = np.sqrt(max(quad, 0.0) * 12) * 100

        print(f"  {Y}: CF_vw={cf_vw:.1f} | target={cf_target:.1f} | "
              f"achieved={cf_ach:.1f} | TE={te:.2f}%")

    print(f"  ✓ done for {len(weights_te)} years.\n")
    return weights_te


# ─────────────────────────────────────────────────────────────
# SECTION 4.1 — Net Zero  P_oos^(vw)(NZ)
# FIXED: exponent is (Y - Y0 + 1) as per consignes
# ─────────────────────────────────────────────────────────────

def run_net_zero(invest_sets, ret_windows, mv_y_bench,
                 co2, mv_y, sigmas, theta=0.10, Y0=2013):
    """
    Tracking-error min with annually tightening CF constraint.
    Per consignes (Section 4.1):
        CF_Y ≤ (1 - θ)^(Y - Y0 + 1) × CF_vw(Y0)
    Note: exponent is (Y - Y0 + 1), NOT (Y - Y0).
    For Y=2013: target = (0.9)^1 × CF_vw(2013)  → 10% reduction from year 1
    For Y=2024: target = (0.9)^12 × CF_vw(2013) → ~71% cumulative reduction
    """
    print("=" * 60)
    print(f"NET ZERO PORTFOLIO  (θ={theta:.0%}/yr, base={Y0})")
    print("=" * 60)

    # CF_vw at Y0
    col_Y0   = _get_dec(mv_y_bench, Y0)
    isins_Y0 = invest_sets.get(Y0, [])
    if col_Y0 is not None and len(isins_Y0) > 0:
        caps_Y0  = mv_y_bench[col_Y0].reindex(isins_Y0).dropna()
        w_vw_Y0  = caps_Y0 / caps_Y0.sum() if caps_Y0.sum() > 0 else pd.Series()
        cf_vw_Y0 = compute_cf(w_vw_Y0, co2, mv_y, Y0)
    else:
        cf_vw_Y0 = np.nan

    if np.isnan(cf_vw_Y0):
        print("  WARNING: CF_vw(Y0) unavailable — fallback 50%.")
        return run_te_carbon(invest_sets, ret_windows, mv_y_bench,
                             co2, mv_y, sigmas, reduction=0.5, label="NZ-fallback")

    print(f"  CF_vw({Y0}) = {cf_vw_Y0:.4f}  (reference level)")
    print(f"  Targets: Y=2013 → {(1-theta)**1 * cf_vw_Y0:.4f} | "
          f"Y=2024 → {(1-theta)**12 * cf_vw_Y0:.4f}")

    weights_nz = {}

    for Y in REBALANCE_YEARS:
        if Y not in invest_sets or Y not in sigmas:
            continue

        Sigma, kept = sigmas[Y]
        n = len(kept)

        # VW benchmark weights
        col_Y  = _get_dec(mv_y_bench, Y)
        caps_Y = mv_y_bench[col_Y].reindex(kept).dropna() \
                 if col_Y is not None else pd.Series()
        if caps_Y.sum() > 0:
            w_vw = (caps_Y / caps_Y.sum()).reindex(kept).fillna(0)
        else:
            w_vw = pd.Series(np.ones(n) / n, index=kept)
        w_vw_arr = w_vw.values

        # ── FIXED: exponent = (Y - Y0 + 1) per consignes ──────
        cf_target = (1 - theta) ** (Y - Y0 + 1) * cf_vw_Y0

        co2_col = _get_dec(co2, Y)
        cap_col = _get_dec(mv_y, Y)
        if co2_col is None or cap_col is None:
            weights_nz[Y] = w_vw
            continue

        co2_arr, cap_arr = _prepare_cf_arrays(
            kept, co2[co2_col], mv_y[cap_col] / 1000)

        result = minimize(
            fun         = lambda w, S=Sigma, b=w_vw_arr: (w-b) @ S @ (w-b),
            x0          = w_vw_arr.copy(),
            method      = "SLSQP",
            bounds      = [(0.0, 1.0)] * n,
            constraints = [
                {"type": "eq",   "fun": lambda w: w.sum() - 1},
                {"type": "ineq", "fun": lambda w, t=cf_target, c=co2_arr, p=cap_arr:
                 t - _cf_constraint_value(w, c, p)},
            ],
            options = {"ftol": 1e-9, "maxiter": 500},
        )

        w = result.x if result.success else w_vw_arr
        weights_nz[Y] = pd.Series(w, index=kept)

        cf_ach = _cf_constraint_value(w, co2_arr, cap_arr)

        # --- MINIMAL FIX: clamp numerical negatives before sqrt ---
        quad = float((w - w_vw_arr) @ Sigma @ (w - w_vw_arr))
        te   = np.sqrt(max(quad, 0.0) * 12) * 100

        print(f"  {Y}: target={cf_target:.4f} | achieved={cf_ach:.4f} | TE={te:.2f}%")

    print(f"  ✓ done for {len(weights_nz)} years.\n")
    return weights_nz
