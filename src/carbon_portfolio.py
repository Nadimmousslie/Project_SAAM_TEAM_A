# =============================================================
# carbon_portfolio.py — Sections 3.2, 3.3, 4.1
# Carbon-constrained portfolio optimization
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
    cols = pd.DatetimeIndex(df.columns)
    dec  = cols[(cols.year == year) & (cols.month == 12)]
    return df[dec[-1]] if len(dec) > 0 else None


def _cf_constraint_value(w: np.ndarray,
                          co2_arr: np.ndarray,
                          cap_arr: np.ndarray) -> float:
    """
    Compute CF using pure numpy (fast inside optimizer loop).
    CF = Σ_i (w_i / Cap_i) * E_i
    co2_arr and cap_arr are pre-aligned numpy arrays (no NaN).
    """
    return float(np.dot(w, co2_arr / cap_arr))


def _prepare_cf_arrays(kept: list,
                       co2_year: pd.Series,
                       cap_year_m: pd.Series):
    """
    Pre-compute aligned numpy arrays for CF constraint (called once per year).
    Returns (co2_arr, cap_arr) — pure numpy, no pandas overhead in optimizer.
    """
    co2_r = co2_year.reindex(kept)
    cap_r = cap_year_m.reindex(kept)
    valid = co2_r.notna().values & cap_r.notna().values & (cap_r.values > 0)
    co2_arr = np.where(valid, co2_r.values, 0.0)
    cap_arr = np.where(valid, cap_r.values, 1e12)  # huge cap → negligible CF
    return co2_arr, cap_arr


def _vw_weights_year(invest_sets: dict,
                     mv_y: pd.DataFrame,
                     year: int) -> pd.Series:
    """Compute VW weights for the investment set at end of year Y."""
    isins_Y = invest_sets.get(year, [])
    col = _get_dec(mv_y, year)
    if col is None:
        return pd.Series(dtype=float)
    caps = mv_y[col].reindex(isins_Y).dropna()
    if caps.sum() == 0:
        return pd.Series(dtype=float)
    return caps / caps.sum()


def _build_sigma(ret_windows: dict, isins_Y: list, Y: int):
    """Build and clean covariance matrix for year Y."""
    ret_window = ret_windows[Y].loc[isins_Y]
    sigma = ret_window.T.cov(min_periods=36)
    valid = sigma.notna().all(axis=1)
    sigma = sigma.loc[valid, valid]
    return sigma


# ─────────────────────────────────────────────────────────────
# SECTION 3.2 — MV with CF constraint  P_oos^(mv)(0.5)
# ─────────────────────────────────────────────────────────────

def run_mv_carbon(invest_sets: dict,
                  ret_windows: dict,
                  weights_mv: dict,
                  co2: pd.DataFrame,
                  mv_y: pd.DataFrame,
                  reduction: float = 0.5) -> dict:
    """
    Minimum-variance portfolio with carbon footprint constraint:
        min  w'Σw
        s.t. CF(w) ≤ reduction × CF(P_mv)
             Σw = 1,  w ≥ 0
    """
    print("=" * 60)
    print(f"MV WITH CARBON CONSTRAINT  (CF ≤ {reduction:.0%} × CF_mv)")
    print("=" * 60)

    weights_mvc = {}

    for Y in REBALANCE_YEARS:
        if Y not in invest_sets or Y not in weights_mv:
            continue

        isins_Y = invest_sets[Y]
        sigma   = _build_sigma(ret_windows, isins_Y, Y)
        kept    = sigma.index.tolist()

        if len(kept) < 2:
            continue

        Sigma = sigma.values
        n     = len(kept)

        # CF of unconstrained MV portfolio
        cf_mv = compute_cf(weights_mv[Y], co2, mv_y, Y)
        if np.isnan(cf_mv) or cf_mv <= 0:
            weights_mvc[Y] = weights_mv[Y]
            print(f"  {Y}: CF_mv unavailable — using unconstrained MV.")
            continue

        cf_target = reduction * cf_mv

        # Prepare CO2 and cap arrays (once per year — fast numpy)
        co2_year = _get_dec(co2,  Y)
        cap_year = _get_dec(mv_y, Y)
        if co2_year is None or cap_year is None:
            weights_mvc[Y] = weights_mv[Y]
            continue

        co2_arr, cap_arr = _prepare_cf_arrays(kept, co2_year, cap_year / 1000)

        # Warm start from unconstrained MV weights
        w0 = weights_mv[Y].reindex(kept).fillna(0).values
        w0 = w0 / w0.sum() if w0.sum() > 0 else np.ones(n) / n

        constraints = [
            {"type": "eq",   "fun": lambda w: w.sum() - 1},
            {"type": "ineq", "fun": lambda w: cf_target -
             _cf_constraint_value(w, co2_arr, cap_arr)},
        ]

        result = minimize(
            fun         = lambda w: w @ Sigma @ w,
            x0          = w0,
            method      = "SLSQP",
            bounds      = [(0.0, 1.0)] * n,
            constraints = constraints,
            options     = {"ftol": 1e-9, "maxiter": 500},
        )

        w = result.x if result.success else w0
        weights_mvc[Y] = pd.Series(w, index=kept)

        cf_ach = _cf_constraint_value(w, co2_arr, cap_arr)
        vol    = np.sqrt(w @ Sigma @ w * 12) * 100
        print(f"  {Y}: {len(kept):>4} firms | "
              f"CF_mv={cf_mv:.2f} | CF_target={cf_target:.2f} | "
              f"CF_achieved={cf_ach:.2f} | ann.vol={vol:.2f}%")

    print(f"  ✓ MV carbon-constrained done for {len(weights_mvc)} years.\n")
    return weights_mvc


# ─────────────────────────────────────────────────────────────
# SECTION 3.3 — Tracking Error min with CF constraint
# ─────────────────────────────────────────────────────────────

def run_te_carbon(invest_sets: dict,
                  ret_windows: dict,
                  mv_y_bench: pd.DataFrame,
                  co2: pd.DataFrame,
                  mv_y: pd.DataFrame,
                  reduction: float = 0.5,
                  label: str = "vw") -> dict:
    """
    Tracking-error minimization with carbon footprint constraint:
        min  (w - w_vw)' Σ (w - w_vw)
        s.t. CF(w) ≤ reduction × CF(P_vw)
             Σw = 1,  w ≥ 0
    """
    print("=" * 60)
    print(f"TRACKING-ERROR MIN WITH CARBON CONSTRAINT  [{label}]")
    print("=" * 60)

    weights_te = {}

    for Y in REBALANCE_YEARS:
        if Y not in invest_sets:
            continue

        isins_Y = invest_sets[Y]
        sigma   = _build_sigma(ret_windows, isins_Y, Y)
        kept    = sigma.index.tolist()

        if len(kept) < 2:
            continue

        Sigma = sigma.values
        n     = len(kept)

        # VW benchmark weights
        w_vw_full = _vw_weights_year(invest_sets, mv_y_bench, Y)
        w_vw      = w_vw_full.reindex(kept).fillna(0)
        w_vw      = w_vw / w_vw.sum() if w_vw.sum() > 0 else w_vw
        w_vw_arr  = w_vw.values

        # CF of VW portfolio
        cf_vw = compute_cf(w_vw, co2, mv_y, Y)
        if np.isnan(cf_vw) or cf_vw <= 0:
            weights_te[Y] = pd.Series(w_vw_arr, index=kept)
            print(f"  {Y}: CF_vw unavailable — using VW weights.")
            continue

        # Year-varying or fixed reduction
        red_Y     = reduction[Y] if isinstance(reduction, dict) else reduction
        cf_target = red_Y * cf_vw

        # Prepare CO2 and cap arrays (once per year)
        co2_year = _get_dec(co2,  Y)
        cap_year = _get_dec(mv_y, Y)
        if co2_year is None or cap_year is None:
            weights_te[Y] = pd.Series(w_vw_arr, index=kept)
            continue

        co2_arr, cap_arr = _prepare_cf_arrays(kept, co2_year, cap_year / 1000)

        # Warm start from VW weights
        w0 = w_vw_arr.copy()

        constraints = [
            {"type": "eq",   "fun": lambda w: w.sum() - 1},
            {"type": "ineq", "fun": lambda w: cf_target -
             _cf_constraint_value(w, co2_arr, cap_arr)},
        ]

        def te_objective(w, _S=Sigma, _b=w_vw_arr):
            diff = w - _b
            return diff @ _S @ diff

        result = minimize(
            fun         = te_objective,
            x0          = w0,
            method      = "SLSQP",
            bounds      = [(0.0, 1.0)] * n,
            constraints = constraints,
            options     = {"ftol": 1e-9, "maxiter": 500},
        )

        w = result.x if result.success else w0
        weights_te[Y] = pd.Series(w, index=kept)

        cf_ach = _cf_constraint_value(w, co2_arr, cap_arr)
        te     = np.sqrt(te_objective(w) * 12) * 100
        print(f"  {Y}: CF_vw={cf_vw:.2f} | target={cf_target:.2f} | "
              f"achieved={cf_ach:.2f} | TE={te:.2f}%")

    print(f"  ✓ Tracking-error optimization done for {len(weights_te)} years.\n")
    return weights_te


# ─────────────────────────────────────────────────────────────
# SECTION 4.1 — Net Zero  P_oos^(vw)(NZ)
# ─────────────────────────────────────────────────────────────

def run_net_zero(invest_sets: dict,
                 ret_windows: dict,
                 mv_y_bench: pd.DataFrame,
                 co2: pd.DataFrame,
                 mv_y: pd.DataFrame,
                 theta: float = 0.10,
                 Y0: int = 2013) -> dict:
    """
    Net-zero portfolio: tracking error min with annually tightening CF constraint.
        CF(w) ≤ (1 - θ)^(Y - Y0) × CF_vw(Y0)
    θ = 10% per year, Y0 = 2013.
    """
    print("=" * 60)
    print(f"NET ZERO PORTFOLIO  (θ={theta:.0%}/yr, base year={Y0})")
    print("=" * 60)

    from carbon import compute_cf as _cf

    # Compute CF_vw at Y0 (reference level)
    col_Y0   = _get_dec(mv_y_bench, Y0)
    isins_Y0 = invest_sets.get(Y0, [])

    if col_Y0 is not None and len(isins_Y0) > 0:
        caps_Y0  = mv_y_bench[col_Y0].reindex(isins_Y0).dropna()
        w_vw_Y0  = caps_Y0 / caps_Y0.sum() if caps_Y0.sum() > 0 else pd.Series()
        cf_vw_Y0 = _cf(w_vw_Y0, co2, mv_y, Y0)
    else:
        cf_vw_Y0 = np.nan

    if np.isnan(cf_vw_Y0):
        print("  WARNING: CF_vw(Y0) unavailable — fallback to 50% reduction.")
        return run_te_carbon(invest_sets, ret_windows, mv_y_bench, co2, mv_y,
                             reduction=0.5, label="NZ-fallback")

    print(f"  CF_vw({Y0}) = {cf_vw_Y0:.4f}  (reference level)")

    weights_nz = {}

    for Y in REBALANCE_YEARS:
        if Y not in invest_sets:
            continue

        isins_Y = invest_sets[Y]
        sigma   = _build_sigma(ret_windows, isins_Y, Y)
        kept    = sigma.index.tolist()

        if len(kept) < 2:
            continue

        Sigma = sigma.values
        n     = len(kept)

        # VW benchmark weights
        col_Y  = _get_dec(mv_y_bench, Y)
        caps_Y = mv_y_bench[col_Y].reindex(kept).dropna() \
                 if col_Y is not None else pd.Series()
        if caps_Y.sum() > 0:
            w_vw = (caps_Y / caps_Y.sum()).reindex(kept).fillna(0)
        else:
            w_vw = pd.Series(np.ones(n) / n, index=kept)
        w_vw_arr = w_vw.values

        # Net-zero CF target: (1-θ)^(Y-Y0) × CF_vw(Y0)
        cf_target = (1 - theta) ** (Y - Y0) * cf_vw_Y0

        # Prepare CO2 and cap arrays
        co2_year = _get_dec(co2,  Y)
        cap_year = _get_dec(mv_y, Y)
        if co2_year is None or cap_year is None:
            weights_nz[Y] = w_vw
            continue

        co2_arr, cap_arr = _prepare_cf_arrays(kept, co2_year, cap_year / 1000)

        # Warm start from VW weights
        w0 = w_vw_arr.copy()

        constraints = [
            {"type": "eq",   "fun": lambda w: w.sum() - 1},
            {"type": "ineq", "fun": lambda w: cf_target -
             _cf_constraint_value(w, co2_arr, cap_arr)},
        ]

        def te_obj(w, _S=Sigma, _b=w_vw_arr):
            diff = w - _b
            return diff @ _S @ diff

        result = minimize(
            fun         = te_obj,
            x0          = w0,
            method      = "SLSQP",
            bounds      = [(0.0, 1.0)] * n,
            constraints = constraints,
            options     = {"ftol": 1e-9, "maxiter": 500},
        )

        w = result.x if result.success else w0
        weights_nz[Y] = pd.Series(w, index=kept)

        cf_ach = _cf_constraint_value(w, co2_arr, cap_arr)
        te     = np.sqrt(te_obj(w) * 12) * 100
        print(f"  {Y}: target={cf_target:.4f} | achieved={cf_ach:.4f} | TE={te:.2f}%")

    print(f"  ✓ Net-zero optimization done for {len(weights_nz)} years.\n")
    return weights_nz

