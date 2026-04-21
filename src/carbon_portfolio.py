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
# HELPER
# ─────────────────────────────────────────────────────────────

def _get_dec(df: pd.DataFrame, year: int):
    cols = pd.DatetimeIndex(df.columns)
    dec  = cols[(cols.year == year) & (cols.month == 12)]
    return df[dec[-1]] if len(dec) > 0 else None


def _cf_constraint_value(w: np.ndarray,
                          isins: list,
                          co2_year: pd.Series,
                          cap_year_m: pd.Series) -> float:
    """
    Compute CF for a weight vector w (numpy array).
    CF = Σ_i (w_i / Cap_i) * E_i
    """
    w_s   = pd.Series(w, index=isins)
    valid = co2_year.notna() & cap_year_m.notna() & (cap_year_m > 0)
    w_v   = w_s[valid]
    e_v   = co2_year[valid]
    c_v   = cap_year_m[valid]
    return float((w_v / c_v * e_v).sum())


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

    Parameters
    ----------
    weights_mv  : unconstrained MV weights {year → pd.Series}
    reduction   : CF target as fraction of MV CF (default 0.5)

    Returns
    -------
    weights_mvc : {year → pd.Series}
    """
    print("=" * 60)
    print(f"MV WITH CARBON CONSTRAINT  (CF ≤ {reduction:.0%} × CF_mv)")
    print("=" * 60)

    weights_mvc = {}

    for Y in REBALANCE_YEARS:
        if Y not in invest_sets or Y not in weights_mv:
            continue

        isins_Y    = invest_sets[Y]
        ret_window = ret_windows[Y].loc[isins_Y]

        # Covariance matrix
        sigma = ret_window.T.cov(min_periods=36)
        valid = sigma.notna().all(axis=1)
        sigma = sigma.loc[valid, valid]
        kept  = sigma.index.tolist()

        if len(kept) < 2:
            continue

        Sigma = sigma.values
        n     = len(kept)

        # CF of unconstrained MV portfolio
        cf_mv = compute_cf(weights_mv[Y], co2, mv_y, Y)
        if np.isnan(cf_mv) or cf_mv <= 0:
            # No valid CF → fall back to unconstrained
            weights_mvc[Y] = weights_mv[Y]
            print(f"  {Y}: CF_mv unavailable — using unconstrained MV.")
            continue

        cf_target = reduction * cf_mv

        # Prepare CO2 and cap data for constraint
        co2_year  = _get_dec(co2,  Y)
        cap_year  = _get_dec(mv_y, Y)
        if co2_year is None or cap_year is None:
            weights_mvc[Y] = weights_mv[Y]
            continue

        cap_year_m = cap_year / 1000  # → millions USD
        co2_y = co2_year.reindex(kept).fillna(0)
        cap_y = cap_year_m.reindex(kept).fillna(1e9)  # large cap → negligible

        # Optimization
        w0 = np.ones(n) / n
        constraints = [
            {"type": "eq",  "fun": lambda w: w.sum() - 1},
            {"type": "ineq","fun": lambda w: cf_target -
             _cf_constraint_value(w, kept, co2_y, cap_y)},
        ]
        bounds = [(0.0, 1.0)] * n

        result = minimize(
            fun         = lambda w: w @ Sigma @ w,
            x0          = w0,
            method      = "SLSQP",
            bounds      = bounds,
            constraints = constraints,
            options     = {"ftol": 1e-12, "maxiter": 2000},
        )

        w = result.x if result.success else w0
        weights_mvc[Y] = pd.Series(w, index=kept)

        cf_achieved = _cf_constraint_value(w, kept, co2_y, cap_y)
        vol = np.sqrt(w @ Sigma @ w * 12) * 100
        print(f"  {Y}: {len(kept):>4} firms | "
              f"CF_mv={cf_mv:.2f} | CF_target={cf_target:.2f} | "
              f"CF_achieved={cf_achieved:.2f} | ann.vol={vol:.2f}%")

    print(f"  ✓ MV carbon-constrained done for {len(weights_mvc)} years.\n")
    return weights_mvc


# ─────────────────────────────────────────────────────────────
# SECTION 3.3 — Tracking Error min with CF constraint  P_oos^(vw)(0.5)
# ─────────────────────────────────────────────────────────────

def _vw_weights_year(invest_sets: dict,
                     mv_y: pd.DataFrame,
                     year: int) -> pd.Series:
    """Compute VW weights for the investment set at end of year Y using yearly market cap."""
    isins_Y = invest_sets.get(year, [])
    col = _get_dec(mv_y, year)
    if col is None:
        return pd.Series(dtype=float)

    caps = mv_y[col].reindex(isins_Y).dropna()
    if caps.sum() == 0:
        return pd.Series(dtype=float)
    return caps / caps.sum()


def run_te_carbon(invest_sets: dict,
                  ret_windows: dict,
                  mv_m: pd.DataFrame,
                  co2: pd.DataFrame,
                  mv_y: pd.DataFrame,
                  reduction: float = 0.5,
                  label: str = "vw") -> dict:
    """
    Tracking-error minimization with carbon footprint constraint:
        min  (w - w_vw)' Σ (w - w_vw)
        s.t. CF(w) ≤ reduction × CF(P_vw)
             Σw = 1,  w ≥ 0

    Used for both Section 3.3 (reduction=0.5) and Section 4.1 (net zero).

    Parameters
    ----------
    reduction  : float or dict {year → float}
                 If dict, allows year-varying targets (for net zero).
    label      : for printing

    Returns
    -------
    weights_te : {year → pd.Series}
    """
    print("=" * 60)
    print(f"TRACKING-ERROR MIN WITH CARBON CONSTRAINT  [{label}]")
    print("=" * 60)

    weights_te = {}

    for Y in REBALANCE_YEARS:
        if Y not in invest_sets:
            continue

        isins_Y    = invest_sets[Y]
        ret_window = ret_windows[Y].loc[isins_Y]

        # Covariance matrix
        sigma = ret_window.T.cov(min_periods=36)
        valid = sigma.notna().all(axis=1)
        sigma = sigma.loc[valid, valid]
        kept  = sigma.index.tolist()

        if len(kept) < 2:
            continue

        Sigma = sigma.values
        n     = len(kept)

        # VW weights (benchmark)
        w_vw_full = _vw_weights_year(invest_sets, mv_m, Y)
        w_vw = w_vw_full.reindex(kept).fillna(0)
        if w_vw.sum() > 0:
            w_vw = w_vw / w_vw.sum()
        w_vw_arr = w_vw.values

        # CF of VW portfolio
        cf_vw = compute_cf(w_vw, co2, mv_y, Y)
        if np.isnan(cf_vw) or cf_vw <= 0:
            weights_te[Y] = w_vw
            print(f"  {Y}: CF_vw unavailable — using VW weights.")
            continue

        # Year-varying or fixed reduction
        red_Y = reduction[Y] if isinstance(reduction, dict) else reduction
        cf_target = red_Y * cf_vw

        # CO2 and cap data
        co2_year  = _get_dec(co2,  Y)
        cap_year  = _get_dec(mv_y, Y)
        if co2_year is None or cap_year is None:
            weights_te[Y] = w_vw
            continue

        cap_year_m = cap_year / 1000
        co2_y = co2_year.reindex(kept).fillna(0)
        cap_y = cap_year_m.reindex(kept).fillna(1e9)

        # Optimization — minimize tracking error
        w0 = w_vw_arr.copy()
        constraints = [
            {"type": "eq",  "fun": lambda w: w.sum() - 1},
            {"type": "ineq","fun": lambda w: cf_target -
             _cf_constraint_value(w, kept, co2_y, cap_y)},
        ]
        bounds = [(0.0, 1.0)] * n

        def te_objective(w):
            diff = w - w_vw_arr
            return diff @ Sigma @ diff

        result = minimize(
            fun         = te_objective,
            x0          = w0,
            method      = "SLSQP",
            bounds      = bounds,
            constraints = constraints,
            options     = {"ftol": 1e-12, "maxiter": 2000},
        )

        w = result.x if result.success else w0
        weights_te[Y] = pd.Series(w, index=kept)

        cf_achieved = _cf_constraint_value(w, kept, co2_y, cap_y)
        te = np.sqrt(te_objective(w) * 12) * 100
        print(f"  {Y}: CF_vw={cf_vw:.2f} | target={cf_target:.2f} | "
              f"achieved={cf_achieved:.2f} | TE={te:.2f}%")

    print(f"  ✓ Tracking-error optimization done for {len(weights_te)} years.\n")
    return weights_te


# ─────────────────────────────────────────────────────────────
# SECTION 4.1 — Net Zero  P_oos^(vw)(NZ)
# ─────────────────────────────────────────────────────────────

def run_net_zero(invest_sets: dict,
                 ret_windows: dict,
                 mv_m: pd.DataFrame,
                 co2: pd.DataFrame,
                 mv_y: pd.DataFrame,
                 theta: float = 0.10,
                 Y0: int = 2013) -> dict:
    """
    Net-zero portfolio: tracking error min with annually tightening CF constraint.
        CF(w) ≤ (1 - θ)^(Y - Y0) × CF_vw(Y0)

    θ = 10% per year by default, Y0 = 2013.

    Returns
    -------
    weights_nz : {year → pd.Series}
    """
    print("=" * 60)
    print(f"NET ZERO PORTFOLIO  (θ={theta:.0%}/yr, base year={Y0})")
    print("=" * 60)

    # First compute CF_vw at Y0 (the reference level)
    # We will compute it inside run_te_carbon using year-varying reduction dict

    # Build year-varying reduction factors
    # reduction[Y] = (1 - theta)^(Y - Y0)   relative to CF_vw(Y)
    # But the constraint uses CF_vw(Y0) as reference, not CF_vw(Y)
    # So we need to pass CF_vw(Y0) explicitly

    # Compute VW weights and CF at Y0
    from carbon import compute_cf as _cf

    # Get VW weights at Y0
    col_Y0 = _get_dec(mv_m, Y0)
    isins_Y0 = invest_sets.get(Y0, [])
    if col_Y0 is not None and len(isins_Y0) > 0:
        caps_Y0 = mv_m[col_Y0].reindex(isins_Y0).dropna()
        w_vw_Y0 = caps_Y0 / caps_Y0.sum() if caps_Y0.sum() > 0 else pd.Series()
        cf_vw_Y0 = _cf(w_vw_Y0, co2, mv_y, Y0)
    else:
        cf_vw_Y0 = np.nan

    if np.isnan(cf_vw_Y0):
        print("  WARNING: CF_vw(Y0) unavailable — using year-by-year VW CF.")
        # Fall back to year-varying 50% reduction
        return run_te_carbon(invest_sets, ret_windows, mv_m, co2, mv_y,
                             reduction=0.5, label="NZ-fallback")

    print(f"  CF_vw({Y0}) = {cf_vw_Y0:.4f}  (reference level)")

    # Build reduction dict: for each year Y, target = (1-θ)^(Y-Y0) * CF_vw(Y0)
    # expressed as a fraction of CF_vw(Y) for compatibility with run_te_carbon
    # Since run_te_carbon computes CF_vw(Y) internally and multiplies by reduction[Y],
    # we need: reduction[Y] × CF_vw(Y) = (1-θ)^(Y-Y0) × CF_vw(Y0)
    # → reduction[Y] = (1-θ)^(Y-Y0) × CF_vw(Y0) / CF_vw(Y)

    # We handle this directly here with a custom optimization loop
    weights_nz = {}

    for Y in REBALANCE_YEARS:
        if Y not in invest_sets:
            continue

        isins_Y    = invest_sets[Y]
        ret_window = ret_windows[Y].loc[isins_Y]

        sigma = ret_window.T.cov(min_periods=36)
        valid = sigma.notna().all(axis=1)
        sigma = sigma.loc[valid, valid]
        kept  = sigma.index.tolist()

        if len(kept) < 2:
            continue

        Sigma = sigma.values
        n     = len(kept)

        # VW benchmark weights
        col_Y     = _get_dec(mv_m, Y)
        caps_Y    = mv_m[col_Y].reindex(kept).dropna() if col_Y is not None else pd.Series()
        w_vw      = (caps_Y / caps_Y.sum()).reindex(kept).fillna(0) \
                    if caps_Y.sum() > 0 else pd.Series(np.ones(n)/n, index=kept)
        w_vw_arr  = w_vw.values

        # Net-zero CF target
        cf_target = (1 - theta) ** (Y - Y0) * cf_vw_Y0

        # CO2 and cap data
        co2_year  = _get_dec(co2,  Y)
        cap_year  = _get_dec(mv_y, Y)
        if co2_year is None or cap_year is None:
            weights_nz[Y] = w_vw
            continue

        cap_year_m = cap_year / 1000
        co2_y = co2_year.reindex(kept).fillna(0)
        cap_y = cap_year_m.reindex(kept).fillna(1e9)

        w0 = w_vw_arr.copy()
        constraints = [
            {"type": "eq",  "fun": lambda w: w.sum() - 1},
            {"type": "ineq","fun": lambda w: cf_target -
             _cf_constraint_value(w, kept, co2_y, cap_y)},
        ]
        bounds = [(0.0, 1.0)] * n

        def te_obj(w):
            diff = w - w_vw_arr
            return diff @ Sigma @ diff

        result = minimize(
            fun=te_obj, x0=w0, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 2000},
        )

        w = result.x if result.success else w0
        weights_nz[Y] = pd.Series(w, index=kept)

        cf_ach = _cf_constraint_value(w, kept, co2_y, cap_y)
        te     = np.sqrt(te_obj(w) * 12) * 100
        print(f"  {Y}: target={cf_target:.4f} | achieved={cf_ach:.4f} | TE={te:.2f}%")

    print(f"  ✓ Net-zero optimization done for {len(weights_nz)} years.\n")
    return weights_nz

