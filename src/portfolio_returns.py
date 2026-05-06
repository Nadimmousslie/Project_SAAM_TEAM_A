# portfolio_returns.py — Ex-post portfolio returns (MV & VW)

import pandas as pd
import numpy as np
from config import REBALANCE_YEARS


def _get_dec_col(df: pd.DataFrame, year: int):
    cols = pd.DatetimeIndex(df.columns)
    dec_cols = cols[(cols.year == year) & (cols.month == 12)]
    return dec_cols[-1] if len(dec_cols) > 0 else None


# Min-Variance ex-post returns

def compute_mv_returns(weights_dict: dict, returns_m: pd.DataFrame) -> pd.Series:
    """
    Compute monthly ex-post returns of the minimum-variance portfolio.
    Weights are set at end of Y and drift dynamically within year Y+1.
    """
    all_cols  = pd.DatetimeIndex(returns_m.columns)
    mv_returns = {}

    for Y in REBALANCE_YEARS:
        if Y not in weights_dict:
            continue

        w0        = weights_dict[Y]
        isins_Y   = w0.index.tolist()
        year_cols = all_cols[all_cols.year == Y + 1]

        if len(year_cols) == 0:
            continue

        firms_in_data = [i for i in isins_Y if i in returns_m.index]
        ret_year = (
            returns_m.loc[firms_in_data, year_cols]
            .reindex(index=isins_Y)
            .fillna(0)
        )

        alpha = w0.values.copy()

        for col in year_cols:
            r_vec = ret_year[col].values
            r_p   = float(alpha @ r_vec)
            mv_returns[col] = r_p

            denom = 1 + r_p
            if abs(denom) > 1e-10:
                alpha = alpha * (1 + r_vec) / denom

    mv_series = pd.Series(mv_returns).sort_index()
    print(f"  MV : {len(mv_series)} months  "
          f"({mv_series.index[0].date()} → {mv_series.index[-1].date()})")
    return mv_series


# Value-weighted benchmark returns

def compute_vw_returns(invest_sets: dict,
                       returns_m: pd.DataFrame,
                       mv_m: pd.DataFrame) -> pd.Series:
    """
    Compute monthly returns of the value-weighted benchmark.
    w_{i,t} = Cap_{i,t} / Σ_j Cap_{j,t}
    """
    all_cols_ret = pd.DatetimeIndex(returns_m.columns)
    all_cols_mv  = pd.DatetimeIndex(mv_m.columns)
    vw_returns   = {}

    for Y in REBALANCE_YEARS:
        if Y not in invest_sets:
            continue

        isins_Y   = invest_sets[Y]
        year_cols = all_cols_ret[all_cols_ret.year == Y + 1]

        if len(year_cols) == 0:
            continue

        for col in year_cols:
            prev_month    = col - pd.offsets.MonthEnd(1)
            mv_cols_avail = all_cols_mv[all_cols_mv <= prev_month]
            if len(mv_cols_avail) == 0:
                continue
            mv_col = mv_cols_avail[-1]

            firms_in_mv = [i for i in isins_Y if i in mv_m.index]
            caps = mv_m.loc[firms_in_mv, mv_col].dropna()

            if caps.sum() == 0 or len(caps) == 0:
                continue

            w_vw = caps / caps.sum()

            firms_in_ret = [i for i in w_vw.index if i in returns_m.index]
            ret_col = returns_m.loc[firms_in_ret, col].fillna(0)

            r_vw = (w_vw.reindex(ret_col.index).fillna(0) * ret_col).sum()
            vw_returns[col] = r_vw

    vw_series = pd.Series(vw_returns).sort_index()
    print(f"  VW : {len(vw_series)} months  "
          f"({vw_series.index[0].date()} → {vw_series.index[-1].date()})")
    return vw_series  # ← FIX: était "return mv_series, vw_series" par erreur


def compute_all_returns(weights_dict, invest_sets, data):
    """Convenience wrapper — returns (mv_series, vw_series)."""
    mv_series = compute_mv_returns(weights_dict, data["returns_m"])
    vw_series = compute_vw_returns(invest_sets, data["returns_m"], data["mv_m"])
    print("  ✓ Portfolio returns computed.\n")
    return mv_series, vw_series
