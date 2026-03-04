# src/backtest.py
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config
from .universe import eligible_assets_part1, get_estimation_window
from .portfolios import min_var_long_only, value_weighted_weights


def _months_in_year(cols: pd.Index, year: int) -> pd.Index:
    cols = pd.to_datetime(cols, errors="coerce")
    cols = cols[cols.notna()]
    return cols[cols.year == year].sort_values()


def run_backtest_part1(data: dict, cfg: Config, require_mv_universe: bool = True) -> pd.DataFrame:
    """
    Backtest Part I:
    - For each year Y = cfg.START_YEAR ... cfg.END_YEAR:
        * build investable set at end of Y (invest in Y+1)
        * estimate covariance on 10y window up to end Y
        * compute MinVar weights (long-only)
        * compute monthly returns for year Y+1 using fixed weights
        * compute VW benchmark monthly returns for year Y+1 using MV weights each month

    Returns: DataFrame with columns ["MinVar", "ValueWeighted"], indexed by month-end dates (from RI columns).
    """
    rets = data["RET_M"]         # firms x months
    ri = data["RI_M_CLEAN"]      # firms x months (for month index reference)
    mv = data["MV_M_CLEAN"]      # firms x months

    results = []

    for Y in range(cfg.START_YEAR, cfg.END_YEAR + 1):
        invest_year = Y + 1

        # investable set (end of year Y)
        assets = eligible_assets_part1(data, Y, cfg, require_mv=require_mv_universe)
        if len(assets) < 2:
            continue

        # estimation window (10y up to end of Y)
        window = get_estimation_window(rets.loc[assets], Y, cfg)

        # covariance estimation (monthly returns)
        cov = window.T.cov()  # assets x assets
        cov = cov.loc[assets, assets]

        # MinVar weights
        w_mv = min_var_long_only(cov)

        # out-of-sample months for invest_year
        months = _months_in_year(rets.columns, invest_year)
        if len(months) == 0:
            continue

        oos_rets = rets.loc[assets, months]

        # MinVar portfolio returns: fixed weights for the year
        r_minvar = (oos_rets.T @ w_mv).rename("MinVar")

        # Value-weighted benchmark returns: weights updated monthly
        r_vw_list = []
        for m in months:
            mv_row = mv.loc[assets, m]
            w_vw = value_weighted_weights(mv_row)
            # align assets (in case some MV are missing)
            month_assets = w_vw.index.intersection(oos_rets.index)
            if len(month_assets) == 0:
                r_vw_list.append(np.nan)
                continue
            r_vw_list.append(float(oos_rets.loc[month_assets, m] @ w_vw.loc[month_assets]))

        r_vw = pd.Series(r_vw_list, index=months, name="ValueWeighted")

        results.append(pd.concat([r_minvar, r_vw], axis=1))

    if not results:
        return pd.DataFrame(columns=["MinVar", "ValueWeighted"])

    bt = pd.concat(results).sort_index()

    # restrict to project horizon
    bt = bt.loc[pd.to_datetime(cfg.START_OOS) : pd.to_datetime(cfg.END_OOS)]
    return bt
