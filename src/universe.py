# src/universe.py
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config


# -------------------------------------------------
# Date helpers
# -------------------------------------------------
def _last_col_in_year(df: pd.DataFrame, year: int) -> pd.Timestamp:
    """
    Return the last available time column whose year == given year.
    Datastream monthly columns are usually month-end business days.
    """
    cols = pd.to_datetime(pd.Index(df.columns), errors="coerce")
    cols = cols[cols.notna()]
    cols_y = cols[cols.year == year]
    if len(cols_y) == 0:
        raise ValueError(f"No columns found for year={year}.")
    return cols_y.max()


def get_estimation_window(rets: pd.DataFrame, year: int, cfg: Config) -> pd.DataFrame:
    """
    10-year estimation window up to end of year Y:
    - uses the last available month in year Y
    - takes the last cfg.WINDOW_MONTHS columns
    """
    end = _last_col_in_year(rets, year)
    cols = pd.to_datetime(pd.Index(rets.columns), errors="coerce")
    cols = cols[cols.notna()]
    cols = cols[cols <= end]
    window_cols = cols[-cfg.WINDOW_MONTHS :]
    return rets.loc[:, window_cols]


# -------------------------------------------------
# Filters inside the investment set definition
# -------------------------------------------------
def filter_min_history(window: pd.DataFrame, cfg: Config) -> pd.Index:
    """Keep firms with at least cfg.MIN_MONTHS non-NaN returns in the window."""
    obs = window.notna().sum(axis=1)
    return obs[obs >= cfg.MIN_MONTHS].index


def filter_stale_prices(window: pd.DataFrame, cfg: Config) -> pd.Index:
    """
    Stale prices: compute share of months with return == 0 (tolerance) over the 10-year window.
    Exclude if share > cfg.STALE_THRESHOLD.
    """
    denom = window.notna().sum(axis=1).replace(0, np.nan)
    zero_share = (window.abs() < cfg.EPS_ZERO).sum(axis=1) / denom
    return zero_share[zero_share <= cfg.STALE_THRESHOLD].index


def filter_price_available_end_year(ri_m_clean: pd.DataFrame, year: int) -> pd.Index:
    """
    Low prices are treated as missing in RI_M_CLEAN.
    If price is missing at end of year Y -> not investable in Y+1.
    """
    end = _last_col_in_year(ri_m_clean, year)
    return ri_m_clean.loc[:, end].dropna().index


def _choose_annual_column_leq(df_y: pd.DataFrame, year: int):
    """
    For annual tables (CO2, REV, MV_Y), choose the latest available column <= year.
    This avoids killing the universe at the beginning of the sample.
    """
    cols = pd.Index(df_y.columns)
    yrs = np.array([c.year if isinstance(c, pd.Timestamp) else int(c) for c in cols])
    ok = yrs <= year
    if not ok.any():
        return None
    chosen_year = yrs[ok].max()
    col_idx = np.where(yrs == chosen_year)[0][0]
    return cols[col_idx]


def filter_carbon_available_end_year(co2_s1_y_clean: pd.DataFrame, year: int) -> pd.Index:
    """
    Requirement from instructions:
    Exclude firms without carbon data available at the end of year Y.
    Group A uses Scope 1, so we check CO2 Scope 1.

    Robust rule (no look-ahead):
    - use the latest available CO2 year <= Y
    - require non-NaN CO2 for that year
    """
    col = _choose_annual_column_leq(co2_s1_y_clean, year)
    if col is None:
        return pd.Index([])
    return co2_s1_y_clean.loc[:, col].dropna().index


# -------------------------------------------------
# Region filter (North America)
# -------------------------------------------------
def _find_region_column(static_df: pd.DataFrame) -> str:
    """
    Find the region column in Static_2025.xlsx.
    In your file, the column is 'Region'.
    """
    candidates = ["Region", "REGION", "region"]
    for c in candidates:
        if c in static_df.columns:
            return c
    raise ValueError(
        "Could not find 'Region' column in STATIC.\n"
        f"Available columns: {list(static_df.columns)}"
    )


def _find_isin_column(static_df: pd.DataFrame) -> str:
    """Find an ISIN column in Static (best effort)."""
    candidates = ["ISIN", "Isin", "isin"]
    for c in candidates:
        if c in static_df.columns:
            return c
    return static_df.columns[0]


def filter_region(static_df: pd.DataFrame, region_name: str) -> pd.Index:
    """
    Return ISINs belonging to the assigned region.

    IMPORTANT (based on your Static_2025.xlsx):
    - static_df['Region'] uses short codes like: AMER, EM, EUR
    - So for North America, we must match 'AMER'

    We accept:
    - 'AMER'
    - 'North America' (mapped to AMER)
    - 'Americas' (mapped to AMER)
    """
    isin_col = _find_isin_column(static_df)
    reg_col = _find_region_column(static_df)

    tmp = static_df[[isin_col, reg_col]].copy()
    tmp[isin_col] = tmp[isin_col].astype(str).str.strip()
    tmp[reg_col] = tmp[reg_col].astype(str).str.strip().str.upper()

    target = region_name.strip().upper()

    alias_map = {
        "NORTH AMERICA": "AMER",
        "AMERICAS": "AMER",
        "AMERICA": "AMER",
        "AMER": "AMER",
    }
    target_code = alias_map.get(target, target)  # if already a code, keep it

    mask = tmp[reg_col] == target_code
    return pd.Index(tmp.loc[mask, isin_col].unique())


# -------------------------------------------------
# Main function: eligible investment set at end of year Y
# -------------------------------------------------
def eligible_assets_part1(
    data: dict,
    year: int,
    cfg: Config,
    require_mv: bool = True,
) -> pd.Index:
    """
    Investment set at end of year Y (to invest during Y+1), following instructions.

    Uses:
    - RET_M (monthly returns)
    - RI_M_CLEAN (monthly prices after low-price rule)
    - MV_M_CLEAN (monthly market values)
    - CO2_S1_Y_CLEAN (annual Scope 1)
    - STATIC (region classification)
    """
    rets = data["RET_M"]
    ri = data["RI_M_CLEAN"]
    mv = data["MV_M_CLEAN"]
    co2 = data["CO2_S1_Y_CLEAN"]
    static = data["STATIC"]

    # 0) start from common index across core tables (avoid KeyErrors)
    common = rets.index.intersection(ri.index).intersection(mv.index).intersection(co2.index)

    # 0b) region filter (North America -> AMER)
    if cfg.REGION is not None:
        isin_allowed = filter_region(static, cfg.REGION)
        if isinstance(common, pd.MultiIndex):
            isin_level = common.get_level_values(0).astype(str)
            common = common[isin_level.isin(isin_allowed.astype(str))]
        else:
            common = common.intersection(isin_allowed)

    # subset
    rets = rets.loc[common]
    ri = ri.loc[common]
    mv = mv.loc[common]
    co2 = co2.loc[common]

    # 1) estimation window (120 months up to end Y)
    window = get_estimation_window(rets, year, cfg)

    # 2) minimum history
    ok_hist = filter_min_history(window, cfg)

    # 3) stale prices on the same window (no look-ahead)
    ok_stale = filter_stale_prices(window.loc[ok_hist], cfg)

    # 4) price available end of year Y (after low-price masking)
    ok_price = filter_price_available_end_year(ri.loc[ok_stale], year)

    # 5) carbon data available end of year Y (Scope 1)
    ok_price = ok_price.intersection(co2.index)
    ok_co2 = filter_carbon_available_end_year(co2.loc[ok_price], year)

    # 6) MV available end of year Y (recommended for VW benchmark)
    if require_mv:
        end = _last_col_in_year(mv, year)
        ok_mv = mv.loc[ok_co2, end].dropna().index
        return ok_mv

    return ok_co2


# -------------------------------------------------
# Quick test: python3 -m src.universe
# -------------------------------------------------
if __name__ == "__main__":
    from .io import load_raw
    from .data_cleaning import clean_part1

    cfg = Config()
    raw = load_raw(cfg)
    clean = clean_part1(raw, cfg)

    # Example: Y=2013 -> invest 2014
    u = eligible_assets_part1(clean, year=2013, cfg=cfg, require_mv=True)
    print("Eligible assets (2013 -> invest 2014):", len(u))
