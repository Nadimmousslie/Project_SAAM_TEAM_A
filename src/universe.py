# src/universe.py
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config

EPS_ZERO = 1e-12  # tolerance for "return == 0"


# -------------------------
# Date helpers
# -------------------------
def _year_end_date(year: int) -> pd.Timestamp:
    return pd.Timestamp(f"{year}-12-31")


def _last_col_in_year(df: pd.DataFrame, year: int) -> pd.Timestamp:
    cols = pd.to_datetime(pd.Index(df.columns), errors="coerce")
    cols = cols[cols.notna()]
    cols_y = cols[cols.year == year]
    if len(cols_y) == 0:
        raise ValueError(f"No columns found for year={year}.")
    return cols_y.max()


# -------------------------
# Estimation window
# -------------------------
def get_estimation_window(rets: pd.DataFrame, year: int, cfg: Config) -> pd.DataFrame:
    end = _last_col_in_year(rets, year)
    cols = pd.to_datetime(pd.Index(rets.columns), errors="coerce")
    cols = cols[cols.notna()]
    cols = cols[cols <= end]
    window_cols = cols[-cfg.WINDOW_MONTHS :]
    return rets.loc[:, window_cols]


def filter_min_history(window: pd.DataFrame, cfg: Config) -> pd.Index:
    count = window.notna().sum(axis=1)
    return count[count >= cfg.MIN_MONTHS_IN_WINDOW].index


def filter_stale_prices(window: pd.DataFrame, cfg: Config) -> pd.Index:
    denom = window.notna().sum(axis=1).replace(0, np.nan)
    zero_share = (window.abs() < EPS_ZERO).sum(axis=1) / denom
    ok = zero_share[zero_share <= cfg.STALE_THRESHOLD].index
    return ok


# -------------------------
# End-of-year availability filters
# -------------------------
def filter_price_available_end_year(ri_clean: pd.DataFrame, year: int) -> pd.Index:
    end = _last_col_in_year(ri_clean, year)
    return ri_clean.loc[:, end].dropna().index


def filter_co2_available(co2_y_clean: pd.DataFrame, year: int) -> pd.Index:
    """
    CO2 is annual. We want "information available at end of year Y" without look-ahead.

    Robust rule:
    - If year is BEFORE the first CO2 year available in the dataset -> do NOT filter (keep all firms),
      because it's missing at the beginning of the sample (nothing we can do).
    - Else choose the latest CO2 year <= Y for which the column exists, and require non-NaN.
    - If that column is entirely NaN (rare) -> do NOT filter to avoid empty universe.
    """
    # map columns to int years
    cols_year = np.array([
        c.year if isinstance(c, pd.Timestamp) else int(c)
        for c in co2_y_clean.columns
    ])

    if len(cols_year) == 0:
        return co2_y_clean.index

    min_year = int(cols_year.min())
    if year < min_year:
        # missing at beginning: don't exclude everyone
        return co2_y_clean.index

    eligible_years = cols_year[cols_year <= year]
    if len(eligible_years) == 0:
        return co2_y_clean.index  # safe fallback

    chosen_year = int(eligible_years.max())
    col_idx = np.where(cols_year == chosen_year)[0][0]
    col = co2_y_clean.columns[col_idx]

    ok = co2_y_clean.loc[:, col].dropna().index

    # If all are missing, don't filter (prevents eligible=0)
    if len(ok) == 0:
        return co2_y_clean.index

    return ok


def filter_mv_available_end_year(mv_m_clean: pd.DataFrame, year: int) -> pd.Index:
    end = _last_col_in_year(mv_m_clean, year)
    return mv_m_clean.loc[:, end].dropna().index


# -------------------------------
# Country / Region filter (USA)
# -------------------------------
def _static_country_filter(static_df: pd.DataFrame, target: str) -> pd.Index:
    if static_df is None or len(static_df) == 0:
        return pd.Index([])

    id_candidates = ["ISIN", "Isin", "isin"]
    id_col = next((c for c in id_candidates if c in static_df.columns), None)
    if id_col is None:
        id_col = static_df.columns[0]

    country_candidates = [
        "COUNTRY", "Country", "country",
        "COUNTRY_CODE", "Country Code", "country_code",
        "ISO", "ISO2", "ISO3",
        "NATION", "Nation",
        "DOMICILE", "Domicile",
        "REGION", "Region",
    ]
    country_col = next((c for c in country_candidates if c in static_df.columns), None)
    if country_col is None:
        raise ValueError(
            "Could not find a country/region column in STATIC. "
            "Open Static_2025.xlsx and tell me the exact column name for country."
        )

    s = static_df[[id_col, country_col]].copy()
    s[id_col] = s[id_col].astype(str).str.strip()
    s[country_col] = s[country_col].astype(str).str.strip().str.upper()

    tgt = target.strip().upper()
    usa_aliases = {"USA", "UNITED STATES", "UNITED STATES OF AMERICA", "US"}
    allowed = usa_aliases if tgt in usa_aliases else {tgt}

    ok_ids = s.loc[s[country_col].isin(allowed), id_col]
    return pd.Index(ok_ids.unique())


# -------------------------
# Main: eligible universe
# -------------------------
def eligible_assets_part1(
    data: dict,
    year: int,
    cfg: Config,
    require_mv: bool = True,
) -> pd.Index:
    rets = data["RET_M"]
    ri = data["RI_M_CLEAN"]
    co2 = data["CO2_Y_CLEAN"]
    mv = data.get("MV_M_CLEAN", None)
    static = data.get("STATIC", None)

    # 0) common index (avoid KeyErrors)
    common = rets.index.intersection(ri.index).intersection(co2.index)
    if require_mv and mv is not None:
        common = common.intersection(mv.index)

    # country filter (optional)
    if cfg.REGION is not None:
        ids = _static_country_filter(static, cfg.REGION)
        if isinstance(common, pd.MultiIndex):
            isin_level = common.get_level_values(0).astype(str)
            common = common[isin_level.isin(ids.astype(str))]
        else:
            common = common.intersection(ids)

    # subset
    rets = rets.loc[common]
    ri = ri.loc[common]
    co2 = co2.loc[common]
    if require_mv and mv is not None:
        mv = mv.loc[common]

    # 1) window
    window = get_estimation_window(rets, year, cfg)

    # 2) history
    ok_hist = filter_min_history(window, cfg)

    # 3) stale
    ok_stale = filter_stale_prices(window.loc[ok_hist], cfg)

    # 4) price end-year
    ok_price = filter_price_available_end_year(ri.loc[ok_stale], year)

    # 5) CO2 availability (robust rule)
    ok_price = ok_price.intersection(co2.index)
    ok_co2 = filter_co2_available(co2.loc[ok_price], year)

    # 6) MV availability (optional)
    if require_mv and mv is not None:
        ok_co2 = ok_co2.intersection(mv.index)
        ok_mv = filter_mv_available_end_year(mv.loc[ok_co2], year)
        return ok_mv

    return ok_co2
