# src/data_cleaning.py
from __future__ import annotations

from typing import Dict
import numpy as np
import pandas as pd

from .config import Config

EPS_ZERO = 1e-12


# -------------------------------------------------
# Helpers (format / alignment)
# -------------------------------------------------
def _ensure_datetime_sorted_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure time columns are datetime-like and sorted.
    Works for Datastream exports where columns are dates.
    """
    df2 = df.copy()
    cols_dt = pd.to_datetime(df2.columns, errors="coerce")
    if cols_dt.notna().mean() > 0.8:
        df2.columns = cols_dt
        df2 = df2.loc[:, df2.columns.notna()].sort_index(axis=1)
    return df2


def _align_to_index(df: pd.DataFrame, ref_index: pd.Index) -> pd.DataFrame:
    """
    Keep only rows whose index is present in ref_index (intersection).
    This enforces the rule 'delete missing prices firm from all tables'.
    """
    common = df.index.intersection(ref_index)
    return df.loc[common]


# -------------------------------------------------
# 1) Missing prices: full row missing -> drop firm
# -------------------------------------------------
def drop_firms_with_no_prices(ri_m: pd.DataFrame) -> pd.DataFrame:
    """
    Missing prices:
    If Datastream cannot match a firm, the entire row is missing.
    Rule: delete the firm from all tables.

    Here we apply it on RI_M and later align all other tables on the remaining firms.
    """
    ri = ri_m.apply(pd.to_numeric, errors="coerce")
    ri = ri.dropna(axis=0, how="all")
    ri = _ensure_datetime_sorted_columns(ri)
    return ri


# -------------------------------------------------
# 2) Low prices: RI < 0.5 treated as missing
# -------------------------------------------------
def apply_low_price_rule(ri_m: pd.DataFrame, min_price: float) -> pd.DataFrame:
    """
    Low prices:
    Datastream RI can be very small (or 0 due to rounding), causing extreme returns.
    Rule: treat RI < 0.5 as missing values (NaN).

    IMPORTANT:
    We do NOT forward-fill prices. Missing end-of-year investability is handled in universe.py.
    """
    ri = ri_m.copy()
    ri = ri.apply(pd.to_numeric, errors="coerce")
    ri = ri.mask(ri < min_price)
    ri = _ensure_datetime_sorted_columns(ri)
    return ri


# -------------------------------------------------
# 3) Annual missing values (middle): use previous year
# -------------------------------------------------
def forward_fill_annual(df_y: pd.DataFrame) -> pd.DataFrame:
    """
    Missing values between two available annual values:
    suggested to use the previous year's value -> forward-fill across columns (time axis).

    Applies to annual tables like MV_Y, REV_Y, CO2 Scope 1.
    """
    df = df_y.copy()
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=0, how="all")
    df = _ensure_datetime_sorted_columns(df)
    return df.ffill(axis=1)


# -------------------------------------------------
# 4) Monthly returns + delisting (-100%)
# -------------------------------------------------
def compute_monthly_returns_from_ri(ri_m_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Compute simple monthly returns from RI:
        r_t = RI_t / RI_{t-1} - 1
    """
    ri = _ensure_datetime_sorted_columns(ri_m_clean)
    rets = ri.pct_change(axis=1)
    return rets


def detect_delisted_firms(index: pd.Index) -> pd.Index:
    """
    Heuristic:
    Datastream often appends delisting info/date to the firm name.
    We detect 'DEAD'/'DELIST' and also patterns like 'YYYY-MM-DD' in NAME.
    """
    if not isinstance(index, pd.MultiIndex) or index.nlevels < 2:
        return pd.Index([])

    names = index.get_level_values(1).astype(str)
    mask = (
        names.str.contains(r"DEAD|DELIST", case=False, na=False)
        | names.str.contains(r"\d{4}-\d{2}-\d{2}", case=False, na=False)
    )
    return index[mask]


def apply_delisting_minus_one(
    returns: pd.DataFrame,
    ri_before_low_price: pd.DataFrame,
) -> pd.DataFrame:
    """
    Missing values at end of sample often correspond to default/delisting.
    Rule: price goes to 0 at delisting -> realized return = -100%.

    Implementation:
    - detect delisted firms from NAME patterns
    - find last valid RI in the original RI series (before low-price masking)
    - set return = -1.0 on the next month (if it exists)
    - set returns after that to NaN
    """
    out = returns.copy()
    delisted = detect_delisted_firms(out.index)

    cols = list(out.columns)

    for firm in delisted:
        if firm not in ri_before_low_price.index:
            continue

        prices = ri_before_low_price.loc[firm]
        last_valid = prices.last_valid_index()
        if last_valid is None:
            continue
        if last_valid not in out.columns:
            continue

        pos = cols.index(last_valid)
        if pos + 1 < len(cols):
            delist_month = cols[pos + 1]
            out.loc[firm, delist_month] = -1.0

            if pos + 2 < len(cols):
                out.loc[firm, cols[pos + 2 :]] = np.nan

    return out


# -------------------------------------------------
# Main pipeline (Part I)
# -------------------------------------------------
def clean_part1(raw: Dict[str, pd.DataFrame], cfg: Config) -> Dict[str, pd.DataFrame]:
    """
    Apply Part I Data Cleaning rules to raw inputs.

    Input keys expected:
      RI_M, MV_M, MV_Y, REV_Y, CO2_S1_Y, STATIC  (RI_Y and RF exist but not needed for cleaning core)

    Output keys:
      RI_M_RAW         : RI after dropping missing-price firms (reference universe)
      RI_M_CLEAN       : RI after low-price rule
      RET_M            : monthly returns from RI_M_CLEAN + delisting -100%
      MV_M_CLEAN       : MV_M aligned to RI_M_RAW universe
      MV_Y_CLEAN       : MV_Y aligned + forward-filled
      REV_Y_CLEAN      : REV_Y aligned + forward-filled
      CO2_S1_Y_CLEAN   : CO2 Scope 1 aligned + forward-filled (annual)
      STATIC           : unchanged (region filtering done later in universe.py)
    """
    data = dict(raw)

    # 1) Missing prices -> drop firms from RI_M (reference universe)
    ri_raw = drop_firms_with_no_prices(data["RI_M"])
    data["RI_M_RAW"] = ri_raw

    # 2) Align other tables to RI_M_RAW universe (delete from all tables)
    mv_m = data["MV_M"].apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="all")
    mv_m = _ensure_datetime_sorted_columns(mv_m)
    data["MV_M_CLEAN"] = _align_to_index(mv_m, ri_raw.index)

    mv_y = forward_fill_annual(data["MV_Y"])
    data["MV_Y_CLEAN"] = _align_to_index(mv_y, ri_raw.index)

    rev_y = forward_fill_annual(data["REV_Y"])
    data["REV_Y_CLEAN"] = _align_to_index(rev_y, ri_raw.index)

    # CO2 Scope 1 (annual): forward-fill between reported years + align
    co2_s1 = forward_fill_annual(data["CO2_S1_Y"])
    data["CO2_S1_Y_CLEAN"] = _align_to_index(co2_s1, ri_raw.index)

    # 3) Low price rule on RI (no forward fill)
    ri_clean = apply_low_price_rule(ri_raw, cfg.MIN_PRICE)
    data["RI_M_CLEAN"] = ri_clean

    # 4) Monthly returns
    ret_m = compute_monthly_returns_from_ri(ri_clean)

    # 5) Delisting -100% rule (use RI_M_RAW to find last valid price)
    ret_m = apply_delisting_minus_one(ret_m, ri_before_low_price=ri_raw)
    data["RET_M"] = ret_m

    # STATIC unchanged
    data["STATIC"] = data["STATIC"]

    return data


# -------------------------------------------------
# Quick test: python3 -m src.data_cleaning
# -------------------------------------------------
if __name__ == "__main__":
    from .io import load_raw

    cfg = Config()
    raw = load_raw(cfg)
    clean = clean_part1(raw, cfg)

    print("Keys:", list(clean.keys()))
    print("RI_M_RAW:", clean["RI_M_RAW"].shape)
    print("RI_M_CLEAN:", clean["RI_M_CLEAN"].shape)
    print("RET_M:", clean["RET_M"].shape)
    print("MV_M_CLEAN:", clean["MV_M_CLEAN"].shape)
    print("REV_Y_CLEAN:", clean["REV_Y_CLEAN"].shape)
    print("CO2_S1_Y_CLEAN:", clean["CO2_S1_Y_CLEAN"].shape)
    