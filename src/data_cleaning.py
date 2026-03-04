### data_cleaning.py

from __future__ import annotations

from typing import Dict
import re
import numpy as np
import pandas as pd

from .config import Config


# =========================================================
# Helpers
# =========================================================
def _ensure_time_sorted(df: pd.DataFrame) -> pd.DataFrame:
    """Sort time columns (dates) if they are datetime-like."""
    df2 = df.copy()
    try:
        # if columns are datetime already, this works
        df2 = df2.sort_index(axis=1)
    except Exception:
        pass
    return df2


def drop_firms_with_no_prices(ri: pd.DataFrame) -> pd.DataFrame:
    """
    Missing prices: Datastream couldn't match a firm => full row missing.
    Rule: delete from all tables.
    Here we only drop from RI; the alignment across tables is handled later.
    """
    return ri.dropna(axis=0, how="all")


def apply_low_price_rule(ri: pd.DataFrame, min_price: float) -> pd.DataFrame:
    """
    Low prices: treat RI < 0.5 as missing (NaN).
    IMPORTANT: do NOT forward-fill prices (keep NaNs for investability rules in universe.py).
    """
    ri2 = ri.copy()
    ri2 = ri2.apply(pd.to_numeric, errors="coerce")
    ri2 = ri2.mask(ri2 < min_price)
    return ri2


def compute_returns_from_ri(ri_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Compute simple monthly returns from RI.
    Datastream format used here: rows=firms, columns=monthly dates.
    """
    ri_clean = _ensure_time_sorted(ri_clean)
    rets = ri_clean.pct_change(axis=1)
    return rets


def ffill_annual_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Missing values between two available values (annual data):
    use previous year number => forward-fill in time.
    Safe for CO2 / Revenues. Not for prices.
    """
    df2 = df.copy()
    df2 = df2.apply(pd.to_numeric, errors="coerce")
    df2 = df2.dropna(axis=0, how="all")
    df2 = _ensure_time_sorted(df2)
    return df2.ffill(axis=1)


def align_firms_to_reference(df: pd.DataFrame, ref_index: pd.Index) -> pd.DataFrame:
    """Keep only firms present in ref_index."""
    # Some tables may have same index type (MultiIndex with ISIN/NAME)
    common = df.index.intersection(ref_index)
    return df.loc[common]


# =========================================================
# Delisting rule (-100%)
# =========================================================
def detect_delisted_firms_from_index(index: pd.Index) -> pd.Index:
    """
    Heuristic: Datastream sometimes appends delisting info/date to the NAME.
    We detect 'DEAD'/'DELIST' and also patterns like '(YYYY-MM-DD)' or 'YYYY-MM-DD'.
    """
    if not isinstance(index, pd.MultiIndex):
        return pd.Index([])

    names = index.get_level_values(1).astype(str)

    pat_words = names.str.contains(r"DEAD|DELIST", case=False, na=False)
    pat_date = names.str.contains(r"\d{4}-\d{2}-\d{2}", case=False, na=False)

    mask = pat_words | pat_date
    return index[mask]


def apply_delisting_minus_one(
    returns: pd.DataFrame,
    ri_before_low_price: pd.DataFrame,
) -> pd.DataFrame:
    """
    Missing values at the end of the sample often correspond to default/delisting.
    Rule: price goes to 0 at delisting => realized return -100% at delisting time.

    Implementation (heuristic):
    - detect delisted firms from index name patterns
    - for each delisted firm:
        * find last valid RI observation in the original RI series (before low-price masking)
        * set return = -1.0 on the next month (if it exists)
        * set returns after that to NaN
    """
    out = returns.copy()
    delisted = detect_delisted_firms_from_index(out.index)

    for firm in delisted:
        if firm not in ri_before_low_price.index:
            continue

        prices = ri_before_low_price.loc[firm]
        last_valid = prices.last_valid_index()
        if last_valid is None:
            continue

        # returns columns are the same time axis as RI columns
        cols = list(out.columns)
        if last_valid not in out.columns:
            # if last_valid is not in returns columns, skip
            continue

        pos = cols.index(last_valid)
        if pos + 1 < len(cols):
            delist_month = cols[pos + 1]
            out.loc[firm, delist_month] = -1.0

            # After delisting month, set to NaN
            if pos + 2 < len(cols):
                out.loc[firm, cols[pos + 2 :]] = np.nan

    return out


# =========================================================
# Main cleaning pipeline for Part I
# =========================================================
def clean_part1_data(raw: Dict[str, pd.DataFrame], cfg: Config) -> Dict[str, pd.DataFrame]:
    """
    Part I cleaning pipeline (static cleaning only):
    Inputs expected in raw:
        - RI_M (monthly RI)
        - MV_M (monthly market value)
        - CO2_Y (annual CO2 scope 1)
        - STATIC (static info)

    Outputs:
        - RI_M_RAW (after dropping full-missing rows, numeric)
        - RI_M_CLEAN (low-price rule applied)
        - RET_M (monthly returns from RI_M_CLEAN + delisting -100% rule applied)
        - MV_M_CLEAN (aligned to RI firms, numeric)
        - CO2_Y_CLEAN (ffill annual)
        - STATIC (unchanged)
    """
    data = dict(raw)

    # --- 1) RI: drop firms with no prices (missing prices)
    ri_raw = data["RI_M"].copy()
    ri_raw = ri_raw.apply(pd.to_numeric, errors="coerce")
    ri_raw = drop_firms_with_no_prices(ri_raw)
    ri_raw = _ensure_time_sorted(ri_raw)
    data["RI_M_RAW"] = ri_raw

    # --- 2) Align MV and CO2 to remaining firms (delete from all tables)
    data["MV_M_CLEAN"] = align_firms_to_reference(
        data["MV_M"].apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="all"),
        ri_raw.index,
    )
    data["MV_M_CLEAN"] = _ensure_time_sorted(data["MV_M_CLEAN"])

    data["CO2_Y_CLEAN"] = align_firms_to_reference(
        ffill_annual_table(data["CO2_Y"]),
        ri_raw.index,
    )

    # --- 3) Low prices rule on RI
    ri_clean = apply_low_price_rule(ri_raw, cfg.MIN_PRICE)
    data["RI_M_CLEAN"] = ri_clean

    # --- 4) Monthly returns
    ret_m = compute_returns_from_ri(ri_clean)

    # --- 5) Delisting -100% rule (use RI_M_RAW to find last valid price)
    ret_m = apply_delisting_minus_one(ret_m, ri_before_low_price=ri_raw)

    data["RET_M"] = ret_m

    # STATIC unchanged
    data["STATIC"] = data["STATIC"]

    return data