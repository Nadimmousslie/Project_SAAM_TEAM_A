# data_cleaning.py — Data cleaning (Section 1 of the project)

import numpy as np
import pandas as pd
from config import LOW_PRICE_THRESHOLD, STALE_THRESHOLD


def clean_all(data: dict) -> dict:
    """
    Apply the cleaning rules from the project statement.
    Returns an updated data dict with cleaned prices, annual data, and returns.
    """
    print("=" * 55)
    print("DATA CLEANING")
    print("=" * 55)

    # Step 1 — Remove ISINs with all monthly prices missing from every table
    valid_isins = _drop_all_nan_isins(data["ri_m"])
    data = _filter_data_tables(data, valid_isins)

    ri_m = data["ri_m"].copy()
    ri_y = data["ri_y"].copy()
    co2 = data["co2"].copy()
    revenue = data.get("revenue")
    if isinstance(revenue, pd.DataFrame):
        revenue = revenue.copy()

    # Step 2 — Low prices → NaN
    ri_m, ri_y = _handle_low_prices(ri_m, ri_y)

    # Step 3 — Only fill gaps inside the price history, not delistings at the end
    ri_m, ri_y = _forward_fill_internal(ri_m, ri_y)

    # Step 4 — Annual accounting data: carry forward the last available observation
    co2, filled_co2 = _forward_fill_annual(co2)
    print(f"  [4] CO2 forward-fill     : {filled_co2} yearly cells filled")

    if revenue is not None:
        revenue, filled_rev = _forward_fill_annual(revenue)
        print(f"  [5] Revenue forward-fill : {filled_rev} yearly cells filled")

    # Step 5 — Compute monthly returns with delisting losses
    returns_m = _compute_returns(ri_m)

    print(f"  Final RI monthly : {ri_m.shape[0]} firms × {ri_m.shape[1]} months")
    print(f"  Final returns_m  : {returns_m.shape[0]} firms × {returns_m.shape[1]} months")
    print("  Cleaning done.\n")

    data["ri_m"] = ri_m
    data["ri_y"] = ri_y
    data["co2"] = co2
    if revenue is not None:
        data["revenue"] = revenue
    data["returns_m"] = returns_m
    return data


# --- Individual cleaning steps ---

def _drop_all_nan_isins(ri_m: pd.DataFrame) -> pd.Index:
    """
    Identify ISINs for which at least one monthly price is available.
    These correspond to firms Datastream could not match.
    """
    before = ri_m.shape[0]
    valid_isins = ri_m.dropna(how="all").index
    after = len(valid_isins)
    print(f"  [1] Drop all-NaN ISINs   : {before} → {after} firms ({before - after} removed)")
    return valid_isins


def _filter_data_tables(data: dict, valid_isins: pd.Index) -> dict:
    """Keep the same valid ISIN universe across all firm-level tables."""
    filtered = data.copy()
    for key in ("ri_m", "ri_y", "mv_m", "mv_y", "co2", "revenue", "static"):
        value = filtered.get(key)
        if isinstance(value, pd.DataFrame):
            filtered[key] = value.reindex(valid_isins)

    if "amer_isins" in filtered:
        filtered["amer_isins"] = [isin for isin in filtered["amer_isins"] if isin in valid_isins]
    return filtered


def _handle_low_prices(ri_m: pd.DataFrame, ri_y: pd.DataFrame):
    """
    Treat RI values below LOW_PRICE_THRESHOLD as missing (NaN).
    Values at 0 or very small are due to rounding and produce
    infinite / extreme returns.
    """
    n_low_m = (ri_m < LOW_PRICE_THRESHOLD).sum().sum()
    n_low_y = (ri_y < LOW_PRICE_THRESHOLD).sum().sum()
    ri_m[ri_m < LOW_PRICE_THRESHOLD] = np.nan
    ri_y[ri_y < LOW_PRICE_THRESHOLD] = np.nan
    print(f"  [2] Low prices → NaN     : {n_low_m} cells in monthly, {n_low_y} in yearly")
    return ri_m, ri_y


def _forward_fill_internal(ri_m: pd.DataFrame, ri_y: pd.DataFrame):
    """
    Forward-fill only internal gaps in the price history.
    Trailing NaNs are preserved to encode delistings/defaults.
    """
    ri_m_filled = _fill_internal_gaps(ri_m)
    ri_y_filled = _fill_internal_gaps(ri_y)

    filled_m = ri_m.isna().sum().sum() - ri_m_filled.isna().sum().sum()
    filled_y = ri_y.isna().sum().sum() - ri_y_filled.isna().sum().sum()
    print(f"  [3] Forward-fill         : {filled_m} cells filled in monthly, {filled_y} in yearly")
    return ri_m_filled, ri_y_filled


def _fill_internal_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill gaps only when a prior and a later observation exist.
    Leading NaNs stay NaN, and trailing NaNs remain available for delisting handling.
    """
    ffilled = df.T.ffill().T
    has_future_obs = df.notna().iloc[:, ::-1].cummax(axis=1).iloc[:, ::-1]
    fill_mask = df.isna() & has_future_obs
    return df.where(~fill_mask, ffilled)


def _forward_fill_annual(df: pd.DataFrame):
    """
    For annual data (CO2, revenue), carry the last available observation forward.
    Leading NaNs remain NaN, which correctly blocks investment until data exists.
    """
    filled = df.T.ffill().T
    n_filled = df.isna().sum().sum() - filled.isna().sum().sum()
    return filled, n_filled


def _compute_returns(ri_m: pd.DataFrame) -> pd.DataFrame:
    """
    Compute simple monthly returns from the total return index.
    R_t = RI_t / RI_{t-1} - 1
    The first missing month after a valid price is treated as a delisting loss of -100%.
    """
    returns_m = ri_m.pct_change(axis=1, fill_method=None)
    delist_mask = ri_m.isna() & ri_m.shift(1, axis=1).notna()
    returns_m = returns_m.mask(delist_mask, -1.0)
    return returns_m


# Stale price check (used per firm per window in investment_set)

def is_stale(return_series: pd.Series, threshold: float = STALE_THRESHOLD) -> bool:
    """
    Return True if the proportion of zero monthly returns exceeds `threshold`.
    Zero returns = no price change = stale / illiquid price.
    """
    valid = return_series.dropna()
    if len(valid) == 0:
        return True
    return (valid == 0).mean() > threshold
