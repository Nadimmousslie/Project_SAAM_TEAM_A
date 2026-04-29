### investment_set.py — Section 2.1: Investment Set per Year ###

import pandas as pd
from config import TAU, MIN_MONTHS_DATA, REBALANCE_YEARS
from data_cleaning import is_stale


# ─────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────

def _get_dec_col(df: pd.DataFrame, year: int):
    """
    Return the column of `df` corresponding to December of `year`.
    Returns None if not found.
    """
    cols = pd.DatetimeIndex(df.columns)
    dec_cols = cols[(cols.year == year) & (cols.month == 12)]
    return dec_cols[-1] if len(dec_cols) > 0 else None


# ─────────────────────────────────────────────────────────────
# Main function
# ─────────────────────────────────────────────────────────────

def build_investment_sets(data: dict) -> dict:
    """
    For each rebalancing year Y, determine the set of eligible firms.

    Eligibility criteria:
      1. Firm belongs to the assigned region (already filtered in data_loader)
      2. CO2 data available at end of year Y
      3. RI (price) not missing at end of year Y (i.e. not low price)
      4. At least MIN_MONTHS_DATA months of valid returns in the 10-year window
      5. Firm is not subject to stale prices (≤ 50% zero returns in window)

    Returns
    -------
    invest_sets : dict  {year → list of eligible ISINs}
    ret_windows : dict  {year → DataFrame of returns in estimation window}
    """
    print("=" * 55)
    print("BUILDING INVESTMENT SETS")
    print("=" * 55)

    returns_m = data["returns_m"]
    ri_m      = data["ri_m"]
    co2       = data["co2"]

    invest_sets = {}
    ret_windows = {}

    for Y in REBALANCE_YEARS:
        dec_Y = _get_dec_col(returns_m, Y)
        if dec_Y is None:
            print(f"  {Y}: no December returns column — skipped.")
            continue

        # ── Estimation window: TAU months ending at Dec Y ────
        all_cols   = pd.DatetimeIndex(returns_m.columns)
        window_cols = all_cols[all_cols <= dec_Y][-TAU:]
        ret_window  = returns_m[window_cols]

        # ── Criterion 2: CO2 available at end of Y ───────────
        co2_dec = _get_dec_col(co2, Y)
        if co2_dec is not None:
            has_co2 = co2[co2_dec].notna()
        else:
            has_co2 = pd.Series(False, index=co2.index)

        # ── Criterion 3: price not missing at end of Y ───────
        ri_dec = _get_dec_col(ri_m, Y)
        if ri_dec is not None:
            has_price = ri_m[ri_dec].notna()
        else:
            has_price = pd.Series(False, index=ri_m.index)

        # ── Criterion 4: enough valid monthly observations ────
        n_valid     = ret_window.notna().sum(axis=1)
        enough_data = n_valid >= MIN_MONTHS_DATA

        # ── Criterion 5: not stale ────────────────────────────
        not_stale = ret_window.apply(
            lambda row: not is_stale(row), axis=1
        )

        # ── Combine ───────────────────────────────────────────
        eligible = (
            has_co2.reindex(ret_window.index,  fill_value=False)
            & has_price.reindex(ret_window.index, fill_value=False)
            & enough_data
            & not_stale
        )

        isins_Y = eligible[eligible].index.tolist()
        invest_sets[Y] = isins_Y
        ret_windows[Y] = ret_window

        print(f"  {Y}: {len(isins_Y):>4} eligible firms "
              f"(window: {window_cols[0].date()} → {window_cols[-1].date()})")

    print(f"  ✓ Investment sets built for {len(invest_sets)} years.\n")
    return invest_sets, ret_windows

