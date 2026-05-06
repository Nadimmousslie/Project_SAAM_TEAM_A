# data_loader.py — Load & parse all Datastream Excel files

import pandas as pd
from config import PATHS, REGION


def _load_ds_wide(path, sheet, sheet_candidates=None):
    """
    Generic loader for wide Datastream files.
    - Drops the $$ER error row
    - Sets ISIN as index
    - Converts column headers to pd.Timestamp where possible
    Returns: DataFrame (ISIN × dates)
    """
    workbook = pd.ExcelFile(path)
    candidates = [sheet, *(sheet_candidates or [])]
    selected_sheet = next(
        (candidate for candidate in candidates if candidate in workbook.sheet_names),
        workbook.sheet_names[0],
    )
    df = pd.read_excel(workbook, sheet_name=selected_sheet, index_col=None)

    # Drop rows with no ISIN or error-code ISIN
    df = df[df["ISIN"].notna()].copy()
    df = df[~df["ISIN"].astype(str).str.startswith("$$")].copy()

    df = df.drop(columns=["NAME"], errors="ignore")
    df = df.set_index("ISIN")

    # Convert column names to Timestamps (handles both datetime and int-year)
    new_cols = []
    for c in df.columns:
        if isinstance(c, pd.Timestamp):
            new_cols.append(c)
        else:
            try:
                # integer year → December 31 of that year
                year = int(float(str(c)))
                new_cols.append(pd.Timestamp(f"{year}-12-31"))
            except Exception:
                new_cols.append(c)
    df.columns = new_cols

    # Keep only Timestamp columns
    df = df[[c for c in df.columns if isinstance(c, pd.Timestamp)]]
    return df


def load_all(region=REGION):
    """
    Load every dataset, filter to the requested region, and return a dict.

    Returns
    -------
    data : dict with keys:
        'ri_m'    – RI monthly  (firms × months)
        'ri_y'    – RI yearly   (firms × years)
        'mv_m'    – Market-cap monthly
        'mv_y'    – Market-cap yearly
        'co2'     – CO2 Scope-1 yearly
        'rf'      – Risk-free rate (pd.Series, monthly)
        'static'  – Static info DataFrame (ISIN index)
        'amer_isins' – list of ISINs in the region
    """
    print("=" * 55)
    print("LOADING DATA")
    print("=" * 55)

    # Static file
    static = pd.read_excel(PATHS["static"]).set_index("ISIN")
    amer_isins = static[static["Region"] == region].index.tolist()
    print(f"  Region '{region}' → {len(amer_isins)} firms")

    # RI monthly
    ri_m = _load_ds_wide(PATHS["ri_monthly"], sheet="RI")
    ri_m = ri_m[ri_m.index.isin(amer_isins)]
    ri_m = ri_m.apply(pd.to_numeric, errors="coerce")
    print(f"  RI monthly  : {ri_m.shape[0]} firms × {ri_m.shape[1]} months")

    # RI yearly
    ri_y = _load_ds_wide(PATHS["ri_yearly"], sheet="RI")
    ri_y = ri_y[ri_y.index.isin(amer_isins)]
    ri_y = ri_y.apply(pd.to_numeric, errors="coerce")
    print(f"  RI yearly   : {ri_y.shape[0]} firms × {ri_y.shape[1]} years")

    # Market-cap monthly
    mv_m = _load_ds_wide(PATHS["mv_monthly"], sheet="MV")
    mv_m = mv_m[mv_m.index.isin(amer_isins)]
    mv_m = mv_m.apply(pd.to_numeric, errors="coerce")
    print(f"  MV monthly  : {mv_m.shape[0]} firms × {mv_m.shape[1]} months")

    # Market-cap yearly
    mv_y = _load_ds_wide(PATHS["mv_yearly"], sheet="MV")
    mv_y = mv_y[mv_y.index.isin(amer_isins)]
    mv_y = mv_y.apply(pd.to_numeric, errors="coerce")
    print(f"  MV yearly   : {mv_y.shape[0]} firms × {mv_y.shape[1]} years")

    # CO2 Scope 1
    co2 = _load_ds_wide(PATHS["co2"], sheet="Scope1", sheet_candidates=["SCOPE1"])
    co2 = co2[co2.index.isin(amer_isins)]
    co2 = co2.apply(pd.to_numeric, errors="coerce")
    print(f"  CO2 Scope-1 : {co2.shape[0]} firms × {co2.shape[1]} years")

    # Revenue
    revenue = _load_ds_wide(
        PATHS["revenue"],
        sheet="Revenue",
        sheet_candidates=["REV", "Revenues", "Revenue USD"],
    )
    revenue = revenue[revenue.index.isin(amer_isins)]
    revenue = revenue.apply(pd.to_numeric, errors="coerce")
    print(f"  Revenue     : {revenue.shape[0]} firms × {revenue.shape[1]} years")

    # Risk-free rate
    rf_raw = pd.read_excel(PATHS["risk_free"])
    rf_raw.columns = ["date", "RF"]
    rf_raw["date"] = pd.to_datetime(rf_raw["date"].astype(str), format="%Y%m")
    rf_raw["date"] = rf_raw["date"] + pd.offsets.MonthEnd(0)
    rf = rf_raw.set_index("date")["RF"] / 100   # % → decimal
    print(f"  Risk-free   : {rf.index[0].date()} → {rf.index[-1].date()}")

    print("  ✓ All files loaded.\n")

    return {
        "ri_m"      : ri_m,
        "ri_y"      : ri_y,
        "mv_m"      : mv_m,
        "mv_y"      : mv_y,
        "co2"       : co2,
        "revenue"   : revenue,
        "rf"        : rf,
        "static"    : static,
        "amer_isins": amer_isins,
    }
