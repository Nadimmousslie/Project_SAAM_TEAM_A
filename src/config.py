# config.py — Global parameters & file paths

from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent

DATA_DIR = _SRC_DIR.parent / "data" / "raw"

PATHS = {
    "ri_monthly" : DATA_DIR / "DS_RI_T_USD_M_2025.xlsx",
    "ri_yearly"  : DATA_DIR / "DS_RI_T_USD_Y_2025.xlsx",
    "mv_monthly" : DATA_DIR / "DS_MV_T_USD_M_2025.xlsx",
    "mv_yearly"  : DATA_DIR / "DS_MV_T_USD_Y_2025.xlsx",
    "co2"        : DATA_DIR / "DS_CO2_SCOPE_1_Y_2025.xlsx",
    "static"     : DATA_DIR / "Static_2025.xlsx",
    "risk_free"  : DATA_DIR / "Risk_Free_Rate_2025.xlsx",
    "revenue"    : DATA_DIR / "DS_REV_Y_2025.xlsx",
}

# REGION SELECTION 
REGION = "AMER"

# ESTIMATION WINDOW 
ESTIMATION_YEARS = 10
TAU              = ESTIMATION_YEARS * 12   # = 120 months

# REBALANCING YEARS 
REBALANCE_YEARS = list(range(2013, 2025))  # 2013 → 2024

# DATA-CLEANING THRESHOLDS 
LOW_PRICE_THRESHOLD = 0.5    # RI below this → treated as missing
STALE_THRESHOLD     = 0.50   # proportion of zero returns → stale
MIN_MONTHS_DATA     = 36     # minimum valid monthly observations

# OUTPUT PATHS
OUTPUT_PLOT = str(_SRC_DIR.parent / "cumulative_returns.png")
