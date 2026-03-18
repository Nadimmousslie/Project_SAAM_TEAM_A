# =============================================================
# config.py — Global parameters & file paths
# =============================================================

from pathlib import Path

# Chemin absolu du dossier contenant config.py (= src/)
_SRC_DIR = Path(__file__).resolve().parent

# Remonte d'un niveau (TESTSAAM/) puis descend dans data/raw/
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

# ── Region ────────────────────────────────────────────────────
REGION = "AMER"

# ── Estimation window ─────────────────────────────────────────
ESTIMATION_YEARS = 10
TAU              = ESTIMATION_YEARS * 12   # = 120 months

# ── Rebalancing years ─────────────────────────────────────────
REBALANCE_YEARS = list(range(2013, 2025))  # 2013 → 2024

# ── Data-cleaning thresholds ──────────────────────────────────
LOW_PRICE_THRESHOLD = 0.5    # RI below this → treated as missing
STALE_THRESHOLD     = 0.50   # proportion of zero returns → stale
MIN_MONTHS_DATA     = 36     # minimum valid monthly observations

# ── Output ────────────────────────────────────────────────────
OUTPUT_PLOT = str(_SRC_DIR.parent / "cumulative_returns.png")

