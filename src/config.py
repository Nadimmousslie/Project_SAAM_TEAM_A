### src/config.py

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    """
    Central configuration for Part I.

    Why this file exists:
    - Single source of truth for file paths and key parameters.
    - Avoids hard-coding values across multiple files.
    - Makes the project reproducible and easier to debug.

    Edit this file if:
    - you rename/move input files in data/raw/
    - you want to change the backtest window or cleaning thresholds
    - your assigned region changes
    """

    # -------------------------
    # Project paths
    # -------------------------
    # Project root = parent folder of src/
    ROOT: Path = Path(__file__).resolve().parents[1]

    # Raw inputs (Datastream Excel files) are stored here
    RAW: Path = ROOT / "data" / "raw"

    # All outputs (CSV results + plots) will be saved here
    OUT: Path = ROOT / "outputs"

    # -------------------------
    # Input files (Datastream)
    # -------------------------
    # RI = Return Index (monthly and annual versions)
    RI_M_FILE: Path = RAW / "DS_RI_T_USD_M_2025.xlsx"
    RI_Y_FILE: Path = RAW / "DS_RI_T_USD_Y_2025.xlsx"

    # MV = Market Value (monthly and annual versions)
    MV_M_FILE: Path = RAW / "DS_MV_T_USD_M_2025.xlsx"
    MV_Y_FILE: Path = RAW / "DS_MV_T_USD_Y_2025.xlsx"

    # Revenues (annual)
    REV_Y_FILE: Path = RAW / "DS_REV_Y_2025.xlsx"

    # Carbon emissions (annual) — Group A uses Scope 1
    CO2_S1_Y_FILE: Path = RAW / "DS_CO2_SCOPE_1_Y_2025.xlsx"

    # Static firm information (region/country classifications, identifiers, etc.)
    STATIC_FILE: Path = RAW / "Static_2025.xlsx"

    # Risk-free rate (optional for Part I; may be used later for Sharpe)
    RF_FILE: Path = RAW / "Risk_Free_Rate_2025.xlsx"

    # -------------------------
    # Part I parameters (per instructions)
    # -------------------------
    # Low prices rule: treat RI < 0.5 as missing to avoid extreme returns
    MIN_PRICE: float = 0.5

    # Estimation window: 10 years of monthly data = 120 months
    WINDOW_MONTHS: int = 120

    # Minimum required observations inside the 10-year window (e.g., 3 years = 36 months)
    MIN_MONTHS: int = 36

    # Stale prices rule: if share of ~0 monthly returns in the window > 50%, exclude asset
    STALE_THRESHOLD: float = 0.50

    # Numerical tolerance used when checking "return == 0"
    EPS_ZERO: float = 1e-12

    # Backtest timeline:
    # We compute weights at end of year Y and apply them during year Y+1.
    START_YEAR: int = 2013  # weights end-2013 -> invest 2014
    END_YEAR: int = 2024    # weights end-2024 -> invest 2025

    # -------------------------
    # Assigned investment universe
    # -------------------------
    # Your project group region filter (applied via Static_2025.xlsx)
    REGION: str | None = "AMER"