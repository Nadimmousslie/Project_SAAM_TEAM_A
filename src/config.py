# src/config.py
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Config:
    # ---------- Paths ----------
    ROOT_DIR: Path = Path(__file__).resolve().parents[1]
    DATA_DIR: Path = ROOT_DIR / "data"
    RAW_DIR: Path = DATA_DIR / "raw"
    CLEANED_DIR: Path = DATA_DIR / "cleaned"

    OUTPUT_DIR: Path = ROOT_DIR / "outputs"
    FIGURES_DIR: Path = OUTPUT_DIR / "figures"
    TABLES_DIR: Path = OUTPUT_DIR / "tables"

    # ---------- Part I horizon ----------
    START_OOS: str = "2014-01-01"
    END_OOS: str = "2025-12-31"

    # Annual loop: weights computed at end of Y applied to year Y+1
    START_YEAR: int = 2013  # -> invest 2014
    END_YEAR: int = 2024    # -> invest 2025

    # ---------- Cleaning rules ----------
    MIN_PRICE: float = 0.5
    STALE_THRESHOLD: float = 0.50

    # ---------- Estimation window ----------
    WINDOW_MONTHS: int = 120
    MIN_MONTHS_IN_WINDOW: int = 36

    # ---------- Optional ----------
    # IMPORTANT: keep None while debugging universe size.
    # Later, set to "USA" once we confirm the correct country column/value in Static_2025.xlsx.
    REGION: str | None = None

    # If you want to enforce country filtering later, set:
    # REGION = "USA"