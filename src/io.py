# src/io.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

from .config import Config


def read_datastream_excel(
    path: Path,
    index_cols: Tuple[int, ...] = (0, 1),
    sheet_name: str | int = 0,
) -> pd.DataFrame:
    """
    Read a Datastream Excel export.

    Expected format:
    - rows: firms (MultiIndex like (ISIN, NAME))
    - columns: dates (monthly or annual)
    """
    df = pd.read_excel(path, sheet_name=sheet_name, index_col=list(index_cols))
    df = df.dropna(axis=1, how="all")

    # Try to parse columns as datetimes and sort
    cols_dt = pd.to_datetime(df.columns, errors="coerce")
    if cols_dt.notna().mean() > 0.8:
        df.columns = cols_dt
        df = df.loc[:, df.columns.notna()]
        df = df.sort_index(axis=1)

    return df


def load_raw(cfg: Config) -> Dict[str, pd.DataFrame]:
    """
    Load raw inputs used in Part I.
    Returns a dict with:
      RI_M, RI_Y, MV_M, MV_Y, REV_Y, CO2_S1_Y, STATIC, RF
    """
    # Ensure outputs folder exists (safe)
    cfg.OUT.mkdir(parents=True, exist_ok=True)

    # Basic file existence check (clear error messages)
    required = [
        cfg.RI_M_FILE,
        cfg.RI_Y_FILE,
        cfg.MV_M_FILE,
        cfg.MV_Y_FILE,
        cfg.REV_Y_FILE,
        cfg.CO2_S1_Y_FILE,
        cfg.STATIC_FILE,
        cfg.RF_FILE,
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing input file(s):\n" + "\n".join(str(p) for p in missing)
        )

    data: Dict[str, pd.DataFrame] = {}
    data["RI_M"] = read_datastream_excel(cfg.RI_M_FILE)
    data["RI_Y"] = read_datastream_excel(cfg.RI_Y_FILE)
    data["MV_M"] = read_datastream_excel(cfg.MV_M_FILE)
    data["MV_Y"] = read_datastream_excel(cfg.MV_Y_FILE)
    data["REV_Y"] = read_datastream_excel(cfg.REV_Y_FILE)
    data["CO2_S1_Y"] = read_datastream_excel(cfg.CO2_S1_Y_FILE).apply(
        pd.to_numeric, errors="coerce"
    )

    # Static & risk-free are typically not Datastream "time-column" tables
    data["STATIC"] = pd.read_excel(cfg.STATIC_FILE)
    data["RF"] = pd.read_excel(cfg.RF_FILE)

    return data


# -------------------------------------------------
# Test runner (stable): python3 -m src.io
# -------------------------------------------------
if __name__ == "__main__":
    cfg = Config()
    d = load_raw(cfg)
    print("Loaded keys:", list(d.keys()))
    print("RI_M shape:", d["RI_M"].shape)
    print("MV_M shape:", d["MV_M"].shape)
    print("CO2_S1_Y shape:", d["CO2_S1_Y"].shape)
    print("STATIC shape:", d["STATIC"].shape)