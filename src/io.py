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
    Read a Datastream Excel output.

    Typical format:
    - index columns: identifiers (e.g., ISIN, NAME)
    - remaining columns: dates (monthly/annual)
    """
    df = pd.read_excel(path, sheet_name=sheet_name, index_col=list(index_cols))
    df = df.dropna(axis=1, how="all")

    # Convert & sort time columns when possible
    cols_dt = pd.to_datetime(df.columns, errors="coerce")
    if cols_dt.notna().mean() > 0.8:
        df.columns = cols_dt
        df = df.loc[:, df.columns.notna()]
        df = df.sort_index(axis=1)

    return df


def load_part1_inputs(cfg: Config, include_rf: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Part I minimal inputs:
    - RI_M: monthly total return index
    - MV_M: monthly market value (for value-weighted benchmark)
    - CO2_Y: annual CO2 scope 1 (used only as "carbon data available" filter)
    - STATIC: static firm info (region, name, etc.)
    - RF (optional)
    """
    paths = {
        "RI_M": cfg.RAW_DIR / "DS_RI_T_USD_M_2025.xlsx",
        "MV_M": cfg.RAW_DIR / "DS_MV_T_USD_M_2025.xlsx",
        "CO2_Y": cfg.RAW_DIR / "DS_CO2_SCOPE_1_Y_2025.xlsx",
        "STATIC": cfg.RAW_DIR / "Static_2025.xlsx",
    }

    if include_rf:
        paths["RF"] = cfg.RAW_DIR / "Risk_Free_Rate_2025.xlsx"

    # Check files exist
    for k, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {k} -> {p}")

    data: Dict[str, pd.DataFrame] = {}

    data["RI_M"] = read_datastream_excel(paths["RI_M"], index_cols=(0, 1))
    data["MV_M"] = read_datastream_excel(paths["MV_M"], index_cols=(0, 1))

    # CO2: force numeric (some cells are strings)
    data["CO2_Y"] = read_datastream_excel(paths["CO2_Y"], index_cols=(0, 1)).apply(
        pd.to_numeric, errors="coerce"
    )

    # Static: often not a timeseries table -> read as plain excel
    data["STATIC"] = pd.read_excel(paths["STATIC"])

    if include_rf:
        data["RF"] = pd.read_excel(paths["RF"])

    return data