from pathlib import Path
import pandas as pd

RAW = Path("data/raw")
CLEANED = Path("data/cleaned")
CLEANED.mkdir(parents=True, exist_ok=True)

files = {
    "RI_M": ("DS_RI_T_USD_M_2025.xlsx", [0, 1]),
    "MV_M": ("DS_MV_T_USD_M_2025.xlsx", [0, 1]),
    "CO2_Y": ("DS_CO2_SCOPE_1_Y_2025.xlsx", [0, 1]),
    "STATIC": ("Static_2025.xlsx", None),
}

def read_ds_excel(path: Path, index_cols):
    if index_cols is None:
        return pd.read_excel(path)
    df = pd.read_excel(path, index_col=index_cols)
    df = df.dropna(axis=1, how="all")
    cols_dt = pd.to_datetime(df.columns, errors="coerce")
    if cols_dt.notna().mean() > 0.8:
        df.columns = cols_dt
        df = df.loc[:, df.columns.notna()].sort_index(axis=1)
    return df

for key, (fname, idx) in files.items():
    src = RAW / fname
    if not src.exists():
        raise FileNotFoundError(f"Missing input: {src}")
    df = read_ds_excel(src, idx)
    out = CLEANED / f"{key}.csv"
    if key == "STATIC":
        df.to_csv(out, index=False)
    else:
        df.to_csv(out)
    print("saved", out)
