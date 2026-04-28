# =============================================================
# performance.py — Performance Statistics, Plot & Monthly Table
# =============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from config import OUTPUT_PLOT, _SRC_DIR


# ─────────────────────────────────────────────────────────────
# Summary statistics
# ─────────────────────────────────────────────────────────────

def compute_stats(ret: pd.Series, rf: pd.Series, label: str) -> dict:
    rf_aligned = rf.reindex(ret.index).fillna(0)
    excess     = ret - rf_aligned
    mu_ann     = ret.mean()  * 12
    sigma_ann  = ret.std()   * np.sqrt(12)
    sr         = excess.mean() / ret.std() * np.sqrt(12) if ret.std() > 0 else np.nan
    return {
        "label"  : label,
        "mu_p"   : mu_ann,
        "sigma_p": sigma_ann,
        "SR_p"   : sr,
        "min"    : ret.min(),
        "max"    : ret.max(),
        "n_obs"  : len(ret),
    }


def print_stats_table(stats_mv: dict, stats_vw: dict) -> None:
    metrics = [
        ("Annualised avg return", "mu_p",    "{:.2%}"),
        ("Annualised volatility", "sigma_p", "{:.2%}"),
        ("Sharpe ratio",          "SR_p",    "{:.4f}"),
        ("Min monthly return",    "min",     "{:.2%}"),
        ("Max monthly return",    "max",     "{:.2%}"),
        ("N observations",        "n_obs",   "{:d}"),
    ]
    col_w = 28

    print("=" * (col_w * 3))
    print("PERFORMANCE SUMMARY")
    print("=" * (col_w * 3))
    header = (f"\n{'Metric':<{col_w}}"
              f"{'Min-Variance':>{col_w}}"
              f"{'Value-Weighted':>{col_w}}")
    print(header)
    print("-" * (col_w * 3))

    for name, key, fmt in metrics:
        v_mv = fmt.format(stats_mv[key])
        v_vw = fmt.format(stats_vw[key])
        print(f"  {name:<{col_w - 2}}{v_mv:>{col_w}}{v_vw:>{col_w}}")

    print("=" * (col_w * 3) + "\n")


# ─────────────────────────────────────────────────────────────
# Monthly returns table
# ─────────────────────────────────────────────────────────────

def build_monthly_table(mv_ret: pd.Series, vw_ret: pd.Series) -> pd.DataFrame:
    common = mv_ret.index.union(vw_ret.index)
    df = pd.DataFrame({
        "MV_Return": mv_ret.reindex(common),
        "VW_Return": vw_ret.reindex(common),
    }, index=common)
    df.index.name = "Date"
    return df.sort_index()


def export_monthly_table(df: pd.DataFrame) -> None:
    """Export monthly returns to Excel only — no terminal print."""
    output_path = str(_SRC_DIR.parent / "monthly_portfolio_returns.xlsx")
    export = df.copy().reset_index()
    export["Date"] = export["Date"].dt.strftime("%Y-%m-%d")

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        export.to_excel(writer, index=False, sheet_name="Monthly Returns")
        ws = writer.sheets["Monthly Returns"]
        for col in ws.columns:
            max_len = max(len(str(cell.value)) for cell in col if cell.value)
            ws.column_dimensions[col[0].column_letter].width = max_len + 4

    print(f"  ✓ Monthly returns exported → {output_path}\n")


# ─────────────────────────────────────────────────────────────
# Cumulative return plot
# ─────────────────────────────────────────────────────────────

def plot_cumulative_returns(mv_ret: pd.Series,
                            vw_ret: pd.Series,
                            output_path: str = OUTPUT_PLOT) -> None:
    common = mv_ret.index.intersection(vw_ret.index)
    cum_mv = (1 + mv_ret.loc[common]).cumprod()
    cum_vw = (1 + vw_ret.loc[common]).cumprod()

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(cum_mv.index, cum_mv.values,
            label="Min-Variance  $P^{mv}_{oos}$",
            linewidth=2.0, color="steelblue")
    ax.plot(cum_vw.index, cum_vw.values,
            label="Value-Weighted  $P^{vw}$",
            linewidth=2.0, color="darkorange", linestyle="--")

    ax.set_title("Cumulative Returns — AMER Region (2014–2025)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Cumulative Return  (base = 1 at start)", fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  ✓ Plot saved → {output_path}\n")


# ─────────────────────────────────────────────────────────────
# Convenience wrapper
# ─────────────────────────────────────────────────────────────

def run_performance(mv_ret: pd.Series, vw_ret: pd.Series, rf: pd.Series) -> None:
    """Compute stats, plot cumulative returns, export monthly table (no terminal print)."""

    print("=" * 55)
    print("PERFORMANCE ANALYSIS")
    print("=" * 55)

    stats_mv = compute_stats(mv_ret, rf, "Min-Variance")
    stats_vw = compute_stats(vw_ret, rf, "Value-Weighted")
    print_stats_table(stats_mv, stats_vw)

    plot_cumulative_returns(mv_ret, vw_ret)

    # Export to Excel only — no terminal print (fast)
    monthly_df = build_monthly_table(mv_ret, vw_ret)
    export_monthly_table(monthly_df)


