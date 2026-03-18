# =============================================================
# performance.py — Section 2.3: Performance Statistics & Plot
# =============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from config import OUTPUT_PLOT, REGION


# ─────────────────────────────────────────────────────────────
# Summary statistics
# ─────────────────────────────────────────────────────────────

def compute_stats(ret: pd.Series, rf: pd.Series, label: str) -> dict:
    """
    Compute annualised performance metrics for a monthly return series.

    Metrics
    -------
    mu_p    : annualised average return
    sigma_p : annualised volatility
    SR_p    : annualised Sharpe ratio  (excess return / vol)
    min     : minimum monthly return
    max     : maximum monthly return
    """
    rf_aligned  = rf.reindex(ret.index).fillna(0)
    excess      = ret - rf_aligned

    mu_ann    = ret.mean()    * 12
    sigma_ann = ret.std()     * np.sqrt(12)
    sr        = (excess.mean() / ret.std() * np.sqrt(12)
                 if ret.std() > 0 else np.nan)

    stats = {
        "label"  : label,
        "mu_p"   : mu_ann,
        "sigma_p": sigma_ann,
        "SR_p"   : sr,
        "min"    : ret.min(),
        "max"    : ret.max(),
        "n_obs"  : len(ret),
    }
    return stats


def print_stats_table(stats_mv: dict, stats_vw: dict) -> None:
    """Pretty-print a comparison table of the two portfolios."""
    metrics = [
        ("Annualised avg return", "mu_p",    "{:.2%}"),
        ("Annualised volatility", "sigma_p", "{:.2%}"),
        ("Sharpe ratio",          "SR_p",    "{:.4f}"),
        ("Min monthly return",    "min",     "{:.2%}"),
        ("Max monthly return",    "max",     "{:.2%}"),
        ("N observations",        "n_obs",   "{:d}"),
    ]
    col_w = 28

    header = (f"\n{'Metric':<{col_w}}"
              f"{'Min-Variance':>{col_w}}"
              f"{'Value-Weighted':>{col_w}}")
    print("=" * (col_w * 3))
    print("PERFORMANCE SUMMARY")
    print("=" * (col_w * 3))
    print(header)
    print("-" * (col_w * 3))

    for name, key, fmt in metrics:
        v_mv = fmt.format(stats_mv[key])
        v_vw = fmt.format(stats_vw[key])
        print(f"  {name:<{col_w - 2}}{v_mv:>{col_w}}{v_vw:>{col_w}}")

    print("=" * (col_w * 3) + "\n")


# ─────────────────────────────────────────────────────────────
# Cumulative return plot
# ─────────────────────────────────────────────────────────────

def plot_cumulative_returns(mv_ret: pd.Series,
                            vw_ret: pd.Series,
                            output_path: str = OUTPUT_PLOT) -> None:
    """
    Plot cumulative returns of the MV and VW portfolios on a shared date axis.
    Saves the figure to `output_path`.
    """
    common = mv_ret.index.intersection(vw_ret.index)
    if common.empty:
        raise ValueError("No overlapping dates between MV and VW return series.")

    cum_mv = (1 + mv_ret.loc[common]).cumprod()
    cum_vw = (1 + vw_ret.loc[common]).cumprod()
    start_year = common[0].year
    end_year = common[-1].year

    fig, ax = plt.subplots(figsize=(13, 6))

    ax.plot(cum_mv.index, cum_mv.values,
            label="Min-Variance  $P^{mv}_{oos}$",
            linewidth=2.0, color="steelblue")
    ax.plot(cum_vw.index, cum_vw.values,
            label="Value-Weighted  $P^{vw}$",
            linewidth=2.0, color="darkorange", linestyle="--")

    ax.set_title(f"Cumulative Returns — {REGION} Region ({start_year}–{end_year})",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Cumulative Return  (base = 1 at start)", fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()
    print(f"  ✓ Plot saved → {output_path}\n")


# ─────────────────────────────────────────────────────────────
# Convenience wrapper
# ─────────────────────────────────────────────────────────────

def run_performance(mv_ret: pd.Series, vw_ret: pd.Series, rf: pd.Series) -> None:
    """Compute stats, print table, and plot cumulative returns."""
    print("=" * 55)
    print("PERFORMANCE ANALYSIS")
    print("=" * 55)

    common = mv_ret.index.intersection(vw_ret.index)
    if common.empty:
        raise ValueError("No overlapping dates between MV and VW return series.")

    mv_common = mv_ret.loc[common]
    vw_common = vw_ret.loc[common]

    stats_mv = compute_stats(mv_common, rf, "Min-Variance")
    stats_vw = compute_stats(vw_common, rf, "Value-Weighted")

    print_stats_table(stats_mv, stats_vw)
    plot_cumulative_returns(mv_common, vw_common)
