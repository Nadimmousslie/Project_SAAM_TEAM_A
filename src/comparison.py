### comparison.py — Sections 3.4 & 4.2: Portfolio Comparison ###

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from config import _SRC_DIR
from performance import compute_stats


# ─────────────────────────────────────────────────────────────
# SECTION 3.4 — Compare 4 portfolios
# P_mv, P_mv(0.5), P_vw, P_vw(0.5)
# ─────────────────────────────────────────────────────────────

def compare_portfolios_34(mv_ret: pd.Series,
                           mvc_ret: pd.Series,
                           vw_ret: pd.Series,
                           tec_ret: pd.Series,
                           rf: pd.Series,
                           metrics_mv: pd.DataFrame,
                           metrics_mvc: pd.DataFrame,
                           metrics_vw: pd.DataFrame,
                           metrics_tec: pd.DataFrame) -> None:
    """
    Section 3.4: Full comparison of the 4 portfolios.
    - Performance stats table
    - Cumulative return plot
    - WACI & CF evolution plot
    """
    print("=" * 70)
    print("SECTION 3.4 — PORTFOLIO COMPARISON")
    print("=" * 70)

    portfolios = {
        "P_mv"     : (mv_ret,  "steelblue",   "-",  "MV  $P^{mv}_{oos}$"),
        "P_mv(0.5)": (mvc_ret, "royalblue",   "--", "MV carbon  $P^{mv}_{oos}(0.5)$"),
        "P_vw"     : (vw_ret,  "darkorange",  "-",  "VW  $P^{vw}_{oos}$"),
        "P_vw(0.5)": (tec_ret, "orangered",   "--", "TE carbon  $P^{vw}_{oos}(0.5)$"),
    }

    # ── Stats table ───────────────────────────────────────────
    _print_comparison_table(portfolios, rf)

    # ── Cumulative returns ────────────────────────────────────
    _plot_cumulative(portfolios,
                     title="Cumulative Returns — (AMER, 2014–2025)",
                     filename="comparison_34_cumulative.png")

    # ── Carbon metrics ────────────────────────────────────────
    _plot_carbon_comparison(
        {
            "MV":       metrics_mv,
            "MV(0.5)":  metrics_mvc,
            "VW":       metrics_vw,
            "VW(0.5)":  metrics_tec,
        },
        title="Carbon Metrics — WACI & CF",
        filename="comparison_34_carbon.png",
    )


# ─────────────────────────────────────────────────────────────
# SECTION 4.2 — Compare 3 VW-based portfolios
# P_vw, P_vw(0.5), P_vw(NZ)
# ─────────────────────────────────────────────────────────────

def compare_portfolios_42(vw_ret: pd.Series,
                           tec_ret: pd.Series,
                           nz_ret: pd.Series,
                           rf: pd.Series,
                           metrics_vw: pd.DataFrame,
                           metrics_tec: pd.DataFrame,
                           metrics_nz: pd.DataFrame) -> None:
    """
    Section 4.2: Compare P_vw, P_vw(0.5), P_vw(NZ).
    """
    print("=" * 70)
    print("SECTION 4.2 — NET ZERO COMPARISON")
    print("=" * 70)

    portfolios = {
        "P_vw"    : (vw_ret,  "darkorange", "-",   "VW  $P^{vw}_{oos}$"),
        "P_vw(0.5)":(tec_ret, "orangered",  "--",  "TE carbon  $P^{vw}_{oos}(0.5)$"),
        "P_vw(NZ)": (nz_ret,  "green",      "-.",  "Net Zero  $P^{vw}_{oos}(NZ)$"),
    }

    # ── Stats table ───────────────────────────────────────────
    _print_comparison_table(portfolios, rf)

    # ── Cumulative returns ────────────────────────────────────
    _plot_cumulative(portfolios,
                     title="Cumulative Returns VW, TE & Net ZERO — (AMER, 2014–2025)",
                     filename="comparison_42_cumulative.png")

    # ── Carbon metrics ────────────────────────────────────────
    _plot_carbon_comparison(
        {
            "VW":      metrics_vw,
            "VW(0.5)": metrics_tec,
            "VW(NZ)":  metrics_nz,
        },
        title="Carbon Metrics — (Net Zero Path)",
        filename="comparison_42_carbon.png",
    )


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def _print_comparison_table(portfolios: dict, rf: pd.Series) -> None:
    metrics_def = [
        ("Ann. avg return", "mu_p",    "{:.2%}"),
        ("Ann. volatility", "sigma_p", "{:.2%}"),
        ("Sharpe ratio",    "SR_p",    "{:.4f}"),
        ("Min monthly",     "min",     "{:.2%}"),
        ("Max monthly",     "max",     "{:.2%}"),
    ]
    n_cols = len(portfolios)
    col_w  = 18

    header = f"{'Metric':<20}" + "".join(f"{v[3]:>{col_w}}" for v in portfolios.values())
    print("\n" + "=" * (20 + col_w * n_cols))
    print(header)
    print("-" * (20 + col_w * n_cols))

    stats_all = {k: compute_stats(v[0], rf, k) for k, v in portfolios.items()}

    for name, key, fmt in metrics_def:
        row = f"  {name:<18}"
        for k in portfolios:
            row += f"{fmt.format(stats_all[k][key]):>{col_w}}"
        print(row)
    print("=" * (20 + col_w * n_cols) + "\n")


def _plot_cumulative(portfolios: dict, title: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(13, 6))

    for key, (ret, color, ls, label) in portfolios.items():
        cum = (1 + ret).cumprod()
        ax.plot(cum.index, cum.values, label=label,
                color=color, linestyle=ls, linewidth=2.0)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return (base = 1)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()

    path = str(_SRC_DIR.parent / filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ✓ Plot saved → {path}")


def _plot_carbon_comparison(metrics_dict: dict,
                             title: str,
                             filename: str) -> None:
    colors = ["steelblue", "royalblue", "darkorange", "orangered", "green"]
    styles = ["-", "--", "-", "--", "-."]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i, (label, df) in enumerate(metrics_dict.items()):
        c, ls = colors[i % len(colors)], styles[i % len(styles)]
        if "WACI" in df.columns:
            axes[0].plot(df.index, df["WACI"], marker="o", label=label,
                         color=c, linestyle=ls, linewidth=2)
        if "CF" in df.columns:
            axes[1].plot(df.index, df["CF"], marker="o", label=label,
                         color=c, linestyle=ls, linewidth=2)

    for ax, ylabel, metric_title in zip(
        axes,
        ["Tonnes CO₂e / M USD Revenue", "Tonnes CO₂e / M USD Invested"],
        ["WACI Evolution", "Carbon Footprint (CF) Evolution"],
    ):
        ax.set_title(metric_title, fontweight="bold")
        ax.set_xlabel("Year")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()

    path = str(_SRC_DIR.parent / filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ✓ Carbon comparison plot saved → {path}\n")


# ─────────────────────────────────────────────────────────────
# SECTION 3.2 — Compare MV vs MV(0.5) only
# Cumulative returns + CF only (no WACI)
# ─────────────────────────────────────────────────────────────

def compare_mv_carbon_32(mv_ret: pd.Series,
                          mvc_ret: pd.Series,
                          rf: pd.Series,
                          metrics_mv: pd.DataFrame,
                          metrics_mvc: pd.DataFrame) -> None:
    """
    Section 3.2: Compare MV and MV(0.5).
    - Performance stats table
    - Cumulative return plot (MV + MV(0.5))
    - CF plot only (no WACI)
    """
    print("=" * 70)
    print("SECTION 3.2 — MV vs MV(0.5) COMPARISON")
    print("=" * 70)

    portfolios = {
        "P_mv"     : (mv_ret,  "steelblue",  "-",  "MV  $P^{mv}_{oos}$"),
        "P_mv(0.5)": (mvc_ret, "royalblue",  "--", "MV carbon  $P^{mv}_{oos}(0.5)$"),
    }

    _print_comparison_table(portfolios, rf)

    _plot_cumulative(portfolios,
                     title="Cumulative Returns — MV vs MV(0.5) (AMER, 2014–2025)",
                     filename="comparison_32_cumulative.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(metrics_mv.index, metrics_mv["CF"],
            marker="o", label="MV", color="steelblue", linewidth=2)
    ax.plot(metrics_mvc.index, metrics_mvc["CF"],
            marker="s", label="MV(0.5)", color="royalblue",
            linewidth=2, linestyle="--")
    ax.set_title("Carbon Footprint (CF) — MV vs MV(0.5)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Tonnes CO₂e / M USD Invested")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = str(_SRC_DIR.parent / "comparison_32_carbon.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ✓ CF plot saved → {path}\n")


# ─────────────────────────────────────────────────────────────
# SECTION 3.3 — Compare VW vs VW(0.5) only
# Cumulative returns + CF only (no WACI)
# ─────────────────────────────────────────────────────────────

def compare_vw_carbon_33(vw_ret: pd.Series,
                          tec_ret: pd.Series,
                          rf: pd.Series,
                          metrics_vw: pd.DataFrame,
                          metrics_tec: pd.DataFrame) -> None:
    """
    Section 3.3: Compare VW and VW(0.5).
    - Performance stats table
    - Cumulative return plot (VW + VW(0.5))
    - CF plot only (no WACI)
    """
    print("=" * 70)
    print("SECTION 3.3 — VW vs VW(0.5) COMPARISON")
    print("=" * 70)

    portfolios = {
        "P_vw"     : (vw_ret,  "darkorange",  "-",  "VW  $P^{vw}_{oos}$"),
        "P_vw(0.5)": (tec_ret, "orangered",   "--", "TE carbon  $P^{vw}_{oos}(0.5)$"),
    }

    _print_comparison_table(portfolios, rf)

    _plot_cumulative(portfolios,
                     title="Cumulative Returns — VW vs VW(0.5) (AMER, 2014–2025)",
                     filename="comparison_33_cumulative.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(metrics_vw.index, metrics_vw["CF"],
            marker="o", label="VW", color="darkorange", linewidth=2)
    ax.plot(metrics_tec.index, metrics_tec["CF"],
            marker="s", label="VW(0.5)", color="orangered",
            linewidth=2, linestyle="--")
    ax.set_title("Carbon Footprint (CF) — VW vs VW(0.5)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Tonnes CO₂e / M USD Invested")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = str(_SRC_DIR.parent / "comparison_33_carbon.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ✓ CF plot saved → {path}\n")


# ─────────────────────────────────────────────────────────────
# SECTION 3.6 — Compare ALL 5 portfolios (cumulative only)
# MV, MV(0.5), VW, VW(0.5), Net Zero
# ─────────────────────────────────────────────────────────────

def compare_all_portfolios(mv_ret: pd.Series,
                            mvc_ret: pd.Series,
                            vw_ret: pd.Series,
                            tec_ret: pd.Series,
                            nz_ret: pd.Series,
                            rf: pd.Series) -> None:
    """
    Section 3.6: Compare all 5 portfolios — stats table + cumulative returns.
    """
    print("=" * 70)
    print("SECTION 3.6 — ALL PORTFOLIOS COMPARISON")
    print("=" * 70)

    portfolios = {
        "P_mv"      : (mv_ret,  "steelblue",  "-",   "MV  $P^{mv}_{oos}$"),
        "P_mv(0.5)" : (mvc_ret, "royalblue",  "--",  "MV carbon  $P^{mv}_{oos}(0.5)$"),
        "P_vw"      : (vw_ret,  "darkorange", "-",   "VW  $P^{vw}_{oos}$"),
        "P_vw(0.5)" : (tec_ret, "orangered",  "--",  "TE carbon  $P^{vw}_{oos}(0.5)$"),
        "P_vw(NZ)"  : (nz_ret,  "green",      "-.",  "Net Zero  $P^{vw}_{oos}(NZ)$"),
    }

    _print_comparison_table(portfolios, rf)

    _plot_cumulative(portfolios,
                     title="Cumulative Returns — All Portfolios (AMER, 2014–2025)",
                     filename="comparison_all_cumulative.png")



