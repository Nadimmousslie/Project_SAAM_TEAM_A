# main.py — Entry point. Run: python3 main.py

from data_loader        import load_all
from data_cleaning      import clean_all
from investment_set     import build_investment_sets
from optimization       import run_optimization
from portfolio_returns  import compute_all_returns, compute_mv_returns
from performance        import run_performance
from carbon             import (compute_carbon_intensity,
                                compute_portfolio_carbon_metrics,
                                top_carbon_contributors,
                                plot_carbon_metrics)
from carbon_portfolio   import (precompute_sigmas,
                                run_mv_carbon,
                                run_te_carbon,
                                run_net_zero)
from comparison         import (compare_portfolios_34, compare_portfolios_42,
                                compare_mv_carbon_32, compare_vw_carbon_33,
                                compare_all_portfolios)
from config             import REBALANCE_YEARS, _SRC_DIR
import pandas as pd


def _build_vw_weights(invest_sets, mv_y):
    """VW weights dict using yearly market cap."""
    weights_vw = {}
    for Y in REBALANCE_YEARS:
        if Y not in invest_sets:
            continue
        isins_Y  = invest_sets[Y]
        cols     = pd.DatetimeIndex(mv_y.columns)
        dec_cols = cols[(cols.year == Y) & (cols.month == 12)]
        if len(dec_cols) == 0:
            continue
        caps = mv_y[dec_cols[-1]].reindex(isins_Y).dropna()
        if caps.sum() > 0:
            weights_vw[Y] = caps / caps.sum()
    return weights_vw


def main():
    print("\n" + "█" * 60)
    print("  MINIMUM-VARIANCE PORTFOLIO — AMER REGION")
    print("  Parts I & II: MV + Carbon Constraints + Net Zero")
    print("█" * 60 + "\n")

    # PART I

    data = load_all()
    data = clean_all(data)
    invest_sets, ret_windows = build_investment_sets(data)
    weights_mv = run_optimization(invest_sets, ret_windows)
    mv_ret, vw_ret = compute_all_returns(weights_mv, invest_sets, data)
    run_performance(mv_ret, vw_ret, data["rf"])

    # PART II — Carbon emission reduction

    print("\n" + "█" * 60)
    print("  PART II — CARBON EMISSION REDUCTION")
    print("█" * 60 + "\n")

    # Pre-compute covariance matrices ONCE (shared by 3.2, 3.3, 4.1)
    print("Pre-computing shared resources...")
    sigmas = precompute_sigmas(invest_sets, ret_windows)

    # Carbon intensity & metrics
    ci = compute_carbon_intensity(data["co2"], data["revenue"])
    metrics_mv = compute_portfolio_carbon_metrics(
        weights_mv, ci, data["co2"], data["mv_y"], label="MV")

    weights_vw = _build_vw_weights(invest_sets, data["mv_y"])
    metrics_vw = compute_portfolio_carbon_metrics(
        weights_vw, ci, data["co2"], data["mv_y"], label="VW")

    plot_carbon_metrics(metrics_mv, metrics_vw, output_dir=_SRC_DIR.parent)

    last_year = max(weights_mv.keys())
    print(f"\n  Top-10 carbon contributors ({last_year}):")
    top10 = top_carbon_contributors(weights_mv, ci, data["static"], last_year)
    print(top10.to_string(index=False))

    # Section 3.2 — MV + carbon constraint
    weights_mvc = run_mv_carbon(
        invest_sets, ret_windows, weights_mv,
        data["co2"], data["mv_y"], sigmas, reduction=0.5)
    mvc_ret  = compute_mv_returns(weights_mvc, data["returns_m"])
    metrics_mvc = compute_portfolio_carbon_metrics(
        weights_mvc, ci, data["co2"], data["mv_y"], label="MV(0.5)")

    # Section 3.2 — Comparison MV vs MV(0.5)
    compare_mv_carbon_32(mv_ret, mvc_ret, data["rf"], metrics_mv, metrics_mvc)

    # Section 3.3 — Tracking error + carbon constraint
    weights_tec = run_te_carbon(
        invest_sets, ret_windows, data["mv_y"],
        data["co2"], data["mv_y"], sigmas, reduction=0.5, label="VW(0.5)")
    tec_ret = compute_mv_returns(weights_tec, data["returns_m"])
    metrics_tec = compute_portfolio_carbon_metrics(
        weights_tec, ci, data["co2"], data["mv_y"], label="VW(0.5)")

    # Section 3.3 — Comparison VW vs VW(0.5)
    compare_vw_carbon_33(vw_ret, tec_ret, data["rf"], metrics_vw, metrics_tec)

    # Section 3.4 — Comparison
    compare_portfolios_34(
        mv_ret, mvc_ret, vw_ret, tec_ret, data["rf"],
        metrics_mv, metrics_mvc, metrics_vw, metrics_tec)

    # PART II — Net zero objective

    print("\n" + "█" * 60)
    print("  PART II — NET ZERO OBJECTIVE")
    print("█" * 60 + "\n")

    # Section 4.1 — Net zero
    weights_nz = run_net_zero(
        invest_sets, ret_windows, data["mv_y"],
        data["co2"], data["mv_y"], sigmas, theta=0.10, Y0=2013)
    nz_ret = compute_mv_returns(weights_nz, data["returns_m"])
    metrics_nz = compute_portfolio_carbon_metrics(
        weights_nz, ci, data["co2"], data["mv_y"], label="VW(NZ)")

    # Section 4.2 — Final comparison
    compare_portfolios_42(
        vw_ret, tec_ret, nz_ret, data["rf"],
        metrics_vw, metrics_tec, metrics_nz)

    # All 5 portfolios comparison
    compare_all_portfolios(
        mv_ret, mvc_ret, vw_ret, tec_ret, nz_ret, data["rf"])

    print("\n" + "█" * 60)
    print("  ALL DONE")
    print("█" * 60 + "\n")


if __name__ == "__main__":
    main()
    