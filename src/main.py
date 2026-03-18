# =============================================================
# main.py — Entry point
# Run:  python main.py
# =============================================================

from data_loader       import load_all
from data_cleaning     import clean_all
from investment_set    import build_investment_sets
from optimization      import run_optimization
from portfolio_returns import compute_all_returns
from performance       import run_performance


def main():
    print("\n" + "█" * 55)
    print("  MINIMUM-VARIANCE PORTFOLIO — AMER REGION")
    print("█" * 55 + "\n")

    # ── Step 1: Load raw data ─────────────────────────────────
    data = load_all()

    # ── Step 2: Clean data ────────────────────────────────────
    data = clean_all(data)

    # ── Step 3: Build investment sets (per year) ──────────────
    invest_sets, ret_windows = build_investment_sets(data)

    # ── Step 4: Optimize min-variance weights ─────────────────
    weights_dict = run_optimization(invest_sets, ret_windows)

    # ── Step 5: Compute ex-post portfolio returns ─────────────
    mv_ret, vw_ret = compute_all_returns(weights_dict, invest_sets, data)

    # ── Step 6: Performance statistics & plot ─────────────────
    run_performance(mv_ret, vw_ret, data["rf"])

    print("█" * 55)
    print("  ALL DONE")
    print("█" * 55 + "\n")


if __name__ == "__main__":
    main()
