# carbon.py — Carbon intensity, WACI & CF (Section 3.1)

import numpy as np
import pandas as pd
from config import REBALANCE_YEARS, _SRC_DIR


# 1. Carbon intensity

def compute_carbon_intensity(co2: pd.DataFrame,
                              revenue: pd.DataFrame) -> pd.DataFrame:
    """
    CI_{i,Y} = CO2_{i,Y} / (Rev_{i,Y} / 1000)
             = CO2_{i,Y} * 1000 / Rev_{i,Y}

    Units: tonnes CO2e per million USD of revenue.
    Revenue is in thousands USD in Datastream → divide by 1000 to get millions.
    Returns DataFrame (firms × years), same shape as co2.
    """
    # Align on common columns (years)
    common_cols = co2.columns.intersection(revenue.columns)
    co2_a   = co2[common_cols]
    rev_a   = revenue[common_cols] / 1000   # thousands → millions USD

    ci = co2_a / rev_a                      # tonnes CO2e / million USD revenue
    ci = ci.replace([np.inf, -np.inf], np.nan)
    return ci


# 2. Weighted-average carbon intensity (WACI)

def compute_waci(weights: pd.Series, ci: pd.DataFrame, year: int) -> float:
    """
    WACI_Y^(p) = Σ_i α_{i,Y} * CI_{i,Y}

    weights : portfolio weights at end of year Y (pd.Series, index=ISINs)
    ci      : carbon intensity DataFrame (firms × years)
    year    : rebalancing year Y
    """
    # Get December column for year Y
    cols = pd.DatetimeIndex(ci.columns)
    dec_cols = cols[(cols.year == year) & (cols.month == 12)]
    if len(dec_cols) == 0:
        return np.nan
    ci_year = ci[dec_cols[-1]]

    # Align weights and CI
    common = weights.index.intersection(ci_year.index)
    w  = weights.reindex(common).fillna(0)
    c  = ci_year.reindex(common)

    # WACI = Σ_i α_{i,Y} * CI_{i,Y}
    # Firms with missing CI are excluded (NaN × weight = NaN → dropped by sum)
    valid = c.notna()
    w_valid = w[valid]
    c_valid = c[valid]

    if w_valid.sum() == 0:
        return np.nan

    return float(w_valid @ c_valid)


# 3. Carbon footprint (CF)

def compute_cf(weights: pd.Series,
               co2: pd.DataFrame,
               mv_y: pd.DataFrame,
               year: int,
               portfolio_value: float = 1.0) -> float:
    """
    CF_Y^(p) = (1 / V_Y) * Σ_i o_{i,Y} * E_{i,Y}

    where:
        o_{i,Y}  = V_{i,Y} / Cap_{i,Y}   ownership fraction
        V_{i,Y}  = α_{i,Y} * V_Y         dollar value invested in firm i
        E_{i,Y}  = CO2 emissions of firm i in year Y

    Simplified:
        CF_Y^(p) = Σ_i (α_{i,Y} / Cap_{i,Y}) * E_{i,Y}

    Units: tonnes CO2e per million USD invested.
    Cap_{i,Y} is in millions USD (Datastream convention).
    """
    def _get_dec(df):
        cols = pd.DatetimeIndex(df.columns)
        dec  = cols[(cols.year == year) & (cols.month == 12)]
        return df[dec[-1]] if len(dec) > 0 else None

    co2_year = _get_dec(co2)
    cap_year = _get_dec(mv_y)

    if co2_year is None or cap_year is None:
        return np.nan

    cap_year_m = cap_year  # already in million USD

    common = weights.index.intersection(co2_year.index).intersection(cap_year_m.index)
    w   = weights.reindex(common).fillna(0)
    e   = co2_year.reindex(common)
    cap = cap_year_m.reindex(common)

    valid = e.notna() & cap.notna() & (cap > 0)
    w_v   = w[valid]
    e_v   = e[valid]
    cap_v = cap[valid]

    if w_v.sum() == 0:
        return np.nan

    cf = (w_v / cap_v * e_v).sum()
    return float(cf)


# 4. Rolling WACI & CF over all rebalancing years

def compute_portfolio_carbon_metrics(weights_dict: dict,
                                      ci: pd.DataFrame,
                                      co2: pd.DataFrame,
                                      mv_y: pd.DataFrame,
                                      label: str = "Portfolio") -> pd.DataFrame:
    """
    Compute WACI and CF for each rebalancing year Y.

    Returns
    -------
    pd.DataFrame with columns [WACI, CF], indexed by year.
    """
    records = []
    for Y in REBALANCE_YEARS:
        if Y not in weights_dict:
            continue
        w     = weights_dict[Y]
        waci  = compute_waci(w, ci, Y)
        cf    = compute_cf(w, co2, mv_y, Y)
        records.append({"Year": Y, "WACI": waci, "CF": cf})

    df = pd.DataFrame(records).set_index("Year")
    print(f"  {label} carbon metrics computed for {len(df)} years.")
    return df


# 5. Top-N firms driving WACI

def top_carbon_contributors(weights_dict: dict,
                             ci: pd.DataFrame,
                             static: pd.DataFrame,
                             year: int,
                             top_n: int = 10) -> pd.DataFrame:
    """
    Identify the top_n firms that contribute most to WACI in a given year.
    Contribution_i = α_{i,Y} * CI_{i,Y}

    Returns a DataFrame with firm name, ISIN, weight, CI, and contribution.
    """
    if year not in weights_dict:
        return pd.DataFrame()

    cols = pd.DatetimeIndex(ci.columns)
    dec  = cols[(cols.year == year) & (cols.month == 12)]
    if len(dec) == 0:
        return pd.DataFrame()

    w        = weights_dict[year]
    ci_year  = ci[dec[-1]]
    common   = w.index.intersection(ci_year.index)

    contrib = (w.reindex(common).fillna(0) * ci_year.reindex(common)).dropna()
    contrib = contrib.sort_values(ascending=False).head(top_n)

    result = pd.DataFrame({
        "ISIN"        : contrib.index,
        "Contribution": contrib.values,
        "Weight"      : w.reindex(contrib.index).values,
        "CI"          : ci_year.reindex(contrib.index).values,
    })

    # Add firm names from static
    if "NAME" in static.columns:
        result["Name"] = static["NAME"].reindex(contrib.index).values
    elif static.index.name == "ISIN":
        result["Name"] = result["ISIN"]

    result = result[["ISIN", "Name", "Weight", "CI", "Contribution"]] \
             if "Name" in result.columns else result
    result = result.reset_index(drop=True)
    return result


# 6. Plot WACI & CF evolution

def plot_carbon_metrics(metrics_mv: pd.DataFrame,
                         metrics_vw: pd.DataFrame,
                         output_dir=None) -> None:
    """
    Plot WACI and CF evolution for MV and VW portfolios side by side.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # WACI
    axes[0].plot(metrics_mv.index, metrics_mv["WACI"],
                 marker="o", label="Min-Variance", color="steelblue", linewidth=2)
    axes[0].plot(metrics_vw.index, metrics_vw["WACI"],
                 marker="s", label="Value-Weighted", color="darkorange",
                 linewidth=2, linestyle="--")
    axes[0].set_title("Weighted-Average Carbon Intensity (WACI)", fontweight="bold")
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Tonnes CO₂e / M USD Revenue")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # CF
    axes[1].plot(metrics_mv.index, metrics_mv["CF"],
                 marker="o", label="Min-Variance", color="steelblue", linewidth=2)
    axes[1].plot(metrics_vw.index, metrics_vw["CF"],
                 marker="s", label="Value-Weighted", color="darkorange",
                 linewidth=2, linestyle="--")
    axes[1].set_title("Carbon Footprint (CF)", fontweight="bold")
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Tonnes CO₂e / M USD Invested")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("WACI & CF of MV & VW — AMER Region (2013-2024)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    if output_dir:
        path = str(output_dir / "carbon_metrics.png")
        plt.savefig(path, dpi=150)
        print(f"  Carbon metrics plot saved → {path}")
    plt.close()
    
    