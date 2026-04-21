# Sustainability-Aware Asset Management
## Minimum-Variance Portfolio with Carbon Emission Reduction — AMER Region

> **Course:** Sustainability Aware Asset Management — MSc Finance, HEC Lausanne  
> **Professor:** Eric Jondeau  
> **Region:** Americas (AMER)  
> **Period:** Out-of-sample 2014–2024 (estimation window 2004–2013, rolling)

---

## Project Structure

```
PROJECT_SAAM_TEAM_A/
│
├── data/
│   └── raw/                          # Datastream Excel files (not versioned)
│       ├── DS_RI_T_USD_M_2025.xlsx   # Total Return Index — monthly
│       ├── DS_RI_T_USD_Y_2025.xlsx   # Total Return Index — yearly
│       ├── DS_MV_T_USD_M_2025.xlsx   # Market Capitalization — monthly
│       ├── DS_MV_T_USD_Y_2025.xlsx   # Market Capitalization — yearly
│       ├── DS_CO2_SCOPE_1_Y_2025.xlsx# CO2 Scope 1 emissions — yearly
│       ├── DS_REV_Y_2025.xlsx        # Revenue — yearly
│       ├── Static_2025.xlsx          # ISIN, Name, Country, Region
│       └── Risk_Free_Rate_2025.xlsx  # Monthly risk-free rate (Fama-French)
│
├── src/                              # Source code
│   ├── main.py                       # Entry point — runs the full pipeline
│   ├── config.py                     # Global parameters & file paths
│   ├── data_loader.py                # Load & parse all Datastream Excel files
│   ├── data_cleaning.py              # Data cleaning (Section 1)
│   ├── investment_set.py             # Investment set construction (Section 2.1)
│   ├── optimization.py               # Min-variance optimization (Section 2.2)
│   ├── portfolio_returns.py          # Ex-post MV & VW returns (Section 2.2-2.3)
│   ├── performance.py                # Stats, plots & Excel export (Section 2.3)
│   ├── carbon.py                     # Carbon intensity, WACI, CF (Section 3.1)
│   ├── carbon_portfolio.py           # Carbon-constrained portfolios (3.2, 3.3, 4.1)
│   └── comparison.py                 # Portfolio comparisons (3.4, 4.2)
│
├── outputs/                          # Generated files (plots, Excel)
├── requirements.txt
└── README.md
```

---

## Installation

**1. Clone the repository and create a virtual environment:**
```bash
git clone <repo-url>
cd PROJECT_SAAM_TEAM_A
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Place the Datastream Excel files** in `data/raw/` (see structure above).

**4. Run the project:**
```bash
cd src
python3 main.py
```

---

## Dependencies

| Package | Version | Usage |
|---|---|---|
| `pandas` | ≥ 2.0.0 | Data manipulation |
| `numpy` | ≥ 1.24.0 | Matrix computations |
| `scipy` | ≥ 1.10.0 | Portfolio optimization (SLSQP) |
| `matplotlib` | ≥ 3.7.0 | Plots |
| `openpyxl` | ≥ 3.1.0 | Excel read/write |

---

## Pipeline Overview

```
load_all()
    └─► clean_all()
            └─► build_investment_sets()
                    └─► run_optimization()          # Part I
                            └─► compute_all_returns()
                                    └─► run_performance()
                                            └─► compute_carbon_intensity()   # Part II
                                                    └─► run_mv_carbon()      # Section 3.2
                                                    └─► run_te_carbon()      # Section 3.3
                                                    └─► run_net_zero()       # Section 4.1
                                                    └─► compare_portfolios() # 3.4 & 4.2
```

---

## Part I — Minimum-Variance Portfolio

### Section 1 — Data Cleaning (`data_cleaning.py`)

| Step | Rule |
|---|---|
| Missing prices | Drop ISINs with all NaN prices (no Datastream match) |
| Low prices | RI < 0.5 → treated as NaN (avoids infinite returns) |
| Internal gaps | Forward-fill only between two valid observations |
| Delistings | First NaN after a valid price → return of −100% |
| CO2 / Revenue | Forward-fill last available annual observation |

### Section 2.1 — Investment Set (`investment_set.py`)

Each year Y, a firm is eligible if:
1. It belongs to the **AMER** region
2. **CO2 Scope 1** data is available at end of year Y
3. **RI price** is not missing at end of year Y
4. At least **36 months** of valid returns in the 10-year window
5. **Not stale** — proportion of zero monthly returns ≤ 50%

### Section 2.2 — Optimization (`optimization.py`)

Minimum-variance long-only portfolio solved with **SLSQP**:

```
min   w' Σ_Y w
s.t.  Σ w = 1
      w_i ≥ 0   for all i
```

- Estimation window: **10 years** (120 months) ending at December of year Y
- Rebalanced annually from **2013 to 2024**
- Covariance matrix estimated with `min_periods=36`

### Section 2.3 — Benchmark (`portfolio_returns.py`, `performance.py`)

Value-weighted benchmark **P^(vw)** uses monthly market cap weights:

```
w_{i,t} = Cap_{i,t} / Σ_j Cap_{j,t}
```

**Performance metrics:** annualised return (µ), volatility (σ), Sharpe ratio, min, max.

---

## Part II — Portfolio Allocation with Carbon Emission Reduction

### Section 3.1 — Carbon Metrics (`carbon.py`)

**Carbon Intensity:**
```
CI_{i,Y} = CO2_{i,Y} / (Rev_{i,Y} / 1000)    [tonnes CO2e / M USD revenue]
```
> Note: Revenue is in thousands USD in Datastream → divide by 1000 to get millions.

**Weighted-Average Carbon Intensity (WACI):**
```
WACI_Y^(p) = Σ_i α_{i,Y} × CI_{i,Y}
```

**Carbon Footprint (CF):**
```
CF_Y^(p) = Σ_i (α_{i,Y} / Cap_{i,Y}) × E_{i,Y}    [tonnes CO2e / M USD invested]
```

### Section 3.2 — MV with Carbon Constraint (`carbon_portfolio.py`)

Portfolio **P^(mv)(0.5)** — minimize variance with CF ≤ 50% of MV baseline:

```
min   w' Σ_Y w
s.t.  CF(w) ≤ 0.5 × CF(P^(mv))
      Σ w = 1,   w_i ≥ 0
```

### Section 3.3 — Tracking Error with Carbon Constraint (`carbon_portfolio.py`)

Portfolio **P^(vw)(0.5)** — passive investor minimizing tracking error vs VW:

```
min   (w - w_vw)' Σ_Y (w - w_vw)
s.t.  CF(w) ≤ 0.5 × CF(P^(vw))
      Σ w = 1,   w_i ≥ 0
```

### Section 3.4 — Comparison (`comparison.py`)

Compares 4 portfolios: **P^(mv)**, **P^(mv)(0.5)**, **P^(vw)**, **P^(vw)(0.5)**

### Section 4.1 — Net Zero Portfolio (`carbon_portfolio.py`)

Portfolio **P^(vw)(NZ)** — annually tightening carbon constraint (θ = 10%/year):

```
CF_Y(w) ≤ (1 - 0.10)^(Y - 2013) × CF_{2013}(P^(vw))
```

### Section 4.2 — Comparison (`comparison.py`)

Compares 3 portfolios: **P^(vw)**, **P^(vw)(0.5)**, **P^(vw)(NZ)**

---

## Outputs

After running `main.py`, the following files are generated in the project root:

| File | Description |
|---|---|
| `cumulative_returns.png` | Part I — MV vs VW cumulative returns |
| `monthly_portfolio_returns.xlsx` | Part I — Monthly returns table |
| `carbon_metrics.png` | WACI & CF evolution (MV vs VW) |
| `comparison_34_cumulative.png` | Section 3.4 — 4 portfolio comparison |
| `comparison_34_carbon.png` | Section 3.4 — Carbon metrics comparison |
| `comparison_42_cumulative.png` | Section 4.2 — Net zero comparison |
| `comparison_42_carbon.png` | Section 4.2 — Net zero carbon path |

---

## Key Parameters (`config.py`)

| Parameter | Value | Description |
|---|---|---|
| `REGION` | `"AMER"` | Assigned region |
| `ESTIMATION_YEARS` | `10` | Rolling estimation window (years) |
| `TAU` | `120` | Rolling estimation window (months) |
| `REBALANCE_YEARS` | `2013–2024` | Annual rebalancing dates |
| `LOW_PRICE_THRESHOLD` | `0.5` | RI below this → NaN |
| `STALE_THRESHOLD` | `0.50` | Max proportion of zero returns |
| `MIN_MONTHS_DATA` | `36` | Min valid observations required |

---

## Notes

- All portfolios are **long-only** (no short sales) to facilitate carbon footprint interpretation.
- The project uses **Scope 1** CO2 emissions as assigned to the group.
- Out-of-sample period: **January 2014 → December 2024** (144 months).
- The risk-free rate comes from the **Fama-French** data library (monthly, in %).


