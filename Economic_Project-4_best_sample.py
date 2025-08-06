
# ======================================================================
#  Economic_Project‑4_best_sample.py
#  Template notebook – flat‑function style with explicit section headers
# ======================================================================

# %% -------------------------------------------------------------------
# ## Imports & Global Paths
# ----------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR   = Path("data")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# %% -------------------------------------------------------------------
# ## Utility Functions
# ----------------------------------------------------------------------
def load_prices(csv_path: Path) -> pd.DataFrame:
    """
    Load token price levels. Expected shape:
        index   : datetime (monthly, end‑of‑month)
        columns : DAO tickers
    Returns a DataFrame sorted by date.
    """
    prices = (pd.read_csv(csv_path, index_col=0, parse_dates=True)
                .sort_index())
    return prices


def pct_change_monthly(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute simple (non‑log) percent returns.
    """
    return prices.pct_change().dropna(how="all")


# %% -------------------------------------------------------------------
# ## Factor Construction
# ----------------------------------------------------------------------
def long_short_weights(signal: pd.Series, q: float = 0.20) -> pd.Series:
    """
    Create +1/N, 0, –1/N weights for top‑q / bottom‑q buckets.
    """
    top, bottom = signal.quantile([1 - q, q])
    longs  = signal >= top
    shorts = signal <= bottom
    w      = pd.Series(0.0, index=signal.index)
    if longs.any():
        w[longs]  =  1.0 / longs.sum()
    if shorts.any():
        w[shorts] = -1.0 / shorts.sum()
    return w


def build_factor(prices: pd.DataFrame,
                 zscores: pd.DataFrame,
                 q: float = 0.20) -> pd.Series:
    """
    Construct a self‑financing factor return series using lagged z‑scores.
    """
    rets      = pct_change_monthly(prices)
    z_lag     = zscores.shift(1).loc[rets.index]
    fac_rets  = []
    for date in rets.index:
        w = long_short_weights(z_lag.loc[date], q=q)
        fac_rets.append((w * rets.loc[date]).sum(skipna=True))
    return pd.Series(fac_rets, index=rets.index, name="FactorReturn")


# %% -------------------------------------------------------------------
# ## Regression (Newey‑West HAC)
# ----------------------------------------------------------------------
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_hac

def run_ts_regressions(dao_rets: pd.DataFrame,
                       factor: pd.Series,
                       hac_lag: int = 3) -> pd.DataFrame:
    """
    Regress each DAO's return on the factor return.
    Returns a tidy DataFrame of α, β, t‑stats, and R².
    """
    merged = pd.concat([dao_rets, factor], axis=1, join="inner").dropna()
    results = []
    for token in dao_rets.columns:
        y = merged[token]
        X = sm.add_constant(merged[factor.name])
        model = sm.OLS(y, X).fit()
        cov   = cov_hac(model, maxlags=hac_lag)
        se    = np.sqrt(np.diag(cov))
        tvals = model.params / se
        results.append({
            "Token": token,
            "Alpha": model.params["const"],
            "Beta":  model.params[factor.name],
            "t(α)":  tvals["const"],
            "t(β)":  tvals[factor.name],
            "R²":    model.rsquared
        })
    return (pd.DataFrame(results)
              .set_index("Token")
              .sort_index())


# %% -------------------------------------------------------------------
# ## Diagnostics & Plots
# ----------------------------------------------------------------------
def plot_factor_series(factor: pd.Series) -> None:
    """Simple line plot of factor returns."""
    factor.plot(figsize=(8, 4), title="Governance Factor Returns")
    plt.ylabel("Monthly Return")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# %% -------------------------------------------------------------------
# ## Main Routine (for script‑style execution)
# ----------------------------------------------------------------------
def main() -> None:
    prices_fp  = DATA_DIR / "prices.csv"
    zscore_fp  = DATA_DIR / "governance_zscores.csv"

    prices  = load_prices(prices_fp)
    zscore  = load_prices(zscore_fp)  # same loader; file has z‑scores

    factor  = build_factor(prices, zscore)
    dao_ret = pct_change_monthly(prices)

    res_df  = run_ts_regressions(dao_ret, factor)
    res_df.to_csv(OUTPUT_DIR / "regression_results.csv")

    plot_factor_series(factor)
    print(res_df.head())


if __name__ == "__main__":
    main()

# ======================================================================
#  End of Economic_Project‑4_best_sample.py
# ======================================================================
