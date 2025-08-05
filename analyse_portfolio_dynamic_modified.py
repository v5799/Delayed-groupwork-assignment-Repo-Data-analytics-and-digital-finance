"""
Modified dynamic portfolio analysis with risk‑free rate support
=============================================================

This module extends the ``analyse_portfolio_dynamic`` script by adding
support for incorporating a risk‑free rate into portfolio return
calculations.  The original code forms long–short portfolios based on
governance z‑scores and regresses monthly portfolio returns on the
previous month's average z‑score.  Here we allow the user to subtract
a crypto/DAO‑specific risk‑free rate from those returns to obtain
excess returns.  This is particularly important when benchmarking
factor models against a market neutral baseline.

Key additions:

* A ``risk_free`` parameter to ``run_dynamic_analysis``.  It may be
  ``None`` (default, no risk‑free adjustment), a constant float
  representing an *annualised* risk‑free rate (e.g., ``0.03`` for
  3 %), or a path to a CSV file containing monthly risk‑free rates.
  The CSV must have columns ``date`` and ``risk_free`` with dates
  matching the first day of each month.
* If a constant is supplied, it is converted to a monthly rate by
  dividing by 12.  When using a CSV file the rates are assumed to be
  already monthly (e.g., the monthly staking yield or t‑bill yield).
* Excess returns are computed by subtracting the risk‑free rate from
  the raw long–short returns before regression.

The remainder of the logic (data loading, dynamic selection of
eligible DAOs, portfolio construction and regression) is preserved
from the original script.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import math
import statsmodels.api as sm
from datetime import timedelta
from typing import Iterable, Optional, Union


def compute_monthly_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """Compute monthly log returns from a daily price DataFrame.

    This mirrors the helper from the original module: daily percentage
    changes are converted to log returns using ``ln(1 + r)`` and
    aggregated by summation.  The resulting series is labelled by the
    first day of the next month.

    Parameters
    ----------
    price_df : pandas.DataFrame
        DataFrame indexed by datetime with one column per DAO containing
        price levels.

    Returns
    -------
    pandas.DataFrame
        Monthly log returns labelled by month start.
    """
    def safe_log(x: float) -> float:
        try:
            return math.log(x) if x > 0 else np.nan
        except Exception:
            return np.nan

    # Calculate daily percentage changes and convert to log returns
    daily_returns = price_df.pct_change().applymap(
        lambda x: safe_log(1 + x) if pd.notnull(x) else np.nan
    )
    # Sum daily log returns to get monthly log return
    monthly = daily_returns.resample("M").sum()
    # Label by first day of next month
    monthly.index = monthly.index + pd.offsets.MonthBegin(1)
    return monthly


def _load_risk_free(
    risk_free: Optional[Union[float, str]], common_dates: Iterable[pd.Timestamp]
) -> Optional[pd.Series]:
    """Load or construct a monthly risk‑free rate series.

    Parameters
    ----------
    risk_free : float | str | None
        If ``None`` then no adjustment is applied.  If a float, it is
        interpreted as an annualised rate (e.g., 0.03 for 3 %) and
        converted to a constant monthly rate by dividing by 12.  If a
        string, it is treated as a path to a CSV file containing
        monthly rates.  The CSV must have columns ``date`` and
        ``risk_free``.
    common_dates : iterable of pandas.Timestamp
        Dates to align the risk‑free series to (month start dates).

    Returns
    -------
    pandas.Series | None
        A series indexed by ``common_dates`` containing monthly
        risk‑free rates, or ``None`` if no adjustment should be
        applied.
    """
    if risk_free is None:
        return None
    # constant annualised rate
    if isinstance(risk_free, (int, float)):
        monthly_rate = float(risk_free) / 12.0
        return pd.Series(monthly_rate, index=pd.Index(common_dates, name="date"))
    # CSV file with monthly rates
    if isinstance(risk_free, str):
        rf_df = pd.read_csv(risk_free)
        if "date" not in rf_df.columns:
            raise ValueError("Risk‑free CSV must contain a 'date' column")
        # Parse dates and set index
        rf_df["date"] = pd.to_datetime(rf_df["date"])
        # Use risk_free column if present, otherwise try common names
        rate_col = None
        for c in ["risk_free", "rate", "rf_rate"]:
            if c in rf_df.columns:
                rate_col = c
                break
        if rate_col is None:
            raise ValueError("Risk‑free CSV must have a 'risk_free' column")
        rf_series = rf_df.set_index("date")[rate_col]
        # Reindex to common dates.  Forward fill to handle missing months.
        rf_series = rf_series.reindex(pd.Index(common_dates), method="ffill")
        return rf_series
    raise TypeError("risk_free must be None, a float or a path to a CSV file")


def run_dynamic_analysis(
    zscore_file: str,
    price_file: str,
    top_daos_file: str,
    sample_sizes: Optional[list[int]] = None,
    risk_free: Optional[Union[float, str]] = None,
    start_date: Optional[str] = "2023-01-01",
    end_date: Optional[str] = "2025-01-01",
) -> None:
    """Run dynamic portfolio analysis with optional risk‑free adjustment.

    This function largely mirrors the behaviour of
    :func:`analyse_portfolio_dynamic.run_dynamic_analysis`, but allows
    subtracting a risk‑free rate from the long–short returns.  The
    ``risk_free`` parameter may be ``None`` (no adjustment), a float
    representing an annualised rate, or a path to a CSV containing
    monthly rates.  See :func:`_load_risk_free` for details.

    Parameters
    ----------
    zscore_file : str
        Path to CSV containing monthly z‑scores with dates as rows and
        DAO names as columns.
    price_file : str
        Path to CSV containing daily prices with a 'date' column and one
        column per DAO.
    top_daos_file : str
        Path to CSV containing top DAOs by AUM with at least 'id' and
        'title' columns.  This file is used to order DAOs by AUM.
    sample_sizes : list[int], optional
        List of sample sizes to try (in descending order).  Defaults to
        [60, 40, 20].
    risk_free : float | str | None
        Risk‑free rate specification.  If a float, interpret as an
        annualised rate and convert to monthly; if a string, treat as a
        CSV path; if ``None``, no adjustment is made.
    """
    if sample_sizes is None:
        sample_sizes = [60, 40, 20]
    # Load z‑scores
    z_df = pd.read_csv(zscore_file, index_col=0, parse_dates=True)
    print(f"Loaded z‑scores for {z_df.shape[1]} DAOs across {len(z_df)} months")
    # Load price data
    p_df = pd.read_csv(price_file)
    if "date" not in p_df.columns:
        raise ValueError("Price file must contain a 'date' column")
    p_df["date"] = pd.to_datetime(p_df["date"])
    # Determine analysis window.  If explicit start_date/end_date are
    # provided, use them; otherwise fall back to the last two years.
    if start_date and end_date:
        try:
            window_start = pd.to_datetime(start_date).normalize()
            last_date = pd.to_datetime(end_date).normalize()
        except Exception as e:
            raise ValueError(f"Unable to parse start_date/end_date: {e}")
    else:
        last_date = p_df["date"].max().normalize()
        window_start = last_date - timedelta(days=730)
    print(f"Price data window from {window_start.date()} to {last_date.date()}")
    # Filter to the window and set index
    p_window = p_df[(p_df["date"] >= window_start) & (p_df["date"] <= last_date)].copy()
    p_window = p_window.set_index("date")
    # Identify DAOs with complete data in this window
    price_daos = [c for c in p_window.columns if not p_window[c].isna().any()]
    print(f"DAOs with complete price data in window: {len(price_daos)}")
    # Load top DAOs by AUM to order selection
    top_df = pd.read_csv(top_daos_file)
    # Use 'title' or 'name' to match columns; ensure we have a set
    aum_order = [row.get("title") or row.get("name") for _, row in top_df.iterrows()]
    # Keep only DAOs that are in z_df and have full price data
    valid_daos = [dao for dao in aum_order if dao in price_daos and dao in z_df.columns]
    print(f"Valid DAOs (in z‑scores and full price): {len(valid_daos)}")
    # Compute monthly returns for the window
    monthly_prices = p_window[valid_daos]
    monthly_ret = compute_monthly_returns(monthly_prices)
    # Align z‑scores to the same monthly dates
    common_dates = z_df.index.intersection(monthly_ret.index)
    z_aligned = z_df.loc[common_dates, valid_daos]
    r_aligned = monthly_ret.loc[common_dates, valid_daos]
    # Load risk‑free series if provided
    rf_series = _load_risk_free(risk_free, common_dates)
    # For each sample size, attempt regression
    for size in sample_sizes:
        if len(valid_daos) < size:
            print(f"Not enough DAOs ({len(valid_daos)}) for sample size {size}")
            continue
        selected_daos = valid_daos[:size]
        # Build portfolio returns
        port_records = []
        for dt in common_dates:
            z_scores = z_aligned.loc[dt, selected_daos]
            returns = r_aligned.loc[dt, selected_daos]
            ranks = z_scores.rank(ascending=False, method="first")
            q = size // 5
            top = ranks <= q
            bottom = ranks > size - q
            w_top = 1.0 / top.sum()
            w_bottom = -1.0 / bottom.sum()
            p_ret = (returns[top] * w_top).sum() + (returns[bottom] * w_bottom).sum()
            # Subtract risk‑free rate if available
            if rf_series is not None:
                rf_val = rf_series.loc[dt]
                p_ret = p_ret - rf_val
            port_records.append({"date": dt, "return": p_ret, "z_mean": z_scores.mean()})
        port_df = pd.DataFrame(port_records).set_index("date")
        # Lag z‑scores to avoid look‑ahead bias
        port_df["z_lag"] = port_df["z_mean"].shift(1)
        port_df = port_df.dropna()
        # Regression of (excess) returns on lagged z‑scores
        X = sm.add_constant(port_df["z_lag"])
        model = sm.OLS(port_df["return"], X).fit()
        print(f"\n=== Regression Results for sample size {size} ===")
        print(model.summary())
        # Stop after first feasible size
        break


if __name__ == "__main__":
    # Example usage when running as a script
    import argparse
    parser = argparse.ArgumentParser(
        description="Analyse DAO portfolios with optional risk‑free adjustment"
    )
    parser.add_argument("--zscore-file", required=True, help="Path to CSV containing monthly z‑scores")
    parser.add_argument("--price-file", required=True, help="Path to CSV containing daily prices")
    parser.add_argument("--top-daos-file", required=True, help="Path to CSV containing top DAOs by AUM")
    parser.add_argument(
        "--risk-free",
        help="Risk‑free rate specification: float for annualised rate (e.g., 0.03) or path to CSV",
        default=None,
    )
    parser.add_argument(
        "--start-date",
        help="Start date for price window (YYYY-MM-DD). Overrides the default 2023-01-01.",
        default="2023-01-01",
    )
    parser.add_argument(
        "--end-date",
        help="End date for price window (YYYY-MM-DD). Overrides the default 2025-01-01.",
        default="2025-01-01",
    )
    parser.add_argument(
        "--sample-sizes",
        nargs="*",
        type=int,
        default=[60, 40, 20],
        help="Sample sizes to attempt (in descending order)",
    )
    args = parser.parse_args()
    # Convert risk_free argument to float if possible
    rf_val: Optional[Union[float, str]] = None
    if args.risk_free is not None:
        try:
            rf_val = float(args.risk_free)
        except ValueError:
            rf_val = args.risk_free
    run_dynamic_analysis(
        zscore_file=args.zscore_file,
        price_file=args.price_file,
        top_daos_file=args.top_daos_file,
        sample_sizes=args.sample_sizes,
        risk_free=rf_val,
        start_date=args.start_date,
        end_date=args.end_date,
    )