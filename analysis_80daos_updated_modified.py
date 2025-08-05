"""
Modified wrapper for analysis_80daos_updated
===========================================

This module exposes a `run_analysis` function that wraps the original
``analysis_80daos_updated.run_analysis`` but overrides the default
monthly index boundaries.  The unmodified package computes governance
z‑scores using a three‑year window starting one month prior to the
``start_label`` and ending one month before ``end_label``.  In the
original code the defaults span December 2021 through December 2024.

For the mini‑project we need a two‑year window of monthly
observations covering January 2023 through December 2024.  To
achieve this we set the ``start_label`` default to ``"2023-01-01"``
and the ``end_label`` default to ``"2025-01-01"``.  The underlying
analysis code aggregates monthly data from one month prior to
``start_label`` up to one month before ``end_label``; therefore
these defaults yield a 24‑month series spanning the 2023–2024
calendar years.  When importing this wrapper the rest of the
pipeline remains unchanged, so you still benefit from all
functionality provided by the underlying script (API access, data
processing and file output).

Example usage::

    from analysis_80daos_updated_modified import run_analysis

    run_analysis(api_key="YOUR_DEEPDAO_KEY")

You may still override ``start_label`` or ``end_label`` when calling
the function explicitly.
"""

from __future__ import annotations

from typing import Optional

# We import the original module lazily so that this wrapper remains
# lightweight.  This approach ensures that all dependencies (pandas,
# requests, etc.) are only loaded if the function is invoked.

def run_analysis(
    api_key: str,
    cmc_key: Optional[str] = None,
    price_directory: Optional[str] = None,
    start_label: str = "2023-01-01",
    end_label: str = "2025-01-01",
    max_daos: int = 80,
    top_limit: int = 120,
    verbose: bool = True,
) -> None:
    """Wrapper around :func:`analysis_80daos_updated.run_analysis`.

    The parameters mirror those of the original function, but the
    default ``start_label`` and ``end_label`` values have been updated
    to produce exactly 24 monthly observations (January 2023 through
    December 2024).  If you pass explicit values for these arguments
    they will override the defaults.

    Parameters
    ----------
    api_key : str
        DeepDAO API key (mandatory).
    cmc_key : str, optional
        CoinMarketCap API key for fetching missing price data.
    price_directory : str, optional
        Directory containing manually downloaded price CSV files.  If
        provided, price histories are loaded from this directory.
    start_label, end_label : str
        Start and end labels for monthly aggregation (YYYY‑MM‑DD).  See
        :mod:`analysis_80daos_updated` for details.
    max_daos : int
        Maximum number of DAOs to process.
    top_limit : int
        Number of top DAOs by AUM to request from DeepDAO.
    verbose : bool
        Whether to print progress messages.
    """
    # Import inside the function to avoid unnecessary overhead when
    # simply inspecting this module.
    from analysis_80daos_updated import run_analysis as _original_run_analysis

    # Forward the call with updated defaults.  Any user‑supplied
    # arguments override these defaults.
    return _original_run_analysis(
        api_key=api_key,
        cmc_key=cmc_key,
        price_directory=price_directory,
        start_label=start_label,
        end_label=end_label,
        max_daos=max_daos,
        top_limit=top_limit,
        verbose=verbose,
    )


def main() -> None:
    """Command‑line entry point.

    This replicates the argument parser from the original script but
    updates the default start and end labels.  It allows running the
    wrapper directly via ``python analysis_80daos_updated_modified.py``
    with the same interface as the original.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute governance z‑scores for top DAOs (modified for 24‑month window)"
    )
    parser.add_argument(
        "--api-key",
        required=False,
        help="DeepDAO API key (can also be set via DEEPDAO_API_KEY)",
    )
    parser.add_argument(
        "--cmc-key",
        required=False,
        help="CoinMarketCap API key (optional)",
    )
    parser.add_argument(
        "--price-dir",
        required=False,
        help="Directory containing price CSV files",
    )
    parser.add_argument(
        "--start",
        default="2023-01-01",
        help="Start label for monthly aggregation (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        default="2025-01-01",
        help="End label for monthly aggregation (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--max-daos",
        type=int,
        default=80,
        help="Maximum number of DAOs to process",
    )
    parser.add_argument(
        "--top-limit",
        type=int,
        default=120,
        help="Number of top DAOs by AUM to request",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print progress messages",
    )
    args = parser.parse_args()

    if not args.api_key:
        import os
        api_key = os.getenv("DEEPDAO_API_KEY")
        if not api_key:
            raise SystemExit("A DeepDAO API key must be provided via --api-key or the DEEPDAO_API_KEY environment variable")
        args.api_key = api_key

    run_analysis(
        api_key=args.api_key,
        cmc_key=args.cmc_key,
        price_directory=args.price_dir,
        start_label=args.start,
        end_label=args.end,
        max_daos=args.max_daos,
        top_limit=args.top_limit,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()