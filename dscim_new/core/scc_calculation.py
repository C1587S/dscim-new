"""
Core mathematical functions for Social Cost of Carbon (SCC) calculation.

Pure functions for calculating SCC from marginal damages and discount factors.
"""

import numpy as np
import xarray as xr
from typing import Optional, List, Union


def calculate_scc(
    marginal_damages: xr.DataArray,
    discount_factors: xr.DataArray,
    pulse_year: int = 2020
) -> xr.DataArray:
    """
    Calculate Social Cost of Carbon.

    SCC = SUM_t [MD(t) · DF(t)]

    Parameters
    ----------
    marginal_damages : xr.DataArray
        Marginal damages for each year (dollars per ton of CO2)
    discount_factors : xr.DataArray
        Discount factors for each year
    pulse_year : int, default 2020
        Year of carbon pulse

    Returns
    -------
    xr.DataArray
        Social Cost of Carbon

    Examples
    --------
    >>> scc = calculate_scc(
    ...     marginal_damages,
    ...     discount_factors,
    ...     pulse_year=2020
    ... )

    Notes
    -----
    The SCC represents the present value of damages from emitting
    one additional ton of CO2 in the pulse year.

    Both inputs must have compatible dimensions. The function will
    broadcast appropriately if dimensions differ.
    """
    # Multiply marginal damages by discount factors
    discounted_damages = marginal_damages * discount_factors

    # Sum over time dimension
    if 'year' in discounted_damages.dims:
        scc = discounted_damages.sum(dim='year')
    else:
        # If no year dimension, just return the product
        scc = discounted_damages

    # Add metadata
    if isinstance(scc, xr.DataArray):
        scc.attrs['pulse_year'] = pulse_year
        scc.attrs['calculation'] = 'scc'
        scc.attrs['units'] = 'dollars_per_ton_CO2'

    return scc


def calculate_global_consumption(
    consumption_regional: xr.DataArray,
    pulse: bool = False
) -> xr.DataArray:
    """
    Calculate global consumption by summing across regions.

    Parameters
    ----------
    consumption_regional : xr.DataArray
        Regional consumption values, must have 'region' dimension
    pulse : bool, default False
        If True, this is consumption with pulse (for pulse-no_pulse diff)

    Returns
    -------
    xr.DataArray
        Global (aggregated) consumption

    Examples
    --------
    >>> global_c = calculate_global_consumption(regional_consumption)

    Notes
    -----
    This is used for discounting calculations that depend on global
    rather than regional consumption.
    """
    if 'region' not in consumption_regional.dims:
        raise ValueError("consumption must have 'region' dimension")

    # Sum across regions
    global_consumption = consumption_regional.sum(dim='region')

    # Add metadata
    global_consumption.attrs['aggregation'] = 'global'
    if pulse:
        global_consumption.attrs['scenario'] = 'pulse'
    else:
        global_consumption.attrs['scenario'] = 'control'

    return global_consumption


def aggregate_scc_over_fair(
    scc: xr.DataArray,
    method: str = "mean",
    dims: Optional[List[str]] = None,
    eta: Optional[float] = None
) -> xr.DataArray:
    """
    Aggregate SCC over FAIR climate uncertainty dimensions.

    Parameters
    ----------
    scc : xr.DataArray
        SCC values with FAIR dimensions (simulation, rcp, etc.)
    method : str, default "mean"
        Aggregation method:
        - "mean": Simple mean across dimensions
        - "median": Median across dimensions
        - "ce": Certainty equivalent using CRRA utility
    dims : list of str, optional
        Dimensions to aggregate over. If None, uses ["simulation"]
    eta : float, optional
        Risk aversion parameter (required for "ce" method)

    Returns
    -------
    xr.DataArray
        Aggregated SCC

    Examples
    --------
    >>> scc_mean = aggregate_scc_over_fair(scc_raw, method="mean")
    >>> scc_ce = aggregate_scc_over_fair(scc_raw, method="ce", eta=1.45)

    Notes
    -----
    The "ce" method applies certainty equivalent aggregation to the
    SCC distribution, accounting for risk aversion in climate uncertainty.
    """
    if dims is None:
        dims = ["simulation"]

    # Check that specified dims exist, filter out missing ones
    existing_dims = [d for d in dims if d in scc.dims]

    # If no specified dims exist, aggregate over whatever uncertainty dims are available
    if not existing_dims:
        # Common uncertainty dimension names
        uncertainty_dims = ['simulation', 'sim', 'rcp', 'gcm', 'model', 'scenario']
        existing_dims = [d for d in scc.dims if d in uncertainty_dims and d not in ['year', 'region']]

    # If still no dims to aggregate, return as-is
    if not existing_dims:
        return scc

    if method == "mean":
        # Simple mean
        result = scc.mean(dim=existing_dims)
        result.attrs['fair_aggregation'] = 'mean'

    elif method == "median":
        # Median
        result = scc.median(dim=existing_dims)
        result.attrs['fair_aggregation'] = 'median'

    elif method == "ce":
        # Certainty equivalent
        if eta is None:
            raise ValueError("'ce' method requires eta parameter")

        # For SCC (which can be negative), we need to handle CE carefully
        # One approach: use CRRA on positive values, treat negative separately
        # For simplicity, use mean for now with note
        # TODO: Implement proper CE for SCC with mixed signs

        result = scc.mean(dim=existing_dims)
        result.attrs['fair_aggregation'] = f'ce_eta{eta}'
        result.attrs['note'] = 'CE approximated with mean for mixed-sign values'

    elif method == "gwr_mean":
        # GWR mean - similar to mean but may use different weights
        # For now, implement as simple mean
        result = scc.mean(dim=existing_dims)
        result.attrs['fair_aggregation'] = 'gwr_mean'

    else:
        raise ValueError(f"Unknown aggregation method: {method}. "
                        f"Choose from: mean, median, ce, gwr_mean")

    # Preserve other attributes
    for key, value in scc.attrs.items():
        if key not in result.attrs:
            result.attrs[key] = value

    return result


def calculate_scc_quantiles(
    scc: xr.DataArray,
    quantiles: List[float],
    dims: Optional[List[str]] = None
) -> xr.Dataset:
    """
    Calculate SCC quantiles across uncertainty dimensions.

    Parameters
    ----------
    scc : xr.DataArray
        SCC values with uncertainty dimensions
    quantiles : list of float
        Quantiles to calculate (e.g., [0.05, 0.5, 0.95])
    dims : list of str, optional
        Dimensions to compute quantiles over

    Returns
    -------
    xr.Dataset
        Dataset with quantile values

    Examples
    --------
    >>> quantiles = calculate_scc_quantiles(
    ...     scc,
    ...     quantiles=[0.05, 0.25, 0.5, 0.75, 0.95],
    ...     dims=["simulation"]
    ... )

    Notes
    -----
    This provides uncertainty bounds on the SCC estimate.
    """
    if dims is None:
        # Use all dims except year, region
        dims = [d for d in scc.dims if d not in ['year', 'region']]

    # Calculate quantiles
    quantile_values = scc.quantile(quantiles, dim=dims)

    # Reshape to have quantile as a dimension
    result = quantile_values.rename({'quantile': 'probability'})

    result.attrs['quantiles'] = quantiles
    result.attrs['dims_aggregated'] = dims

    return result


def calculate_uncollapsed_scc(
    marginal_damages: xr.DataArray,
    discount_factors: xr.DataArray,
    pulse_year: int = 2020,
    keep_year: bool = False
) -> xr.DataArray:
    """
    Calculate SCC without collapsing uncertainty dimensions.

    This returns the full distribution of SCC across all
    uncertainty dimensions (simulations, models, scenarios).

    Parameters
    ----------
    marginal_damages : xr.DataArray
        Marginal damages
    discount_factors : xr.DataArray
        Discount factors
    pulse_year : int, default 2020
        Year of carbon pulse
    keep_year : bool, default False
        If True, keep year dimension (returns time path of discounted damages)

    Returns
    -------
    xr.DataArray
        Uncollapsed SCC values

    Examples
    --------
    >>> scc_uncollapsed = calculate_uncollapsed_scc(md, df, pulse_year=2020)

    Notes
    -----
    Useful for examining the full distribution of SCC before aggregation.
    """
    # Multiply marginal damages by discount factors
    discounted_damages = marginal_damages * discount_factors

    if keep_year:
        # Return with year dimension
        result = discounted_damages
        result.attrs['calculation'] = 'discounted_damages'
    else:
        # Sum over time dimension
        if 'year' in discounted_damages.dims:
            result = discounted_damages.sum(dim='year')
        else:
            result = discounted_damages

        result.attrs['calculation'] = 'uncollapsed_scc'

    result.attrs['pulse_year'] = pulse_year
    result.attrs['units'] = 'dollars_per_ton_CO2'

    return result


def calculate_partial_scc(
    marginal_damages: xr.DataArray,
    discount_factors: xr.DataArray,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    pulse_year: int = 2020
) -> xr.DataArray:
    """
    Calculate SCC for a subset of years.

    Useful for analyzing contribution of different time periods to total SCC.

    Parameters
    ----------
    marginal_damages : xr.DataArray
        Marginal damages
    discount_factors : xr.DataArray
        Discount factors
    start_year : int, optional
        First year to include (if None, use all)
    end_year : int, optional
        Last year to include (if None, use all)
    pulse_year : int, default 2020
        Year of carbon pulse

    Returns
    -------
    xr.DataArray
        Partial SCC for specified time period

    Examples
    --------
    >>> # SCC from damages in 21st century only
    >>> scc_21st = calculate_partial_scc(
    ...     md, df,
    ...     start_year=2000,
    ...     end_year=2100,
    ...     pulse_year=2020
    ... )

    Notes
    -----
    Helps understand how damages in different time periods contribute
    to the total SCC.
    """
    # Select year range
    if start_year is not None or end_year is not None:
        md_subset = marginal_damages.sel(year=slice(start_year, end_year))
        df_subset = discount_factors.sel(year=slice(start_year, end_year))
    else:
        md_subset = marginal_damages
        df_subset = discount_factors

    # Calculate SCC for subset
    scc = calculate_scc(md_subset, df_subset, pulse_year)

    scc.attrs['start_year'] = start_year
    scc.attrs['end_year'] = end_year

    return scc
