"""
Core mathematical functions for equity-weighted damage aggregation.

The equity recipe weights damages by consumption levels to account for
diminishing marginal utility across regions with different income levels.

Pure functions with no I/O or state.
"""

import numpy as np
import xarray as xr
from typing import Optional, Union


def calculate_equity_weights(
    consumption: xr.DataArray,
    eta: float,
    equity_dimension: str = "region"
) -> xr.DataArray:
    """
    Calculate equity weights based on consumption levels.

    Equity weight for region r: w_r = (C_r / C_global)^(-eta)

    Where:
    - C_r is consumption in region r
    - C_global is global average consumption
    - eta is the inequality aversion parameter

    Higher eta means more weight on damages in poor regions.

    Parameters
    ----------
    consumption : xr.DataArray
        Consumption by region (typically GDP per capita)
    eta : float
        Inequality aversion parameter (elasticity of marginal utility)
        Typical values: 0.5 to 3.0
    equity_dimension : str, default "region"
        Dimension along which to calculate equity weights

    Returns
    -------
    xr.DataArray
        Equity weights for each region, normalized to sum to 1

    Examples
    --------
    >>> consumption = xr.DataArray(
    ...     [10000, 50000, 100000],  # GDP per capita by region
    ...     dims=['region'],
    ...     coords={'region': ['poor', 'middle', 'rich']}
    ... )
    >>> weights = calculate_equity_weights(consumption, eta=1.5)
    >>> # Poor regions get higher weight, rich regions lower weight

    Notes
    -----
    The equity weighting reflects that an additional dollar of damages
    matters more to poor regions than rich regions due to diminishing
    marginal utility of consumption.

    References
    ----------
    Anthoff, D., & Emmerling, J. (2019). Inequality and the social cost
    of carbon. Journal of the Association of Environmental and Resource
    Economists, 6(2), 243-273.
    """
    # Calculate global average consumption
    # Weight by dimension size to get proper average
    if equity_dimension in consumption.dims:
        global_avg = consumption.mean(dim=equity_dimension)
    else:
        raise ValueError(
            f"Dimension '{equity_dimension}' not found in consumption data. "
            f"Available dimensions: {list(consumption.dims)}"
        )

    # Calculate relative consumption: C_r / C_global
    relative_consumption = consumption / global_avg

    # Calculate equity weights: (C_r / C_global)^(-eta)
    equity_weights = np.power(relative_consumption, -eta)

    # Normalize weights to sum to 1 along equity dimension
    # This ensures that total damages are preserved when aggregating
    weights_sum = equity_weights.sum(dim=equity_dimension)
    normalized_weights = equity_weights / weights_sum

    # Add metadata
    normalized_weights.attrs['equity_eta'] = eta
    normalized_weights.attrs['equity_dimension'] = equity_dimension
    normalized_weights.attrs['description'] = 'Equity weights based on consumption'

    return normalized_weights


def aggregate_equity(
    damages: xr.DataArray,
    consumption: xr.DataArray,
    eta: float,
    equity_dimension: str = "region"
) -> xr.DataArray:
    """
    Aggregate damages using equity weights based on consumption.

    This implements the "equity recipe" which weights damages by
    consumption levels to account for diminishing marginal utility.

    Formula: D_equity = Σ_r w_r * D_r

    where w_r = (C_r / C_global)^(-eta) / Σ_r (C_r / C_global)^(-eta)

    Parameters
    ----------
    damages : xr.DataArray
        Damages by region and other dimensions
    consumption : xr.DataArray
        Consumption by region (must have equity_dimension)
    eta : float
        Inequality aversion parameter
        - eta = 0: No equity weighting (simple average)
        - eta = 1: Inverse proportional to consumption
        - eta > 1: Higher weight on poor regions
    equity_dimension : str, default "region"
        Dimension along which to apply equity weights and aggregate

    Returns
    -------
    xr.DataArray
        Equity-weighted aggregated damages

    Examples
    --------
    >>> damages = xr.DataArray(
    ...     [[100, 200, 300],   # Year 2020
    ...      [110, 220, 330]],  # Year 2021
    ...     dims=['year', 'region'],
    ...     coords={'year': [2020, 2021], 'region': ['poor', 'middle', 'rich']}
    ... )
    >>> consumption = xr.DataArray(
    ...     [10000, 50000, 100000],
    ...     dims=['region'],
    ...     coords={'region': ['poor', 'middle', 'rich']}
    ... )
    >>> equity_damages = aggregate_equity(damages, consumption, eta=1.5)
    >>> # Result is weighted sum across regions for each year

    Notes
    -----
    The consumption data must be broadcastable to the damages data.
    Typically consumption varies by region, year, and SSP scenario.

    If consumption varies over time, equity weights are calculated
    separately for each time period.

    References
    ----------
    Anthoff, D., & Emmerling, J. (2019). Inequality and the social cost
    of carbon. Journal of the Association of Environmental and Resource
    Economists, 6(2), 243-273.
    """
    # Validate inputs
    if equity_dimension not in damages.dims:
        raise ValueError(
            f"Dimension '{equity_dimension}' not found in damages data. "
            f"Available dimensions: {list(damages.dims)}"
        )

    if equity_dimension not in consumption.dims:
        raise ValueError(
            f"Dimension '{equity_dimension}' not found in consumption data. "
            f"Available dimensions: {list(consumption.dims)}"
        )

    # Align consumption with damages dimensions
    # This handles cases where consumption has fewer dimensions than damages
    consumption_aligned, damages_aligned = xr.align(consumption, damages, join='right')

    # Calculate equity weights
    equity_weights = calculate_equity_weights(
        consumption_aligned,
        eta=eta,
        equity_dimension=equity_dimension
    )

    # Apply weights and aggregate
    weighted_damages = damages_aligned * equity_weights
    aggregated = weighted_damages.sum(dim=equity_dimension)

    # Add metadata
    aggregated.attrs['aggregation_method'] = 'equity'
    aggregated.attrs['equity_eta'] = eta
    aggregated.attrs['equity_dimension'] = equity_dimension
    aggregated.attrs['description'] = 'Equity-weighted aggregated damages'

    return aggregated


def compare_equity_vs_adding_up(
    damages: xr.DataArray,
    consumption: xr.DataArray,
    eta: float,
    equity_dimension: str = "region"
) -> dict:
    """
    Compare equity weighting vs simple averaging (adding_up).

    Useful for understanding the impact of equity weights on final damages.

    Parameters
    ----------
    damages : xr.DataArray
        Damages by region
    consumption : xr.DataArray
        Consumption by region
    eta : float
        Inequality aversion parameter
    equity_dimension : str, default "region"
        Dimension along which to aggregate

    Returns
    -------
    dict
        Dictionary with keys:
        - 'equity': Equity-weighted damages
        - 'adding_up': Simple average damages
        - 'ratio': Ratio of equity to adding_up
        - 'difference': Absolute difference
        - 'pct_difference': Percentage difference

    Examples
    --------
    >>> comparison = compare_equity_vs_adding_up(damages, consumption, eta=1.5)
    >>> print(f"Equity increases damages by {comparison['pct_difference']}%")

    Notes
    -----
    When poor regions have higher per-capita damages, equity weighting
    typically increases total damages compared to simple averaging.
    """
    # Calculate equity-weighted damages
    equity_damages = aggregate_equity(
        damages, consumption, eta, equity_dimension
    )

    # Calculate simple average (adding_up)
    adding_up_damages = damages.mean(dim=equity_dimension)

    # Calculate comparisons
    ratio = equity_damages / adding_up_damages
    difference = equity_damages - adding_up_damages
    pct_difference = (difference / adding_up_damages) * 100

    return {
        'equity': equity_damages,
        'adding_up': adding_up_damages,
        'ratio': ratio,
        'difference': difference,
        'pct_difference': pct_difference
    }


def validate_equity_weights(
    weights: xr.DataArray,
    equity_dimension: str = "region",
    tolerance: float = 1e-6
) -> bool:
    """
    Validate that equity weights are properly normalized.

    Checks:
    1. All weights are positive
    2. Weights sum to 1.0 along equity dimension
    3. Weights are finite (no NaN or inf)

    Parameters
    ----------
    weights : xr.DataArray
        Equity weights to validate
    equity_dimension : str, default "region"
        Dimension along which weights should sum to 1
    tolerance : float, default 1e-6
        Tolerance for sum-to-one check

    Returns
    -------
    bool
        True if weights are valid

    Raises
    ------
    ValueError
        If weights fail validation with detailed error message

    Examples
    --------
    >>> weights = calculate_equity_weights(consumption, eta=1.5)
    >>> validate_equity_weights(weights)  # Returns True if valid
    """
    # Check for finite values
    if not np.all(np.isfinite(weights.values)):
        raise ValueError(
            "Equity weights contain NaN or infinite values. "
            "Check that consumption data is positive and finite."
        )

    # Check for positive values
    if not np.all(weights.values > 0):
        raise ValueError(
            "Equity weights must be positive. "
            "Check that consumption data is positive."
        )

    # Check sum to 1.0
    weights_sum = weights.sum(dim=equity_dimension)
    if not np.allclose(weights_sum.values, 1.0, atol=tolerance):
        max_deviation = np.max(np.abs(weights_sum.values - 1.0))
        raise ValueError(
            f"Equity weights do not sum to 1.0 along '{equity_dimension}'. "
            f"Maximum deviation: {max_deviation:.2e}. "
            f"Tolerance: {tolerance:.2e}"
        )

    return True
