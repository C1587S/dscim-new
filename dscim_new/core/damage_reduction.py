"""
Core damage reduction mathematical functions.

Functions for calculating consumption under different climate scenarios
and aggregating damages (No I/O operations).
"""

import numpy as np
import xarray as xr
from typing import Optional

from .utils import crra_certainty_equivalent, mean_aggregate


def calculate_no_cc_consumption(
    gdppc: xr.DataArray,
    histclim: xr.DataArray,
    histclim_mean: xr.DataArray
) -> xr.DataArray:
    """
    Calculate consumption under no climate change (counterfactual) scenario.

    Formula: consumption = gdppc + mean(histclim_damages) - histclim_damages

    This represents what consumption would be if we removed the impact of
    historical climate variability, creating a counterfactual baseline.

    Parameters
    ----------
    gdppc : xr.DataArray
        GDP per capita
    histclim : xr.DataArray
        Historical climate damages (with batch dimension)
    histclim_mean : xr.DataArray
        Mean historical climate damages across batches

    Returns
    -------
    xr.DataArray
        Consumption under no climate change scenario

    Examples
    --------
    >>> gdppc = xr.DataArray([100, 100], dims=["region"])
    >>> histclim = xr.DataArray([[95, 105], [90, 110]], dims=["batch", "region"])
    >>> histclim_mean = histclim.mean("batch")  # [92.5, 107.5]
    >>> consumption = calculate_no_cc_consumption(gdppc, histclim, histclim_mean)

    Notes
    -----
    This corresponds to the "no_cc" reduction in the original DSCIM:
        calculation = gdppc + chunk[histclim].mean("batch") - chunk[histclim]
    """
    return gdppc + histclim_mean - histclim


def calculate_cc_consumption(
    gdppc: xr.DataArray,
    delta: xr.DataArray
) -> xr.DataArray:
    """
    Calculate consumption under climate change scenario.

    Formula: consumption = gdppc - delta_damages

    This represents consumption after accounting for additional damages
    from climate change beyond the historical baseline.

    Parameters
    ----------
    gdppc : xr.DataArray
        GDP per capita
    delta : xr.DataArray
        Delta damages (additional damages from climate change)

    Returns
    -------
    xr.DataArray
        Consumption under climate change scenario

    Examples
    --------
    >>> gdppc = xr.DataArray([100, 100], dims=["region"])
    >>> delta = xr.DataArray([[5, 10], [3, 8]], dims=["batch", "region"])
    >>> consumption = calculate_cc_consumption(gdppc, delta)

    Notes
    -----
    This corresponds to the "cc" reduction in the original DSCIM:
        calculation = gdppc - chunk[delta]
    """
    return gdppc - delta


def apply_bottom_coding(
    consumption: xr.DataArray,
    bottom_code: float
) -> xr.DataArray:
    """
    Apply minimum GDP per capita threshold (bottom coding).

    This ensures that consumption never falls below a specified minimum,
    which is important for:
    1. Numerical stability (avoiding log(negative) or division by zero)
    2. Economic realism (minimum subsistence level)

    Formula: consumption = max(consumption, bottom_code)

    Parameters
    ----------
    consumption : xr.DataArray
        Consumption values to bottom-code
    bottom_code : float
        Minimum allowed GDP per capita value (default in DSCIM: 39.39265060424805)

    Returns
    -------
    xr.DataArray
        Bottom-coded consumption

    Examples
    --------
    >>> consumption = xr.DataArray([20, 50, 100], dims=["region"])
    >>> bottom_coded = apply_bottom_coding(consumption, bottom_code=40.0)
    >>> bottom_coded
    <xr.DataArray ([40, 50, 100])>

    Notes
    -----
    From original DSCIM:
        calculation = np.maximum(calculation, bottom_code)
    """
    return xr.ufuncs.maximum(consumption, bottom_code)


def aggregate_adding_up(
    consumption: xr.DataArray,
    batch_dim: str = "batch"
) -> xr.DataArray:
    """
    Aggregate consumption using simple mean (adding up recipe).

    The "adding up" recipe treats all uncertainty realizations equally
    and takes a simple mean across the batch dimension.

    Parameters
    ----------
    consumption : xr.DataArray
        Consumption data with batch dimension
    batch_dim : str, optional
        Name of batch dimension to aggregate over (default: "batch")

    Returns
    -------
    xr.DataArray
        Mean consumption across batches

    Examples
    --------
    >>> consumption = xr.DataArray([80, 100, 120], dims=["batch"])
    >>> result = aggregate_adding_up(consumption)
    >>> result
    <xr.DataArray ()>
    array(100.0)

    Notes
    -----
    This corresponds to the "adding_up" recipe in original DSCIM:
        result = mean_func(calculation, "batch")
    """
    return mean_aggregate(consumption, batch_dim)


def aggregate_risk_aversion(
    consumption: xr.DataArray,
    eta: float,
    batch_dim: str = "batch"
) -> xr.DataArray:
    """
    Aggregate consumption using CRRA utility (risk aversion recipe).

    The "risk_aversion" recipe accounts for risk aversion by applying
    CRRA utility before aggregating. This gives more weight to bad outcomes
    (lower consumption) when eta > 1.

    Parameters
    ----------
    consumption : xr.DataArray
        Consumption data with batch dimension
    eta : float
        Risk aversion parameter (must be > 0)
        - eta = 1: log utility
        - eta > 1: increasing risk aversion
        - eta < 1: risk tolerance
    batch_dim : str, optional
        Name of batch dimension to aggregate over (default: "batch")

    Returns
    -------
    xr.DataArray
        Certainty equivalent consumption

    Examples
    --------
    >>> consumption = xr.DataArray([80, 100, 120], dims=["batch"])
    >>> result = aggregate_risk_aversion(consumption, eta=2.0)

    Notes
    -----
    This corresponds to the "risk_aversion" recipe in original DSCIM:
        result = ce_func(calculation, "batch", eta=eta)

    The risk aversion parameter eta determines how much to penalize uncertainty:
    - Higher eta = more risk averse = lower certainty equivalent
    - Lower eta = less risk averse = closer to mean
    """
    if eta is None:
        raise ValueError("eta must be specified for risk_aversion aggregation")

    return crra_certainty_equivalent(consumption, eta=eta, dims=batch_dim)