"""
Core utility functions for DSCIM calculations.

Pure mathematical functions with no I/O dependencies.
"""

import numpy as np
import xarray as xr
from typing import Union, List


def power(base: Union[float, np.ndarray, xr.DataArray], exponent: float) -> Union[float, np.ndarray, xr.DataArray]:
    """
    Safe power function that handles negative values.

    This function is needed because standard power operations can fail
    with negative values and non-integer exponents.

    Parameters
    ----------
    base : float, np.ndarray, or xr.DataArray
        Base value(s) to raise to power
    exponent : float
        Exponent to raise base to

    Returns
    -------
    float, np.ndarray, or xr.DataArray
        base ** exponent

    Examples
    --------
    >>> power(2.0, 3.0)
    8.0
    >>> power(np.array([1, 2, 3]), 2.0)
    array([1., 4., 9.])
    """
    if isinstance(base, xr.DataArray):
        return base ** exponent
    elif isinstance(base, np.ndarray):
        return np.power(base, exponent)
    else:
        return base ** exponent


def crra_certainty_equivalent(
    consumption: xr.DataArray,
    eta: float,
    dims: Union[str, List[str]]
) -> xr.DataArray:
    """
    Calculate certainty equivalent using CRRA (Constant Relative Risk Aversion) utility.

    The CRRA utility function is:
    - When eta = 1: U(c) = log(c), CE = exp(E[log(c)])
    - When eta ≠ 1: U(c) = (c^(1-eta) - 1)/(1-eta), CE = [(1-eta) * E[U(c)] + 1]^(1/(1-eta))

    Simplified for eta ≠ 1: CE = [E[c^(1-eta)]]^(1/(1-eta))

    Parameters
    ----------
    consumption : xr.DataArray
        Consumption data array
    eta : float
        Risk aversion parameter (eta >= 0)
        - eta = 0: risk neutral (simple mean)
        - eta = 1: log utility
        - eta > 1: risk averse
    dims : str or list of str
        Dimension(s) to aggregate over

    Returns
    -------
    xr.DataArray
        Certainty equivalent consumption

    Examples
    --------
    >>> consumption = xr.DataArray([80, 100, 120], dims=["batch"])
    >>> ce = crra_certainty_equivalent(consumption, eta=1.5, dims="batch")

    Notes
    -----
    This implements the formula from the original DSCIM:
    - For eta = 1: exp(mean(log(c)))
    - For eta ≠ 1: [mean(c^(1-eta))]^(1/(1-eta))
    """
    if isinstance(dims, str):
        dims = [dims]

    if eta == 1.0:
        # Log utility case: CE = exp(E[log(c)])
        return xr.ufuncs.exp(xr.ufuncs.log(consumption).mean(dims))
    else:
        # Power utility case: CE = [E[c^(1-eta)]]^(1/(1-eta))
        utility = power(consumption, (1 - eta))
        mean_utility = utility.mean(dims)
        return power(mean_utility, (1 / (1 - eta)))


def mean_aggregate(
    data: xr.DataArray,
    dims: Union[str, List[str]]
) -> xr.DataArray:
    """
    Simple mean aggregation over specified dimensions.

    This is a wrapper for xr.DataArray.mean() to maintain consistency
    with other aggregation functions.

    Parameters
    ----------
    data : xr.DataArray
        Data to aggregate
    dims : str or list of str
        Dimension(s) to aggregate over

    Returns
    -------
    xr.DataArray
        Mean aggregated data

    Examples
    --------
    >>> data = xr.DataArray([1, 2, 3, 4], dims=["batch"])
    >>> mean_aggregate(data, "batch")
    <xr.DataArray ()>
    array(2.5)
    """
    if isinstance(dims, str):
        dims = [dims]

    return data.mean(dims)