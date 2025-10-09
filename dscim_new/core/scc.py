"""
Social Cost of Carbon (SCC) calculation.

Integrates marginal damages and discount factors to compute the SCC.
"""

import numpy as np
import pandas as pd
import xarray as xr
from typing import Optional, List, Union


def discount_damages(
    marginal_damages: xr.DataArray,
    discount_factors: Optional[xr.DataArray] = None,
    discount_type: str = "ramsey",
    discount_rate: Optional[float] = None,
    pulse_year: int = 2020,
    discrete: bool = False,
    constant_discount_rates: Optional[List[float]] = None
) -> xr.DataArray:
    """
    Discount marginal damages to present value.

    Parameters
    ----------
    marginal_damages : xr.DataArray
        Marginal damages over time
        Must have 'year' dimension
    discount_factors : xr.DataArray, optional
        Pre-calculated discount factors (for Ramsey/GWR methods)
        If None, will use constant discounting
    discount_type : str, default "ramsey"
        Type of discounting: "constant", "ramsey", "gwr"
    discount_rate : float, optional
        Constant discount rate (required if discount_type="constant" and discount_factors=None)
    pulse_year : int, default 2020
        Base year for discounting
    discrete : bool, default False
        Use discrete discounting for constant rates
    constant_discount_rates : list of float, optional
        List of constant discount rates to evaluate (e.g., [0.015, 0.02, 0.025, 0.03])
        Creates a 'discrate' dimension

    Returns
    -------
    xr.DataArray
        Present value of marginal damages

    Examples
    --------
    >>> # Ramsey discounting (using pre-calculated factors)
    >>> pv_damages = discount_damages(
    ...     marginal_damages,
    ...     discount_factors=discount_factors,
    ...     discount_type="ramsey"
    ... )

    >>> # Constant discounting (multiple rates)
    >>> pv_damages = discount_damages(
    ...     marginal_damages,
    ...     discount_type="constant",
    ...     constant_discount_rates=[0.015, 0.02, 0.025, 0.03],
    ...     pulse_year=2020
    ... )

    Notes
    -----
    From dscim/src/dscim/menu/main_recipe.py:1088
    """
    if 'year' not in marginal_damages.dims:
        raise ValueError("marginal_damages must have 'year' dimension")

    if discount_type in ["constant", "constant_model_collapsed", "constant_gwr"]:
        # Constant discounting
        if constant_discount_rates is None:
            if discount_rate is None:
                raise ValueError(
                    "Either constant_discount_rates or discount_rate must be provided "
                    "for constant discounting"
                )
            constant_discount_rates = [discount_rate]

        # Calculate discount factors for each constant rate
        discounted_damages_list = []

        for r in constant_discount_rates:
            if discrete:
                # Discrete: DF(t) = 1 / (1 + r)^t
                df = np.power(1 + r, -(marginal_damages.year - pulse_year))
            else:
                # Continuous: DF(t) = exp(-r * t)
                df = np.exp(-r * (marginal_damages.year - pulse_year))

            # Apply discount factors
            discounted = marginal_damages * df
            discounted_damages_list.append(discounted)

        # Concatenate along discrate dimension
        pv_damages = xr.concat(
            discounted_damages_list,
            dim=pd.Index(constant_discount_rates, name="discrate")
        )

    else:
        # Ramsey/GWR discounting (using pre-calculated factors)
        if discount_factors is None:
            raise ValueError(
                f"discount_factors must be provided for {discount_type} discounting"
            )

        # Multiply damages by discount factors
        pv_damages = discount_factors * marginal_damages

    return pv_damages


def calculate_scc(
    marginal_damages: xr.DataArray,
    discount_factors: Optional[xr.DataArray] = None,
    discount_type: str = "ramsey",
    discount_rate: Optional[float] = None,
    pulse_year: int = 2020,
    discrete: bool = False,
    constant_discount_rates: Optional[List[float]] = None,
    aggregate_dims: Optional[List[str]] = None
) -> xr.DataArray:
    """
    Calculate Social Cost of Carbon (SCC).

    The SCC is the present value of marginal damages from a pulse of carbon:

    SCC = Σ_t [MD(t) × DF(t)]

    where:
    - MD(t) = marginal damages in year t
    - DF(t) = discount factor for year t
    - Σ_t = sum over all years

    Parameters
    ----------
    marginal_damages : xr.DataArray
        Marginal damages from carbon pulse
        Must have 'year' dimension
    discount_factors : xr.DataArray, optional
        Pre-calculated discount factors
        Required for Ramsey/GWR methods
    discount_type : str, default "ramsey"
        Type of discounting: "constant", "ramsey", "gwr"
    discount_rate : float, optional
        Constant discount rate (for constant discounting)
    pulse_year : int, default 2020
        Year of carbon pulse
    discrete : bool, default False
        Use discrete discounting
    constant_discount_rates : list of float, optional
        Multiple constant discount rates to evaluate
    aggregate_dims : list of str, optional
        Additional dimensions to aggregate over (e.g., ["rcp", "gas"])

    Returns
    -------
    xr.DataArray
        Social Cost of Carbon

    Examples
    --------
    >>> # Calculate SCC with Ramsey discounting
    >>> scc = calculate_scc(
    ...     marginal_damages=marginal_damages,
    ...     discount_factors=discount_factors,
    ...     discount_type="ramsey"
    ... )

    >>> # Calculate SCC with constant discounting (multiple rates)
    >>> scc = calculate_scc(
    ...     marginal_damages=marginal_damages,
    ...     discount_type="constant",
    ...     constant_discount_rates=[0.015, 0.02, 0.025, 0.03],
    ...     pulse_year=2020
    ... )

    Notes
    -----
    From dscim/src/dscim/menu/main_recipe.py:1073

    The SCC represents the economic damage caused by emitting one additional
    tonne of CO2 in the pulse year, accounting for all future damages
    discounted to present value.
    """
    # Discount damages to present value
    pv_damages = discount_damages(
        marginal_damages=marginal_damages,
        discount_factors=discount_factors,
        discount_type=discount_type,
        discount_rate=discount_rate,
        pulse_year=pulse_year,
        discrete=discrete,
        constant_discount_rates=constant_discount_rates
    )

    # Sum over time to get SCC
    scc = pv_damages.sum(dim="year")

    # Aggregate over additional dimensions if specified
    if aggregate_dims:
        dims_to_agg = [d for d in aggregate_dims if d in scc.dims]
        if dims_to_agg:
            scc = scc.mean(dim=dims_to_agg)

    # Name the result
    scc.name = "scc"

    return scc


def calculate_scc_with_uncertainty(
    marginal_damages: xr.DataArray,
    discount_factors: Optional[xr.DataArray] = None,
    discount_type: str = "ramsey",
    fair_aggregation_methods: Optional[List[str]] = None,
    fair_dims: Optional[List[str]] = None,
    include_median: bool = True,
    pulse_year: int = 2020,
    constant_discount_rates: Optional[List[float]] = None
) -> xr.DataArray:
    """
    Calculate SCC with different uncertainty aggregation methods.

    Handles different FAIR aggregation methods:
    - "ce": Certainty equivalent
    - "mean": Mean across FAIR simulations
    - "median_params": Using median climate parameters
    - "uncollapsed": Full uncertainty distribution
    - "median": Median of uncollapsed distribution

    Parameters
    ----------
    marginal_damages : xr.DataArray
        Marginal damages with 'fair_aggregation' dimension
    discount_factors : xr.DataArray, optional
        Discount factors
    discount_type : str, default "ramsey"
        Type of discounting
    fair_aggregation_methods : list of str, optional
        FAIR aggregation methods to include
        Default: ["ce", "mean", "median_params"]
    fair_dims : list of str, optional
        FAIR dimensions (e.g., ["simulation", "rcp", "gas"])
        Used for median calculation
    include_median : bool, default True
        Include median of uncollapsed distribution
    pulse_year : int, default 2020
        Year of carbon pulse
    constant_discount_rates : list of float, optional
        Constant discount rates (if using constant discounting)

    Returns
    -------
    xr.DataArray
        SCC with 'fair_aggregation' dimension

    Examples
    --------
    >>> scc = calculate_scc_with_uncertainty(
    ...     marginal_damages=marginal_damages,
    ...     discount_factors=discount_factors,
    ...     discount_type="ramsey",
    ...     fair_aggregation_methods=["ce", "mean", "median_params"],
    ...     include_median=True
    ... )

    Notes
    -----
    From dscim/src/dscim/menu/main_recipe.py:1073
    """
    if fair_aggregation_methods is None:
        fair_aggregation_methods = ["ce", "mean", "median_params"]

    if fair_dims is None:
        fair_dims = ["simulation", "rcp", "gas"]

    # Calculate SCC for each aggregation method
    scc_results = []

    for agg_method in fair_aggregation_methods:
        # Select marginal damages for this aggregation method
        if "fair_aggregation" in marginal_damages.dims:
            md = marginal_damages.sel(fair_aggregation=agg_method)
        else:
            md = marginal_damages

        # Select corresponding discount factors
        if discount_factors is not None and "fair_aggregation" in discount_factors.dims:
            df = discount_factors.sel(fair_aggregation=agg_method)
        else:
            df = discount_factors

        # Calculate SCC
        scc = calculate_scc(
            marginal_damages=md,
            discount_factors=df,
            discount_type=discount_type,
            pulse_year=pulse_year,
            constant_discount_rates=constant_discount_rates
        )

        # Add fair_aggregation coordinate
        scc = scc.assign_coords({"fair_aggregation": agg_method})
        scc_results.append(scc)

    # Concatenate results
    if scc_results:
        scc_combined = xr.concat(scc_results, dim="fair_aggregation")
    else:
        scc_combined = None

    # Add median if requested
    if include_median and "uncollapsed" in fair_aggregation_methods:
        # Get uncollapsed SCC
        uncollapsed = scc_combined.sel(fair_aggregation="uncollapsed")

        # Calculate median across FAIR dimensions
        dims_to_median = [d for d in fair_dims if d in uncollapsed.dims]
        if dims_to_median:
            median_scc = uncollapsed.median(dim=dims_to_median)
            median_scc = median_scc.assign_coords({"fair_aggregation": "median"})

            # Merge with other results
            scc_combined = xr.concat([scc_combined, median_scc], dim="fair_aggregation")

    return scc_combined


def calculate_scc_percentiles(
    scc: xr.DataArray,
    percentiles: List[float] = [5, 25, 50, 75, 95],
    uncertainty_dims: Optional[List[str]] = None
) -> xr.DataArray:
    """
    Calculate SCC percentiles across uncertainty dimensions.

    Parameters
    ----------
    scc : xr.DataArray
        SCC values with uncertainty dimensions
    percentiles : list of float, default [5, 25, 50, 75, 95]
        Percentiles to calculate (0-100)
    uncertainty_dims : list of str, optional
        Dimensions to compute percentiles over
        Default: ["simulation", "rcp", "gas", "ssp", "model"]

    Returns
    -------
    xr.DataArray
        SCC percentiles

    Examples
    --------
    >>> scc_pctiles = calculate_scc_percentiles(
    ...     scc,
    ...     percentiles=[5, 50, 95]
    ... )
    """
    if uncertainty_dims is None:
        uncertainty_dims = ["simulation", "rcp", "gas", "ssp", "model"]

    # Find which uncertainty dims exist in data
    dims_to_use = [d for d in uncertainty_dims if d in scc.dims]

    if not dims_to_use:
        raise ValueError(
            f"None of the uncertainty dimensions {uncertainty_dims} "
            f"found in SCC data with dims {scc.dims}"
        )

    # Calculate percentiles
    percentile_values = np.percentile(
        scc.values,
        percentiles,
        axis=[scc.dims.index(d) for d in dims_to_use]
    )

    # Create DataArray with percentile dimension
    result = xr.DataArray(
        percentile_values,
        dims=["percentile"] + [d for d in scc.dims if d not in dims_to_use],
        coords={
            "percentile": percentiles,
            **{d: scc[d] for d in scc.dims if d not in dims_to_use}
        }
    )

    result.name = "scc_percentiles"
    result.attrs = scc.attrs.copy()
    result.attrs['percentiles'] = percentiles
    result.attrs['uncertainty_dims'] = dims_to_use

    return result


def summarize_scc(
    scc: xr.DataArray,
    uncertainty_dims: Optional[List[str]] = None,
    include_percentiles: bool = True
) -> dict:
    """
    Summarize SCC statistics.

    Calculates mean, median, standard deviation, and percentiles.

    Parameters
    ----------
    scc : xr.DataArray
        SCC values
    uncertainty_dims : list of str, optional
        Dimensions to summarize over
    include_percentiles : bool, default True
        Include percentile calculations

    Returns
    -------
    dict
        Dictionary with 'mean', 'median', 'std', and optionally 'percentiles'

    Examples
    --------
    >>> summary = summarize_scc(scc)
    >>> print(f"Mean SCC: ${summary['mean'].values:.2f}")
    >>> print(f"Median SCC: ${summary['median'].values:.2f}")
    """
    if uncertainty_dims is None:
        uncertainty_dims = ["simulation", "rcp", "gas", "ssp", "model"]

    dims_to_use = [d for d in uncertainty_dims if d in scc.dims]

    summary = {}

    if dims_to_use:
        summary['mean'] = scc.mean(dim=dims_to_use)
        summary['median'] = scc.median(dim=dims_to_use)
        summary['std'] = scc.std(dim=dims_to_use)

        if include_percentiles:
            summary['percentiles'] = calculate_scc_percentiles(
                scc,
                uncertainty_dims=dims_to_use
            )
    else:
        # No uncertainty dimensions to aggregate
        summary['mean'] = scc
        summary['median'] = scc
        summary['std'] = xr.zeros_like(scc)

    return summary
