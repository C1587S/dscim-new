"""
Global consumption aggregation and extrapolation.

Handles calculation of global consumption for different discount types,
following the original dscim approach.
"""

import numpy as np
import xarray as xr
from typing import Optional, Literal


def aggregate_global_consumption(
    gdp: xr.DataArray,
    discount_type: str,
    pop: Optional[xr.DataArray] = None,
    region_dim: str = "region"
) -> xr.DataArray:
    """
    Aggregate GDP across regions to calculate global consumption.

    Different discount types require different aggregation methods:
    - constant/ramsey: Sum over regions, keep ssp/model separate
    - constant_model_collapsed: Sum over regions, then average over models
    - gwr: Sum over regions, then average over both model and ssp

    Parameters
    ----------
    gdp : xr.DataArray
        GDP data with dimensions (year, ssp, model, region, ...)
    discount_type : str
        Type of discounting: "constant", "ramsey", "gwr", or variations
    pop : xr.DataArray, optional
        Population data (needed for per-capita calculations)
    region_dim : str, default "region"
        Name of the region dimension to aggregate over

    Returns
    -------
    xr.DataArray
        Global consumption aggregated appropriately for the discount type

    Examples
    --------
    >>> global_cons = aggregate_global_consumption(
    ...     gdp,
    ...     discount_type="constant"
    ... )
    >>> # Result has dimensions: (year, ssp, model)

    >>> global_cons_gwr = aggregate_global_consumption(
    ...     gdp,
    ...     discount_type="gwr_gwr"
    ... )
    >>> # Result has dimensions: (year,) - collapsed across ssp and model

    Notes
    -----
    This follows the pattern from dscim/src/dscim/menu/baseline.py:32
    """
    # First, sum over regions
    if region_dim in gdp.dims:
        global_cons = gdp.sum(dim=region_dim)
    else:
        global_cons = gdp

    # Then apply discount-type specific aggregation
    if discount_type == "constant" or "ramsey" in discount_type:
        # Keep ssp and model separate
        pass

    elif discount_type == "constant_model_collapsed":
        # Average over models, keep ssp
        if "model" in global_cons.dims:
            global_cons = global_cons.mean(dim="model")

    elif "gwr" in discount_type:
        # Average over both model and ssp (Weitzman-style averaging)
        dims_to_collapse = []
        if "model" in global_cons.dims:
            dims_to_collapse.append("model")
        if "ssp" in global_cons.dims:
            dims_to_collapse.append("ssp")

        if dims_to_collapse:
            global_cons = global_cons.mean(dim=dims_to_collapse)

    global_cons.name = f"global_consumption_{discount_type}"

    return global_cons


def extrapolate_global_consumption(
    gdp: xr.DataArray,
    pop: xr.DataArray,
    discount_type: str,
    start_year: int = 2085,
    end_year: int = 2099,
    target_year: int = 2300,
    method: Literal["growth_constant", "linear"] = "growth_constant",
    region_dim: str = "region"
) -> xr.DataArray:
    """
    Extrapolate global consumption beyond the projection period.

    This calculates global consumption out to `target_year` (e.g., 2300) by:
    1. Calculating per-capita consumption growth rate from historical period
    2. Holding population constant at the last available year
    3. Projecting consumption using the growth rate

    Parameters
    ----------
    gdp : xr.DataArray
        GDP data with dimensions (year, ssp, model, region)
    pop : xr.DataArray
        Population data with same dimensions
    discount_type : str
        Type of discounting (determines aggregation method)
    start_year : int, default 2085
        Start year for calculating growth rate
    end_year : int, default 2099
        End year for calculating growth rate (last year of projections)
    target_year : int, default 2300
        Final year to extrapolate to
    method : str, default "growth_constant"
        Extrapolation method: "growth_constant" or "linear"
    region_dim : str, default "region"
        Name of the region dimension

    Returns
    -------
    xr.DataArray
        Global consumption extrapolated to target_year with discount_type dimension

    Examples
    --------
    >>> global_cons = extrapolate_global_consumption(
    ...     gdp,
    ...     pop,
    ...     discount_type="euler_ramsey",
    ...     target_year=2300
    ... )

    Notes
    -----
    This follows the pattern from dscim/src/dscim/menu/main_recipe.py:838
    """
    # Check if GDP already extends to target year
    if target_year in gdp.year:
        # No extrapolation needed
        global_cons = aggregate_global_consumption(gdp, discount_type, pop, region_dim)
    else:
        # Need to extrapolate

        # Step 1: Collapse population according to discount type
        collapsed_pop = _collapse_population(pop, discount_type, region_dim)

        # Step 2: Extend population to target year (hold constant at last value)
        pop_extended = collapsed_pop.sum(region_dim)
        pop_extended = pop_extended.reindex(
            year=range(int(pop_extended.year.min()), target_year + 1),
            method="ffill"
        )

        # Step 3: Calculate per-capita consumption with extrapolation
        global_cons_pc = _calculate_global_consumption_per_capita(
            gdp,
            collapsed_pop,
            discount_type,
            start_year,
            end_year,
            target_year,
            method,
            region_dim
        )

        # Step 4: Convert back to total consumption
        global_cons = global_cons_pc * pop_extended

    # Add discount_type dimension
    global_cons = global_cons.expand_dims({"discount_type": [discount_type]})

    return global_cons


def _collapse_population(
    pop: xr.DataArray,
    discount_type: str,
    region_dim: str = "region"
) -> xr.DataArray:
    """
    Collapse population dimensions according to discount type.

    Parameters
    ----------
    pop : xr.DataArray
        Population data
    discount_type : str
        Type of discounting
    region_dim : str
        Region dimension name

    Returns
    -------
    xr.DataArray
        Collapsed population

    Notes
    -----
    From dscim/src/dscim/menu/main_recipe.py:449
    """
    if discount_type == "constant" or "ramsey" in discount_type:
        # Keep all dimensions
        collapsed = pop

    elif discount_type == "constant_model_collapsed":
        # Average over models
        if "model" in pop.dims:
            collapsed = pop.mean("model")
        else:
            collapsed = pop

    elif "gwr" in discount_type:
        # Average over model and ssp
        dims_to_collapse = []
        if "model" in pop.dims:
            dims_to_collapse.append("model")
        if "ssp" in pop.dims:
            dims_to_collapse.append("ssp")

        if dims_to_collapse:
            collapsed = pop.mean(dims_to_collapse)
        else:
            collapsed = pop
    else:
        collapsed = pop

    return collapsed


def _calculate_global_consumption_per_capita(
    gdp: xr.DataArray,
    pop: xr.DataArray,
    discount_type: str,
    start_year: int,
    end_year: int,
    target_year: int,
    method: str,
    region_dim: str
) -> xr.DataArray:
    """
    Calculate global consumption per capita with extrapolation.

    Parameters
    ----------
    gdp : xr.DataArray
        GDP data
    pop : xr.DataArray
        Population data (already collapsed)
    discount_type : str
        Type of discounting
    start_year : int
        Start year for growth calculation
    end_year : int
        End year for growth calculation
    target_year : int
        Target year for extrapolation
    method : str
        Extrapolation method
    region_dim : str
        Region dimension name

    Returns
    -------
    xr.DataArray
        Per-capita consumption extrapolated to target year
    """
    # Aggregate GDP to global level
    global_gdp = aggregate_global_consumption(gdp, discount_type, pop, region_dim)

    # Calculate per-capita
    pop_sum = pop.sum(region_dim)
    gdp_pc = global_gdp / pop_sum

    # Select historical period (up to end_year)
    gdp_pc_hist = gdp_pc.sel(year=slice(None, end_year))

    # Extrapolate
    if method == "growth_constant":
        # Calculate growth rate from start_year to end_year
        gdp_pc_start = gdp_pc.sel(year=start_year)
        gdp_pc_end = gdp_pc.sel(year=end_year)

        # Number of years
        n_years = end_year - start_year

        # Compound annual growth rate: (final/initial)^(1/n_years) - 1
        growth_rate = np.power(gdp_pc_end / gdp_pc_start, 1.0 / n_years) - 1

        # Create extrapolated years
        extrap_years = range(end_year + 1, target_year + 1)
        extrap_data = []

        for year in extrap_years:
            years_ahead = year - end_year
            # Apply compound growth
            extrap_val = gdp_pc_end * np.power(1 + growth_rate, years_ahead)
            extrap_val = extrap_val.assign_coords(year=year)
            extrap_data.append(extrap_val)

        if extrap_data:
            gdp_pc_extrap = xr.concat(extrap_data, dim="year")
            # Combine historical and extrapolated
            gdp_pc_full = xr.concat([gdp_pc_hist, gdp_pc_extrap], dim="year")
        else:
            gdp_pc_full = gdp_pc_hist

    elif method == "linear":
        # Simple linear extrapolation (extend last value)
        gdp_pc_full = gdp_pc_hist.reindex(
            year=range(int(gdp_pc_hist.year.min()), target_year + 1),
            method="ffill"
        )

    else:
        raise ValueError(f"Unknown extrapolation method: {method}")

    return gdp_pc_full
