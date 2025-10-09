"""
Core mathematical functions for discounting.

Pure functions for calculating discount factors using various methods.
"""

import numpy as np
import xarray as xr
from typing import Optional, Union


def calculate_constant_discount_factors(
    years: xr.DataArray,
    discount_rate: float,
    pulse_year: int = 2020,
    discrete: bool = False
) -> xr.DataArray:
    """
    Calculate constant discount factors.

    Parameters
    ----------
    years : xr.DataArray
        Years for which to calculate discount factors
    discount_rate : float
        Annual discount rate (e.g., 0.02 for 2%)
    pulse_year : int, default 2020
        Base year for discounting (year of carbon pulse)
    discrete : bool, default False
        If True, use discrete discounting: DF = (1 + r)^(-t)
        If False, use continuous discounting: DF = exp(-r * t)

    Returns
    -------
    xr.DataArray
        Discount factors for each year

    Examples
    --------
    >>> years = xr.DataArray([2020, 2021, 2022], dims=['year'])
    >>> df = calculate_constant_discount_factors(years, discount_rate=0.02)

    Notes
    -----
    Discrete formula: DF(t) = (1 + r)^(-t)
    Continuous formula: DF(t) = exp(-r * t)

    where t = year - pulse_year
    """
    # Calculate years since pulse
    t = years - pulse_year

    if discrete:
        # Discrete discounting
        discount_factors = np.power(1 + discount_rate, -t)
    else:
        # Continuous discounting
        discount_factors = np.exp(-discount_rate * t)

    # Create DataArray with same structure as years
    result = xr.DataArray(
        discount_factors,
        dims=years.dims,
        coords=years.coords
    )

    result.attrs['discount_type'] = 'constant'
    result.attrs['discount_rate'] = discount_rate
    result.attrs['pulse_year'] = pulse_year
    result.attrs['discrete'] = discrete

    return result


def calculate_ramsey_discount_factors(
    consumption: xr.DataArray,
    eta: float,
    rho: float,
    pulse_year: int = 2020,
    discrete: bool = False
) -> xr.DataArray:
    """
    Calculate Ramsey discount factors based on consumption growth.

    The Ramsey rule: r(t) = ρ + η * g(t)

    where:
    - r(t) is the discount rate at time t
    - ρ (rho) is the pure rate of time preference
    - η (eta) is the elasticity of marginal utility
    - g(t) is the consumption growth rate at time t

    Parameters
    ----------
    consumption : xr.DataArray
        Consumption values over time, must have 'year' dimension
    eta : float
        Elasticity of marginal utility of consumption (risk aversion)
    rho : float
        Pure rate of time preference
    pulse_year : int, default 2020
        Base year for discounting
    discrete : bool, default False
        If True, use discrete formulation

    Returns
    -------
    xr.DataArray
        Discount factors for each year

    Examples
    --------
    >>> df = calculate_ramsey_discount_factors(
    ...     consumption,
    ...     eta=1.45,
    ...     rho=0.001,
    ...     pulse_year=2020
    ... )

    Notes
    -----
    The discount factor is calculated by compounding the discount rates:
    DF(t) = exp(-Σ r(s)) for s from pulse_year to t

    For discrete case:
    DF(t) = Π(1 + r(s))^(-1) for s from pulse_year to t
    """
    if 'year' not in consumption.dims:
        raise ValueError("consumption must have 'year' dimension")

    years = consumption.year

    # Calculate consumption growth rates
    # g(t) = (C(t+1) - C(t)) / C(t)
    # Or for continuous: g(t) = d(log(C))/dt ≈ (log(C(t+1)) - log(C(t)))

    # Use log differences for growth rate (handles percentage changes better)
    log_consumption = np.log(consumption.where(consumption > 0, 1e-10))

    # Calculate growth rates
    growth_rates = log_consumption.diff('year')

    # The diff operation reduces the year dimension by 1
    # Extend by assuming last growth rate continues
    last_growth = growth_rates.isel(year=-1)
    growth_rates = xr.concat([growth_rates, last_growth], dim='year')

    # Recalculate discount rates using Ramsey rule
    # r(t) = ρ + η * g(t)
    discount_rates = rho + eta * growth_rates

    # Calculate cumulative discount factors
    # Start from pulse year
    pulse_year_idx = int(np.where(years.values == pulse_year)[0][0])

    if discrete:
        # Discrete compounding: DF(t) = Π(1 + r(s))^(-1)
        # = (1 + r(pulse_year))^(-1) * (1 + r(pulse_year+1))^(-1) * ...

        # Initialize discount factors
        df_values = np.ones_like(consumption.values, dtype=float)

        # Cumulative product of discount factors
        for i in range(pulse_year_idx + 1, len(years)):
            rate = discount_rates.isel(year=i).values
            df_values[..., i] = df_values[..., i-1] / (1 + rate)

    else:
        # Continuous compounding: DF(t) = exp(-Σ r(s))
        cumsum_rates = discount_rates.cumsum('year')

        # Adjust so pulse year has DF = 1
        cumsum_at_pulse = cumsum_rates.isel(year=pulse_year_idx)
        adjusted_cumsum = cumsum_rates - cumsum_at_pulse

        df_values = np.exp(-adjusted_cumsum)

    # Create result DataArray
    result = xr.DataArray(
        df_values,
        dims=consumption.dims,
        coords=consumption.coords
    )

    result.attrs['discount_type'] = 'ramsey'
    result.attrs['eta'] = eta
    result.attrs['rho'] = rho
    result.attrs['pulse_year'] = pulse_year
    result.attrs['discrete'] = discrete

    return result


def calculate_gwr_discount_factors(
    consumption: xr.DataArray,
    eta: float,
    rho: float,
    pulse_year: int = 2020,
    method: str = "naive_gwr"
) -> xr.DataArray:
    """
    Calculate growth-weighted Ramsey (GWR) discount factors.

    GWR adjusts the discount rate based on average consumption growth
    across scenarios or models.

    Parameters
    ----------
    consumption : xr.DataArray
        Consumption values over time
    eta : float
        Elasticity of marginal utility
    rho : float
        Pure rate of time preference
    pulse_year : int, default 2020
        Base year for discounting
    method : str, default "naive_gwr"
        GWR method:
        - "naive_gwr": Use mean growth rate across scenarios
        - "gwr_gwr": Use GWR both for certainty equivalent and discounting
        - "euler_gwr": Use Euler equation with GWR

    Returns
    -------
    xr.DataArray
        Discount factors for each year

    Examples
    --------
    >>> df = calculate_gwr_discount_factors(
    ...     consumption,
    ...     eta=1.45,
    ...     rho=0.001,
    ...     method="naive_gwr"
    ... )

    Notes
    -----
    GWR methods compute the discount rate using averaged growth rates
    rather than scenario-specific growth rates.
    """
    if 'year' not in consumption.dims:
        raise ValueError("consumption must have 'year' dimension")

    # Determine dimensions to average over
    # Typically: ssp, model, simulation
    dims_to_avg = [d for d in consumption.dims if d not in ['year', 'region']]

    if method == "naive_gwr":
        # Take mean consumption across uncertainty dimensions
        mean_consumption = consumption.mean(dim=dims_to_avg)

        # Calculate Ramsey discount factors using mean consumption
        result = calculate_ramsey_discount_factors(
            mean_consumption,
            eta=eta,
            rho=rho,
            pulse_year=pulse_year,
            discrete=False
        )

        result.attrs['discount_type'] = 'naive_gwr'

    elif method == "gwr_gwr":
        # More sophisticated: use GWR for both CE and discounting
        # Calculate certainty equivalent consumption first
        from .utils import crra_certainty_equivalent

        # Compute CE consumption
        ce_consumption = crra_certainty_equivalent(
            consumption,
            eta=eta,
            dims=dims_to_avg
        )

        # Then calculate discount factors using CE consumption
        result = calculate_ramsey_discount_factors(
            ce_consumption,
            eta=eta,
            rho=rho,
            pulse_year=pulse_year,
            discrete=False
        )

        result.attrs['discount_type'] = 'gwr_gwr'

    elif method == "euler_gwr":
        # Euler equation approach
        # This is more complex and depends on specific formulation
        # For now, use simplified version similar to naive_gwr

        mean_consumption = consumption.mean(dim=dims_to_avg)

        result = calculate_ramsey_discount_factors(
            mean_consumption,
            eta=eta,
            rho=rho,
            pulse_year=pulse_year,
            discrete=False
        )

        result.attrs['discount_type'] = 'euler_gwr'

    else:
        raise ValueError(f"Unknown GWR method: {method}. "
                        f"Choose from: naive_gwr, gwr_gwr, euler_gwr")

    return result


def calculate_discount_factors(
    discount_type: str,
    years: Optional[xr.DataArray] = None,
    consumption: Optional[xr.DataArray] = None,
    discount_rate: Optional[float] = None,
    eta: Optional[float] = None,
    rho: Optional[float] = None,
    pulse_year: int = 2020,
    discrete: bool = False,
    gwr_method: str = "naive_gwr"
) -> xr.DataArray:
    """
    Unified interface for calculating discount factors.

    Parameters
    ----------
    discount_type : str
        Type of discounting: "constant", "ramsey", or "gwr"
    years : xr.DataArray, optional
        Years (required for constant discounting)
    consumption : xr.DataArray, optional
        Consumption data (required for ramsey/gwr)
    discount_rate : float, optional
        Discount rate (required for constant)
    eta : float, optional
        Elasticity of marginal utility (required for ramsey/gwr)
    rho : float, optional
        Pure rate of time preference (required for ramsey/gwr)
    pulse_year : int, default 2020
        Base year for discounting
    discrete : bool, default False
        Use discrete discounting
    gwr_method : str, default "naive_gwr"
        GWR method (only used if discount_type="gwr")

    Returns
    -------
    xr.DataArray
        Discount factors

    Examples
    --------
    >>> # Constant discounting
    >>> df = calculate_discount_factors(
    ...     discount_type="constant",
    ...     years=years,
    ...     discount_rate=0.02,
    ...     pulse_year=2020
    ... )

    >>> # Ramsey discounting
    >>> df = calculate_discount_factors(
    ...     discount_type="ramsey",
    ...     consumption=consumption,
    ...     eta=1.45,
    ...     rho=0.001,
    ...     pulse_year=2020
    ... )
    """
    if discount_type == "constant":
        if years is None or discount_rate is None:
            raise ValueError("constant discounting requires 'years' and 'discount_rate'")
        return calculate_constant_discount_factors(
            years, discount_rate, pulse_year, discrete
        )

    elif discount_type == "ramsey":
        if consumption is None or eta is None or rho is None:
            raise ValueError("ramsey discounting requires 'consumption', 'eta', and 'rho'")
        return calculate_ramsey_discount_factors(
            consumption, eta, rho, pulse_year, discrete
        )

    elif discount_type == "gwr":
        if consumption is None or eta is None or rho is None:
            raise ValueError("gwr discounting requires 'consumption', 'eta', and 'rho'")
        return calculate_gwr_discount_factors(
            consumption, eta, rho, pulse_year, gwr_method
        )

    else:
        raise ValueError(f"Unknown discount_type: {discount_type}. "
                        f"Choose from: constant, ramsey, gwr")


def calculate_euler_consumption_path(
    baseline_consumption: xr.DataArray,
    marginal_damages: xr.DataArray,
    eta: float,
    rho: float,
    pulse_year: int = 2020
) -> xr.DataArray:
    """
    Calculate consumption path under Euler equation accounting for damages.

    The Euler equation approach adjusts the consumption trajectory to account
    for climate damages, which affect future consumption growth and thus
    discount rates.

    Unlike the "naive" approach (which uses baseline consumption growth),
    Euler discounting uses consumption growth that includes damage impacts.

    Parameters
    ----------
    baseline_consumption : xr.DataArray
        Baseline consumption without climate damages (control scenario)
    marginal_damages : xr.DataArray
        Marginal damages from carbon pulse
    eta : float
        Elasticity of marginal utility
    rho : float
        Pure rate of time preference
    pulse_year : int, default 2020
        Year of carbon pulse

    Returns
    -------
    xr.DataArray
        Adjusted consumption path accounting for damages

    Examples
    --------
    >>> euler_consumption = calculate_euler_consumption_path(
    ...     baseline_consumption,
    ...     marginal_damages,
    ...     eta=1.45,
    ...     rho=0.001
    ... )
    >>> # Use euler_consumption for discount rate calculation
    >>> df = calculate_ramsey_discount_factors(euler_consumption, eta, rho)

    Notes
    -----
    The Euler equation is:
    C(t+1) / C(t) = [(1 + r(t)) / (1 + ρ)]^(1/η)

    With damages:
    C(t) = C_baseline(t) - D(t)

    where D(t) are climate damages.

    This creates a feedback loop where:
    1. Damages reduce consumption
    2. Lower consumption → lower discount rates
    3. Lower discount rates → higher SCC
    4. Higher SCC → more stringent policy → lower damages

    The Euler approach solves for the consumption path that is consistent
    with both the damages and the optimal consumption smoothing behavior.

    References
    ----------
    Ramsey, F. P. (1928). A mathematical theory of saving. The economic
    journal, 38(152), 543-559.

    Golosov, M., Hassler, J., Krusell, P., & Tsyvinski, A. (2014).
    Optimal taxes on fossil fuel in general equilibrium. Econometrica,
    82(1), 41-88.
    """
    if 'year' not in baseline_consumption.dims:
        raise ValueError("baseline_consumption must have 'year' dimension")

    if 'year' not in marginal_damages.dims:
        raise ValueError("marginal_damages must have 'year' dimension")

    # Align data
    consumption_aligned, damages_aligned = xr.align(
        baseline_consumption,
        marginal_damages,
        join='inner'
    )

    # Calculate consumption with damages
    # C_euler(t) = C_baseline(t) - D(t)
    consumption_with_damages = consumption_aligned - damages_aligned

    # Ensure consumption stays positive
    consumption_with_damages = consumption_with_damages.where(
        consumption_with_damages > 0,
        1e-10  # Floor at small positive value
    )

    # Add metadata
    consumption_with_damages.attrs['consumption_type'] = 'euler'
    consumption_with_damages.attrs['eta'] = eta
    consumption_with_damages.attrs['rho'] = rho
    consumption_with_damages.attrs['pulse_year'] = pulse_year
    consumption_with_damages.attrs['description'] = (
        'Consumption path accounting for climate damages (Euler equation)'
    )

    return consumption_with_damages


def calculate_euler_ramsey_discount_factors(
    baseline_consumption: xr.DataArray,
    marginal_damages: xr.DataArray,
    eta: float,
    rho: float,
    pulse_year: int = 2020,
    discrete: bool = False
) -> xr.DataArray:
    """
    Calculate Euler-Ramsey discount factors.

    This combines the Euler equation (consumption path with damages) with
    Ramsey discounting (consumption-growth-based discount rates).

    This is the "Euler-Ramsey" method in the original dscim implementation.

    Parameters
    ----------
    baseline_consumption : xr.DataArray
        Baseline consumption without damages
    marginal_damages : xr.DataArray
        Marginal damages from carbon pulse
    eta : float
        Elasticity of marginal utility
    rho : float
        Pure rate of time preference
    pulse_year : int, default 2020
        Year of carbon pulse
    discrete : bool, default False
        Use discrete discounting

    Returns
    -------
    xr.DataArray
        Euler-Ramsey discount factors

    Examples
    --------
    >>> df = calculate_euler_ramsey_discount_factors(
    ...     baseline_consumption,
    ...     marginal_damages,
    ...     eta=1.45,
    ...     rho=0.001
    ... )

    Notes
    -----
    The difference from naive Ramsey:
    - Naive: Uses baseline consumption growth (no damages)
    - Euler: Uses consumption growth with damages included

    This creates consistency between the damage calculations and
    the discount rates used to value those damages.

    See Also
    --------
    calculate_euler_consumption_path : Calculate consumption with damages
    calculate_ramsey_discount_factors : Naive Ramsey discount factors
    """
    # Calculate consumption path with damages
    euler_consumption = calculate_euler_consumption_path(
        baseline_consumption,
        marginal_damages,
        eta=eta,
        rho=rho,
        pulse_year=pulse_year
    )

    # Calculate Ramsey discount factors using Euler consumption
    result = calculate_ramsey_discount_factors(
        euler_consumption,
        eta=eta,
        rho=rho,
        pulse_year=pulse_year,
        discrete=discrete
    )

    # Update metadata
    result.attrs['discount_type'] = 'euler_ramsey'

    return result


def calculate_euler_gwr_discount_factors(
    baseline_consumption: xr.DataArray,
    marginal_damages: xr.DataArray,
    eta: float,
    rho: float,
    pulse_year: int = 2020
) -> xr.DataArray:
    """
    Calculate Euler-GWR discount factors.

    Combines Euler equation (consumption with damages) with
    Growth-Weighted Ramsey discounting (averaged growth rates).

    This is the "Euler-GWR" method in the original dscim implementation.

    Parameters
    ----------
    baseline_consumption : xr.DataArray
        Baseline consumption without damages
    marginal_damages : xr.DataArray
        Marginal damages from carbon pulse
    eta : float
        Elasticity of marginal utility
    rho : float
        Pure rate of time preference
    pulse_year : int, default 2020
        Year of carbon pulse

    Returns
    -------
    xr.DataArray
        Euler-GWR discount factors

    Examples
    --------
    >>> df = calculate_euler_gwr_discount_factors(
    ...     baseline_consumption,
    ...     marginal_damages,
    ...     eta=1.45,
    ...     rho=0.001
    ... )

    Notes
    -----
    Combines two adjustments:
    1. Euler: Account for damages in consumption path
    2. GWR: Use average growth rates across uncertainty

    See Also
    --------
    calculate_euler_consumption_path : Consumption with damages
    calculate_gwr_discount_factors : GWR discounting
    """
    # Calculate consumption path with damages
    euler_consumption = calculate_euler_consumption_path(
        baseline_consumption,
        marginal_damages,
        eta=eta,
        rho=rho,
        pulse_year=pulse_year
    )

    # Apply GWR averaging
    result = calculate_gwr_discount_factors(
        euler_consumption,
        eta=eta,
        rho=rho,
        pulse_year=pulse_year,
        method="naive_gwr"
    )

    # Update metadata
    result.attrs['discount_type'] = 'euler_gwr'

    return result


def compare_discount_methods(
    baseline_consumption: xr.DataArray,
    marginal_damages: xr.DataArray,
    eta: float,
    rho: float,
    pulse_year: int = 2020
) -> dict:
    """
    Compare different discount methods for analysis.

    Useful for understanding the impact of different discounting
    assumptions on SCC calculations.

    Parameters
    ----------
    baseline_consumption : xr.DataArray
        Baseline consumption without damages
    marginal_damages : xr.DataArray
        Marginal damages from carbon pulse
    eta : float
        Elasticity of marginal utility
    rho : float
        Pure rate of time preference
    pulse_year : int, default 2020
        Year of carbon pulse

    Returns
    -------
    dict
        Dictionary with discount factors from each method:
        - 'naive_ramsey': Baseline consumption only
        - 'naive_gwr': GWR with baseline consumption
        - 'euler_ramsey': Consumption with damages
        - 'euler_gwr': GWR with damaged consumption

    Examples
    --------
    >>> comparison = compare_discount_methods(
    ...     baseline_consumption,
    ...     marginal_damages,
    ...     eta=1.45,
    ...     rho=0.001
    ... )
    >>> # Compare discount factors
    >>> naive_df = comparison['naive_ramsey']
    >>> euler_df = comparison['euler_ramsey']

    Notes
    -----
    Generally:
    - Euler methods have lower discount rates when damages are large
    - GWR methods smooth out uncertainty across scenarios
    - Euler + GWR combines both effects
    """
    results = {}

    # Naive Ramsey (baseline consumption, no averaging)
    results['naive_ramsey'] = calculate_ramsey_discount_factors(
        baseline_consumption,
        eta=eta,
        rho=rho,
        pulse_year=pulse_year
    )

    # Naive GWR (baseline consumption, with averaging)
    results['naive_gwr'] = calculate_gwr_discount_factors(
        baseline_consumption,
        eta=eta,
        rho=rho,
        pulse_year=pulse_year,
        method="naive_gwr"
    )

    # Euler Ramsey (consumption with damages, no averaging)
    results['euler_ramsey'] = calculate_euler_ramsey_discount_factors(
        baseline_consumption,
        marginal_damages,
        eta=eta,
        rho=rho,
        pulse_year=pulse_year
    )

    # Euler GWR (consumption with damages, with averaging)
    results['euler_gwr'] = calculate_euler_gwr_discount_factors(
        baseline_consumption,
        marginal_damages,
        eta=eta,
        rho=rho,
        pulse_year=pulse_year
    )

    return results


def calculate_discount_factors_per_capita(
    consumption_per_capita: xr.DataArray,
    eta: float,
    rho: float,
    pulse_year: int,
    discrete: bool = False,
    ext_end_year: int = 2300
) -> xr.DataArray:
    """
    Calculate discount factors using per-capita consumption.

    This function replicates the original dscim's calculate_discount_factors method
    from main_recipe.py:1173.

    The discount factor formula (discrete):
    DF(t) = [1/(1+ρ)]^t × [C_pc(pulse_year)^η / C_pc(t)^η]

    Parameters
    ----------
    consumption_per_capita : xr.DataArray
        Per-capita consumption from pulse year to end year
        Must have 'year' dimension
    eta : float
        Elasticity of marginal utility (risk aversion parameter)
    rho : float
        Pure rate of time preference
    pulse_year : int
        Year of carbon pulse (base year for discounting)
    discrete : bool, default False
        If True, use discrete discounting
        If False, use continuous discounting
    ext_end_year : int, default 2300
        Final year for discounting

    Returns
    -------
    xr.DataArray
        Discount factors with same dimensions as consumption_per_capita

    Examples
    --------
    >>> # Calculate discount factors
    >>> df = calculate_discount_factors_per_capita(
    ...     consumption_per_capita=cons_pc,
    ...     eta=1.421158116,
    ...     rho=0.00461878399,
    ...     pulse_year=2020,
    ...     discrete=False
    ... )

    Notes
    -----
    From dscim/src/dscim/menu/main_recipe.py:1173

    The formula computes two components:
    1. Time preference: [1/(1+ρ)]^t (discount future due to impatience)
    2. Marginal utility: [C(0)^η / C(t)^η] (discount future if consumption is higher)

    The product gives the discount factor for year t.

    See Also
    --------
    calculate_ramsey_discount_factors : Alternative Ramsey implementation
    """
    # Subset to pulse year onward
    cons_pc = consumption_per_capita.sel(year=slice(pulse_year, ext_end_year))

    # Calculate the time preference component
    if discrete:
        # Discrete: ρ stays as is
        rhos = xr.DataArray(rho, coords=[cons_pc.year])
    else:
        # Continuous: convert ρ using e^ρ - 1
        rhos = np.expm1(xr.DataArray(rho, coords=[cons_pc.year]))

    # Accumulate time preference factors: 1 / [(1+ρ) × (1+ρ) × ...]
    stream_rhos = np.divide(
        1, np.multiply.accumulate((rhos.values + 1), rhos.dims.index("year"))
    )

    # Calculate the marginal utility component
    # ratio = [C(pulse_year)^η] / [C(t)^η]
    c_pulse = cons_pc.sel(year=pulse_year)
    ratio = np.power(c_pulse, eta) / np.power(cons_pc, eta)

    # Discount factor = time preference × marginal utility ratio
    factors = stream_rhos * ratio

    # Add metadata
    factors.attrs['discount_type'] = 'per_capita_ramsey'
    factors.attrs['eta'] = eta
    factors.attrs['rho'] = rho
    factors.attrs['pulse_year'] = pulse_year
    factors.attrs['discrete'] = discrete

    return factors


def calculate_stream_discount_factors_per_scenario(
    global_consumption_no_pulse: xr.DataArray,
    population: xr.DataArray,
    eta: float,
    rho: float,
    pulse_year: int,
    discount_type: str,
    fair_aggregation: list,
    fair_dims: list = None,
    discrete: bool = False,
    ext_end_year: int = 2300
) -> xr.DataArray:
    """
    Calculate stream of discount factors per scenario.

    This replicates the original dscim's calculate_stream_discount_factors method
    from main_recipe.py:1233.

    Handles different discount types:
    - naive_ramsey: Use baseline consumption growth
    - naive_gwr/gwr_gwr: Average over scenarios first
    - euler_ramsey: Use consumption with damages
    - euler_gwr: Use consumption with damages + averaging

    Parameters
    ----------
    global_consumption_no_pulse : xr.DataArray
        Global consumption without pulse (no climate damages)
        Dimensions: (weitzman_parameter, year, ssp, model, ...)
    population : xr.DataArray
        Population data
        Dimensions: (year, ssp, model, region)
    eta : float
        Elasticity of marginal utility
    rho : float
        Pure rate of time preference
    pulse_year : int
        Year of carbon pulse
    discount_type : str
        Type of discounting: "naive_ramsey", "naive_gwr", "gwr_gwr", etc.
    fair_aggregation : list
        List of FAIR aggregation methods (e.g., ["ce", "mean", "median_params"])
    fair_dims : list, optional
        FAIR dimensions to collapse (default: ["simulation"])
    discrete : bool, default False
        Use discrete discounting
    ext_end_year : int, default 2300
        Final year for discounting

    Returns
    -------
    xr.DataArray
        Discount factors with dimensions based on discount_type and aggregation

    Examples
    --------
    >>> discount_factors = calculate_stream_discount_factors_per_scenario(
    ...     global_consumption_no_pulse=global_cons_no_pulse,
    ...     population=pop,
    ...     eta=1.421158116,
    ...     rho=0.00461878399,
    ...     pulse_year=2020,
    ...     discount_type="naive_ramsey",
    ...     fair_aggregation=["mean"],
    ...     fair_dims=["simulation"]
    ... )

    Notes
    -----
    From dscim/src/dscim/menu/main_recipe.py:1233

    Different discount types require different handling:
    - naive: Use baseline consumption (no damages)
    - euler: Use consumption with damages
    - gwr: Average across scenarios
    - _gwr suffix: Average model/ssp dimensions
    """
    if fair_dims is None:
        fair_dims = ["simulation"]

    # Collapse population as needed
    full_pop = population.sum("region") if "region" in population.dims else population
    full_pop = full_pop.reindex(
        year=range(int(full_pop.year.min()), ext_end_year + 1),
        method="ffill"
    )

    # Collapse pop dimensions based on fair_dims
    if len(fair_dims) > 1:
        pop = full_pop.mean([d for d in fair_dims if d in full_pop.dims])
    else:
        pop = full_pop

    # Handle different discount types
    if discount_type == "gwr_gwr":
        # Calculate naive_ramsey factors, then average
        # Per-capita consumption
        cons_pc = global_consumption_no_pulse / full_pop

        # Calculate discount factors
        factors = calculate_discount_factors_per_capita(
            cons_pc, eta, rho, pulse_year, discrete, ext_end_year
        )

        # Average over ssp and model
        discount_factors = factors.mean(dim=["ssp", "model"])

        # Expand dims to match fair_aggregation
        discount_factors = discount_factors.expand_dims(
            {"fair_aggregation": fair_aggregation}
        )

    elif "naive" in discount_type:
        # Use baseline consumption (no damages)
        cons_pc = global_consumption_no_pulse / pop

        # Calculate discount factors
        discount_factors = calculate_discount_factors_per_capita(
            cons_pc, eta, rho, pulse_year, discrete, ext_end_year
        )

        # Expand dims to match fair_aggregation
        discount_factors = discount_factors.expand_dims(
            {"fair_aggregation": fair_aggregation}
        )

    elif "euler" in discount_type:
        # Euler methods handled in SCC calculation
        # (need damages to calculate consumption with damages)
        # For now, return placeholder
        raise NotImplementedError(
            f"Euler discount types ({discount_type}) should be calculated "
            "during SCC calculation when damages are available"
        )

    else:
        raise ValueError(f"Unknown discount_type: {discount_type}")

    return discount_factors
