"""
Core mathematical functions for damage function generation.

Pure functions with no I/O operations.
"""

import numpy as np
import xarray as xr
import pandas as pd
from typing import Tuple, Optional, Dict, Any
from scipy import stats
import statsmodels.formula.api as smf


def fit_damage_function_ols(
    damages: xr.DataArray,
    climate_var: xr.DataArray,
    formula: str,
    coords_to_stack: Optional[list] = None
) -> xr.Dataset:
    """
    Fit damage function using Ordinary Least Squares regression.

    Parameters
    ----------
    damages : xr.DataArray
        Damage values (consumption differences or similar)
    climate_var : xr.DataArray
        Climate variable (temperature anomaly or GMSL)
    formula : str
        Patsy formula string (e.g., "damages ~ -1 + anomaly + np.power(anomaly, 2)")
    coords_to_stack : list, optional
        Coordinates to stack for regression. If None, uses all shared dims.

    Returns
    -------
    xr.Dataset
        Contains:
        - coefficients: Fitted regression coefficients
        - residuals: Regression residuals
        - rsquared: R-squared values
        - fitted_values: Fitted damage values

    Examples
    --------
    >>> formula = "damages ~ -1 + anomaly + np.power(anomaly, 2)"
    >>> result = fit_damage_function_ols(damages, temperature, formula)
    >>> coefficients = result["coefficients"]

    Notes
    -----
    The function handles multi-dimensional data by stacking specified
    coordinates and running regression on the stacked data.
    """
    # Determine which coordinates to stack
    if coords_to_stack is None:
        # Stack all shared dimensions between damages and climate_var
        shared_dims = set(damages.dims) & set(climate_var.dims)
        coords_to_stack = list(shared_dims)

    if len(coords_to_stack) == 0:
        raise ValueError("No shared dimensions to stack for regression")

    # Broadcast damages and climate to same shape before stacking
    # This ensures they align properly along all shared dimensions
    damages_aligned, climate_aligned = xr.broadcast(damages, climate_var)

    # Stack the aligned data
    damages_stacked = damages_aligned.stack(obs=coords_to_stack)
    climate_stacked = climate_aligned.stack(obs=coords_to_stack)

    # Convert to numpy arrays and flatten
    # Handle both DataArray and Dataset cases
    if isinstance(damages_stacked, xr.Dataset):
        # If Dataset, get the first (and should be only) data variable
        damages_values = damages_stacked[list(damages_stacked.data_vars)[0]].values.ravel()
    else:
        damages_values = np.asarray(damages_stacked).ravel()

    if isinstance(climate_stacked, xr.Dataset):
        climate_values = climate_stacked[list(climate_stacked.data_vars)[0]].values.ravel()
    else:
        climate_values = np.asarray(climate_stacked).ravel()

    # Remove NaN values
    valid_mask = ~(np.isnan(damages_values) | np.isnan(climate_values))
    damages_valid = damages_values[valid_mask]
    climate_valid = climate_values[valid_mask]

    # Create DataFrame for statsmodels
    # Extract variable name from formula (e.g., "anomaly" from formula)
    climate_var_name = _extract_climate_var_name(formula)

    df = pd.DataFrame({
        'damages': damages_valid,
        climate_var_name: climate_valid
    })

    # Fit OLS using statsmodels (same as original DSCIM)
    mod = smf.ols(formula=formula, data=df).fit()

    # Extract coefficients
    coefficients = mod.params.values
    coef_names = mod.params.index.tolist()

    # Get fit statistics
    r_squared = mod.rsquared
    n_obs = int(mod.nobs)

    # Create output dataset
    result = xr.Dataset({
        'coefficients': xr.DataArray(
            coefficients,
            dims=['coefficient'],
            coords={'coefficient': coef_names}
        ),
        'rsquared': r_squared,
        'n_obs': n_obs
    })

    # Add metadata
    result.attrs['formula'] = formula
    result.attrs['fit_type'] = 'ols'

    return result


def _extract_climate_var_name(formula: str) -> str:
    """
    Extract climate variable name from patsy formula.

    Parameters
    ----------
    formula : str
        Patsy formula string

    Returns
    -------
    str
        Climate variable name

    Examples
    --------
    >>> _extract_climate_var_name("damages ~ -1 + anomaly + np.power(anomaly, 2)")
    'anomaly'
    >>> _extract_climate_var_name("damages ~ -1 + gmsl + np.power(gmsl, 2)")
    'gmsl'
    """
    # Common climate variable names
    common_vars = ['anomaly', 'gmsl', 'temperature', 'temp']

    for var in common_vars:
        if var in formula:
            return var

    # If not found, try to parse from formula
    # Look for pattern like "~ ... + var +"
    import re
    # Match word characters that appear after ~ or + but before another operator
    matches = re.findall(r'[\+\~]\s*([a-zA-Z_][a-zA-Z0-9_]*)', formula)
    if matches:
        # Return first match that isn't 'damages'
        for match in matches:
            if match != 'damages' and not match.startswith('np.'):
                return match

    raise ValueError(f"Could not extract climate variable name from formula: {formula}")


def evaluate_damage_function(
    coefficients: xr.DataArray,
    climate_values: xr.DataArray,
    formula: str
) -> xr.DataArray:
    """
    Evaluate fitted damage function at specified climate values.

    Parameters
    ----------
    coefficients : xr.DataArray
        Fitted regression coefficients
    climate_values : xr.DataArray
        Climate variable values at which to evaluate
    formula : str
        Original formula used for fitting

    Returns
    -------
    xr.DataArray
        Predicted damage values

    Examples
    --------
    >>> damages_pred = evaluate_damage_function(coefficients, new_temps, formula)
    """
    climate_var_name = _extract_climate_var_name(formula)

    # Create DataFrame for evaluation
    original_shape = climate_values.shape
    df = pd.DataFrame({
        climate_var_name: np.asarray(climate_values).ravel()
    })

    # Manually compute predictions using the formula structure
    # For formula like "damages ~ -1 + anomaly + np.power(anomaly, 2)"
    # We evaluate each term
    coef_dict = {name: float(coefficients.sel(coefficient=name).values)
                 for name in coefficients.coefficient.values}

    predictions = np.zeros(len(df))

    for coef_name, coef_value in coef_dict.items():
        if coef_name == climate_var_name:
            # Linear term
            predictions += coef_value * df[climate_var_name].values
        elif 'power' in coef_name.lower() or '**' in coef_name:
            # Extract power and compute
            # For np.power(anomaly, 2) we need the squared term
            if '2' in coef_name:
                predictions += coef_value * df[climate_var_name].values ** 2
            elif '3' in coef_name:
                predictions += coef_value * df[climate_var_name].values ** 3

    # Reshape to match input shape
    result = xr.DataArray(
        predictions.reshape(original_shape),
        dims=climate_values.dims,
        coords=climate_values.coords
    )

    return result


def calculate_marginal_damages(
    coefficients: xr.DataArray,
    climate_values: xr.DataArray,
    formula: str
) -> xr.DataArray:
    """
    Calculate marginal damages (partial derivative of damage function).

    For quadratic form: damages = B_0 + B_1*x + B_2 * xˆ2
    Marginal damage: ∂damages/∂x = B_1 + 2*B_2*x

    Parameters
    ----------
    coefficients : xr.DataArray
        Fitted regression coefficients
    climate_values : xr.DataArray
        Climate variable values
    formula : str
        Original damage function formula

    Returns
    -------
    xr.DataArray
        Marginal damage values (∂damages/∂climate)

    Examples
    --------
    >>> md = calculate_marginal_damages(coefficients, temperatures, formula)
    >>> # md represents dollars of damage per degree of warming

    Notes
    -----
    This function analytically computes the derivative based on the
    formula structure. For polynomial formulas, it uses the power rule.
    """
    climate_var_name = _extract_climate_var_name(formula)

    # Parse formula to identify terms
    # For now, handle common quadratic case: B_1*x + B_2 * xˆ2
    # Derivative: B_1 + 2*B_2*x

    # Get coefficients
    coef_names = list(coefficients.coefficient.values)

    # Initialize marginal damage
    marginal = xr.zeros_like(climate_values, dtype=float)

    for coef_name in coef_names:
        coef_value = coefficients.sel(coefficient=coef_name).values

        # Determine derivative contribution
        if coef_name == climate_var_name:
            # Linear term: d/dx(B*x) = B
            marginal = marginal + coef_value

        elif f'np.power({climate_var_name}, 2)' in coef_name:
            # Quadratic term: d/dx(B*x²) = 2*B*x
            marginal = marginal + 2 * coef_value * climate_values

        elif f'{climate_var_name}:' in coef_name or f':{climate_var_name}' in coef_name:
            # Interaction term - handle case by case
            # For now, simplified approach
            ### PLEASE IMPLEMENT INTERACTION TERM DERIVATIVES AS NEEDED ###
            pass

    return marginal


def extrapolate_damages(
    damages: xr.DataArray,
    start_year: int,
    end_year: int,
    method: str = "global_c_ratio",
    target_years: Optional[xr.DataArray] = None
) -> xr.DataArray:
    """
    Extrapolate damages beyond the projection period.

    Parameters
    ----------
    damages : xr.DataArray
        Damage values to extrapolate, must have 'year' dimension
    start_year : int
        First year to use for extrapolation calculation
    end_year : int
        Last year of available data
    method : str, default "global_c_ratio"
        Extrapolation method:
        - "global_c_ratio": Use ratio of global consumption
        - "constant": Hold last value constant
        - "linear": Linear extrapolation
    target_years : xr.DataArray, optional
        Years to extrapolate to. If None, extends to 2300

    Returns
    -------
    xr.DataArray
        Damages extended to target years

    Examples
    --------
    >>> damages_extrap = extrapolate_damages(
    ...     damages,
    ...     start_year=2085,
    ...     end_year=2099,
    ...     method="global_c_ratio"
    ... )

    Notes
    -----
    The global_c_ratio method assumes damages grow proportionally to
    global consumption beyond the projection period.
    """
    if 'year' not in damages.dims:
        raise ValueError("damages must have 'year' dimension for extrapolation")

    if target_years is None:
        # Default: extend to 2300
        max_year = damages.year.max().values
        target_years = xr.DataArray(
            np.arange(max_year + 1, 2301),
            dims=['year'],
            coords={'year': np.arange(max_year + 1, 2301)}
        )

    if method == "constant":
        # Hold last value constant
        last_value = damages.sel(year=end_year)
        extrap_values = last_value.expand_dims(year=target_years.values)
        return xr.concat([damages, extrap_values], dim='year')

    elif method == "linear":
        # Linear extrapolation using last two points
        # Get slope from last period
        subset = damages.sel(year=slice(start_year, end_year))
        years_subset = subset.year.values

        # Simple linear fit
        # delta*damage/delta*year
        first_val = subset.isel(year=0)
        last_val = subset.isel(year=-1)
        slope = (last_val - first_val) / (years_subset[-1] - years_subset[0])

        # Extrapolate
        years_ahead = target_years.values - end_year
        extrap_values = last_val + slope * years_ahead

        return xr.concat([damages, extrap_values], dim='year')

    elif method == "global_c_ratio":
        #### IMPLEMENTATION NOTE ####
        # This method requires global consumption data
        # For now, implement simplified version using growth rate
        # In full implementation, this would use actual consumption projections

        # Calculate average growth rate from subset period
        subset = damages.sel(year=slice(start_year, end_year))

        # Compute compound annual growth rate
        first_val = subset.isel(year=0)
        last_val = subset.isel(year=-1)
        n_years = end_year - start_year

        # CAGR = (end_value / start_value)^(1/n_years) - 1
        # Avoid division by zero
        growth_rate = np.where(
            first_val != 0,
            np.power(last_val / first_val, 1 / n_years) - 1,
            0
        )

        # Extrapolate using growth rate
        years_ahead = target_years.values - end_year
        extrap_values = last_val * np.power(1 + growth_rate, years_ahead)

        return xr.concat([damages, extrap_values], dim='year')

    else:
        raise ValueError(f"Unknown extrapolation method: {method}")


def compute_damage_function_points(
    coefficients: xr.DataArray,
    climate_range: Tuple[float, float],
    formula: str,
    n_points: int = 100
) -> xr.Dataset:
    """
    Compute damage function evaluation points for visualization.

    Parameters
    ----------
    coefficients : xr.DataArray
        Fitted regression coefficients
    climate_range : tuple of float
        (min, max) range of climate variable
    formula : str
        Damage function formula
    n_points : int, default 100
        Number of points to evaluate

    Returns
    -------
    xr.Dataset
        Contains:
        - climate: Climate variable values
        - damages: Predicted damage values

    Examples
    --------
    >>> points = compute_damage_function_points(
    ...     coefficients,
    ...     climate_range=(0, 10),
    ...     formula="damages ~ -1 + anomaly + np.power(anomaly, 2)",
    ...     n_points=100
    ... )
    """
    climate_var_name = _extract_climate_var_name(formula)

    # Create evaluation points
    climate_points = np.linspace(climate_range[0], climate_range[1], n_points)

    climate_da = xr.DataArray(
        climate_points,
        dims=['point'],
        coords={'point': np.arange(n_points)}
    )

    # Evaluate damage function
    damage_points = evaluate_damage_function(coefficients, climate_da, formula)

    # Create result dataset
    result = xr.Dataset({
        climate_var_name: climate_da,
        'damages': damage_points
    })

    result.attrs['climate_range'] = climate_range
    result.attrs['formula'] = formula

    return result


def compute_damage_function_grid(
    coefficients: xr.DataArray,
    formula: str,
    min_anomaly: float = 0.0,
    max_anomaly: float = 20.0,
    step_anomaly: float = 0.2,
    min_gmsl: float = 0.0,
    max_gmsl: float = 300.0,
    step_gmsl: float = 3.0,
) -> xr.Dataset:
    """
    Compute damage function predictions on a climate grid.

    This matches the original dscim implementation where "damage_function_coefficients"
    are actually predicted damage values evaluated at a grid of climate points,
    not the regression coefficients themselves.

    Parameters
    ----------
    coefficients : xr.DataArray
        Fitted regression coefficients with dimension (coefficient)
    formula : str
        Damage function formula (e.g., "damages ~ -1 + anomaly + np.power(anomaly, 2)")
    min_anomaly : float, default 0.0
        Minimum temperature anomaly for grid
    max_anomaly : float, default 20.0
        Maximum temperature anomaly for grid
    step_anomaly : float, default 0.2
        Step size for temperature anomaly grid
    min_gmsl : float, default 0.0
        Minimum GMSL for grid
    max_gmsl : float, default 300.0
        Maximum GMSL for grid
    step_gmsl : float, default 3.0
        Step size for GMSL grid

    Returns
    -------
    xr.Dataset
        Dataset with formula terms as data variables, each containing
        predicted damage values on the climate grid.

        For formula "damages ~ -1 + anomaly + np.power(anomaly, 2)":
        Variables: ['anomaly', 'np.power(anomaly, 2)']

        This structure matches the original dscim output where these are
        stored as "damage_function_coefficients" (confusingly named).

    Notes
    -----
    Iriginal dscim saves these as "damage_function_coefficients" but they
    are actually predicted damage function values on a grid, not regression
    coefficients. This allows later interpolation to evaluate damages at
    arbitrary climate points.

    Examples
    --------
    >>> # Fit regression
    >>> result = fit_damage_function_ols(damages, climate, formula)
    >>> coeffs = result['coefficients']
    >>>
    >>> # Generate grid predictions (original dscim format)
    >>> grid = compute_damage_function_grid(coeffs, formula)
    >>> # grid has variables: ['anomaly', 'np.power(anomaly, 2)']
    """
    # Determine which climate variables are in the formula
    has_anomaly = 'anomaly' in formula
    has_gmsl = 'gmsl' in formula

    # Create climate grid
    if has_anomaly and has_gmsl:
        # 2D grid: temperature × GMSL
        temps = np.arange(min_anomaly, max_anomaly, step_anomaly)
        gmsls = np.arange(min_gmsl, max_gmsl, step_gmsl)

        # Create meshgrid
        temp_grid, gmsl_grid = np.meshgrid(temps, gmsls, indexing='ij')

        # Flatten for evaluation
        anomaly_flat = temp_grid.flatten()
        gmsl_flat = gmsl_grid.flatten()

        # Create DataFrame for patsy
        eval_df = pd.DataFrame({
            'anomaly': anomaly_flat,
            'gmsl': gmsl_flat
        })

    elif has_anomaly:
        # 1D grid: temperature only
        temps = np.arange(min_anomaly, max_anomaly, step_anomaly)
        eval_df = pd.DataFrame({'anomaly': temps})

    elif has_gmsl:
        # 1D grid: GMSL only
        gmsls = np.arange(min_gmsl, max_gmsl, step_gmsl)
        eval_df = pd.DataFrame({'gmsl': gmsls})

    else:
        raise ValueError(f"Formula must contain 'anomaly' or 'gmsl': {formula}")

    # Manually evaluate each term in the formula on the grid
    # This avoids complexity with patsy environment and directly computes what we need
    ### ADDITIONAL VERIFICATION NEEDED ###
    result_dict = {}

    # Get coefficient names (these match the formula terms)
    coef_names = coefficients.coefficient.values

    # Evaluate each term
    for i, coef_name in enumerate(coef_names):
        coef_value = float(coefficients.sel(coefficient=coef_name).values)

        # Evaluate the term on the grid based on coefficient name
        if coef_name == 'anomaly':
            term_values = eval_df['anomaly'].values * coef_value
        elif coef_name == 'gmsl':
            term_values = eval_df['gmsl'].values * coef_value
        elif 'np.power(anomaly, 2)' in coef_name:
            term_values = np.power(eval_df['anomaly'].values, 2) * coef_value
        elif 'anomaly ** 2' in coef_name:
            term_values = np.power(eval_df['anomaly'].values, 2) * coef_value
        elif 'np.power(gmsl, 2)' in coef_name:
            term_values = np.power(eval_df['gmsl'].values, 2) * coef_value
        else:
            # Generic evaluation using eval for any other numpy functions
            local_vars = {
                'anomaly': eval_df.get('anomaly', pd.Series(np.zeros(len(eval_df)))).values,
                'gmsl': eval_df.get('gmsl', pd.Series(np.zeros(len(eval_df)))).values,
                'np': np
            }
            try:
                term_values = eval(coef_name, {"__builtins__": {}}, local_vars) * coef_value
            except Exception as e:
                raise ValueError(f"Cannot evaluate coefficient term '{coef_name}': {e}")

        # Store as data variable using the coefficient name
        result_dict[coef_name] = xr.DataArray(
            term_values,
            dims=['grid_point'],
            coords={'grid_point': np.arange(len(term_values))}
        )

    result = xr.Dataset(result_dict)

    # Add metadata
    result.attrs['formula'] = formula
    result.attrs['note'] = (
        "These are predicted damage function values on a climate grid, "
        "not regression coefficients. This matches original dscim format."
    )
    result.attrs['grid_config'] = {
        'min_anomaly': min_anomaly,
        'max_anomaly': max_anomaly,
        'step_anomaly': step_anomaly,
        'min_gmsl': min_gmsl,
        'max_gmsl': max_gmsl,
        'step_gmsl': step_gmsl,
    }

    return result


def fit_damage_function_rolling_window(
    damages: xr.DataArray,
    climate_var: xr.DataArray,
    formula: str,
    year_range: range,
    window_size: int = 5,
    ssp_dim: str = 'ssp',
    model_dim: str = 'model',
    year_dim: str = 'year',
    region_dim: str = 'region',
) -> xr.Dataset:
    """
    Fit damage functions using rolling windows per year and per scenario.

    This matches the original dscim implementation where damage functions
    are fitted separately for each (ssp, model) combination using a rolling
    5-year window centered on each year.

    Parameters
    ----------
    damages : xr.DataArray
        Reduced damages with dimensions (year, ssp, region, model, ...)
    climate_var : xr.DataArray
        Climate variable (anomaly or gmsl) with matching dimensions
    formula : str
        Damage function formula (e.g., "damages ~ -1 + anomaly + np.power(anomaly, 2)")
    year_range : range
        Years to fit (e.g., range(2020, 2101))
    window_size : int, default=5
        Size of rolling window
    ssp_dim : str
        Name of SSP dimension
    model_dim : str
        Name of model dimension
    year_dim : str
        Name of year dimension
    region_dim : str
        Name of region dimension (will be aggregated)

    Returns
    -------
    xr.Dataset
        Dataset with dimensions (ssp, model, year) containing fitted coefficient
        values. Variables correspond to formula terms (e.g., 'anomaly',
        'np.power(anomaly, 2)').
    """
    import pandas as pd
    import statsmodels.formula.api as smf

    # STEP 1: Convert to DataFrame like original dscim
    # Aggregate damages over region dimension first
    if region_dim in damages.dims:
        damages_agg = damages.sum(dim=region_dim)
    else:
        damages_agg = damages

    # Convert to DataFrame
    df_damages = damages_agg.to_dataframe(name='damages').reset_index()

    # Convert climate to DataFrame
    df_climate = climate_var.to_dataframe(name='anomaly' if 'anomaly' not in climate_var.name else climate_var.name).reset_index()

    # Determine the climate variable name
    climate_col_name = 'gmsl' if 'gmsl' in formula else 'anomaly'
    if climate_col_name not in df_climate.columns and len(df_climate.columns) > len(df_climate.index.names) + 1:
        # The climate variable might have a different name - get the last column
        value_cols = [col for col in df_climate.columns if col not in [ssp_dim, model_dim, year_dim, 'region', 'hierid']]
        if value_cols:
            df_climate = df_climate.rename(columns={value_cols[0]: climate_col_name})

    # Merge damages and climate data
    merge_cols = [col for col in [year_dim, ssp_dim, model_dim] if col in df_damages.columns and col in df_climate.columns]
    df = pd.merge(df_damages, df_climate, on=merge_cols, how='inner')

    # STEP 2: Fit using rolling window like original dscim (utils.py lines 398-414)
    # Get unique SSPs and models
    ssps = df[ssp_dim].unique()
    models = df[model_dim].unique()
    years = list(year_range)

    # Initialize storage for results
    results_dict = {}

    # Fit for each (ssp, model, year) combination
    for i_ssp, ssp in enumerate(ssps):
        for i_model, model in enumerate(models):
            # Filter DataFrame for this ssp-model combination
            ssp_model_df = df[(df[ssp_dim] == ssp) & (df[model_dim] == model)]

            for i_year, target_year in enumerate(years):
                # Define rolling window centered on target year
                half_window = window_size // 2
                window_start = target_year - half_window
                window_end = target_year + half_window + 1

                # Filter for window years (like original: year-2 to year+2)
                window_df = ssp_model_df[ssp_model_df[year_dim].isin(range(window_start, window_end))]

                if len(window_df) < 2:
                    # Not enough data to fit
                    continue

                # Remove NaNs
                window_df_clean = window_df.dropna(subset=['damages', climate_col_name])

                if len(window_df_clean) < 2:
                    # Not enough valid observations
                    continue

                # Fit using statsmodels
                try:
                    model_fit = smf.ols(formula, data=window_df_clean).fit()

                    # Extract coefficients
                    for param_name, param_value in model_fit.params.items():
                        if param_name not in results_dict:
                            # Initialize array for this coefficient
                            results_dict[param_name] = np.full(
                                (len(ssps), len(models), len(years)),
                                np.nan
                            )

                        results_dict[param_name][i_ssp, i_model, i_year] = param_value

                except Exception as e:
                    # Fit failed for this window - leave as NaN
                    continue

    # Convert results to xarray Dataset
    coords = {
        ssp_dim: ssps,
        model_dim: models,
        year_dim: np.array(years),
    }

    data_vars = {}
    for term_name, values in results_dict.items():
        data_vars[term_name] = xr.DataArray(
            values,
            dims=[ssp_dim, model_dim, year_dim],
            coords=coords
        )

    result = xr.Dataset(data_vars)

    result.attrs['formula'] = formula
    result.attrs['window_size'] = window_size
    result.attrs['method'] = 'rolling_window_ols'

    return result


def compute_damages_from_climate(
    climate_data: xr.Dataset,
    damage_function_coefficients: xr.Dataset,
    formula: str
) -> xr.DataArray:
    """
    Compute damages using FAIR climate projections and damage function coefficients.

    This is analogous to the original dscim's `compute_damages()` function.
    It evaluates the damage function at specific climate values using the
    fitted coefficients.

    Parameters
    ----------
    climate_data : xr.Dataset
        Climate data (e.g., FAIR projections) containing 'temperature' and/or 'gmsl'
    damage_function_coefficients : xr.Dataset
        Damage function coefficients with dimensions (discount_type, ssp, model, year)
        Variables are formula terms (e.g., 'anomaly', 'np.power(anomaly, 2)')
    formula : str
        Damage function formula (e.g., "damages ~ -1 + anomaly + np.power(anomaly, 2)")

    Returns
    -------
    xr.DataArray
        Computed damages with appropriate dimensions from broadcasting

    Examples
    --------
    >>> damages = compute_damages_from_climate(
    ...     fair_pulse,
    ...     damage_function_coefficients,
    ...     "damages ~ -1 + anomaly + np.power(anomaly, 2)"
    ... )

    Notes
    -----
    This function handles different formula types:
    - Temperature-only: "damages ~ -1 + anomaly + np.power(anomaly, 2)"
    - GMSL-only: "damages ~ -1 + gmsl + np.power(gmsl, 2)"
    - Combined: "damages ~ -1 + anomaly + np.power(anomaly, 2) + gmsl + np.power(gmsl, 2)"

    The function broadcasts the coefficients and climate data to align dimensions.
    """
    # Broadcast coefficients and climate data to align dimensions
    betas, climate = xr.broadcast(damage_function_coefficients, climate_data)

    # Determine temperature variable name from climate data
    # Check for common temperature variable names
    # Also check for FAIR-specific names (control_temperature, pulse_temperature, etc.)
    temp_var = None
    for var_name in ['temperature', 'anomaly', 'temp', 'tas',
                      'control_temperature', 'pulse_temperature',
                      'medianparams_control_temperature', 'medianparams_pulse_temperature']:
        if var_name in climate.data_vars:
            temp_var = var_name
            break

    # Parse formula to understand structure
    # Handle different common formulas

    if formula == "damages ~ -1 + anomaly + np.power(anomaly, 2)":
        # Temperature-only quadratic
        if temp_var is None:
            raise ValueError(f"Temperature variable not found in climate data. Available variables: {list(climate.data_vars)}")
        temp = climate[temp_var]
        damages = (
            betas["anomaly"] * temp +
            betas["np.power(anomaly, 2)"] * np.power(temp, 2)
        )

    elif formula == "damages ~ -1 + np.power(anomaly, 2)":
        # Temperature-only quadratic (no linear term)
        if temp_var is None:
            raise ValueError(f"Temperature variable not found in climate data. Available variables: {list(climate.data_vars)}")
        temp = climate[temp_var]
        damages = betas["np.power(anomaly, 2)"] * np.power(temp, 2)

    elif formula == "damages ~ anomaly + np.power(anomaly, 2)":
        # Temperature with intercept
        if temp_var is None:
            raise ValueError(f"Temperature variable not found in climate data. Available variables: {list(climate.data_vars)}")
        temp = climate[temp_var]
        damages = (
            betas["Intercept"] +
            betas["anomaly"] * temp +
            betas["np.power(anomaly, 2)"] * np.power(temp, 2)
        )

    elif formula == "damages ~ -1 + gmsl + np.power(gmsl, 2)":
        # GMSL-only quadratic
        damages = (
            betas["gmsl"] * climate.gmsl +
            betas["np.power(gmsl, 2)"] * np.power(climate.gmsl, 2)
        )

    elif formula == "damages ~ -1 + gmsl":
        # GMSL-only linear
        damages = betas["gmsl"] * climate.gmsl

    elif formula == "damages ~ -1 + anomaly + np.power(anomaly, 2) + gmsl + np.power(gmsl, 2)":
        # Combined temperature and GMSL
        if temp_var is None:
            raise ValueError(f"Temperature variable not found in climate data. Available variables: {list(climate.data_vars)}")
        temp = climate[temp_var]
        damages = (
            betas["anomaly"] * temp +
            betas["np.power(anomaly, 2)"] * np.power(temp, 2) +
            betas["gmsl"] * climate.gmsl +
            betas["np.power(gmsl, 2)"] * np.power(climate.gmsl, 2)
        )

    elif formula == "damages ~ -1 + anomaly * gmsl + anomaly * np.power(gmsl, 2) + gmsl * np.power(anomaly, 2) + np.power(anomaly, 2) * np.power(gmsl, 2)":
        # Interaction terms
        if temp_var is None:
            raise ValueError(f"Temperature variable not found in climate data. Available variables: {list(climate.data_vars)}")
        temp = climate[temp_var]
        damages = (
            betas["anomaly:gmsl"] * temp * climate.gmsl +
            betas["anomaly:np.power(gmsl, 2)"] * temp * np.power(climate.gmsl, 2) +
            betas["gmsl:np.power(anomaly, 2)"] * climate.gmsl * np.power(temp, 2) +
            betas["np.power(anomaly, 2):np.power(gmsl, 2)"] * np.power(temp, 2) * np.power(climate.gmsl, 2)
        )

    elif formula == "damages ~ -1 + anomaly:gmsl + anomaly:np.power(gmsl, 2) + gmsl:np.power(anomaly, 2) + np.power(anomaly, 2):np.power(gmsl, 2)":
        # Interaction terms (alternative syntax)
        if temp_var is None:
            raise ValueError(f"Temperature variable not found in climate data. Available variables: {list(climate.data_vars)}")
        temp = climate[temp_var]
        damages = (
            betas["anomaly:gmsl"] * temp * climate.gmsl +
            betas["anomaly:np.power(gmsl, 2)"] * temp * np.power(climate.gmsl, 2) +
            betas["gmsl:np.power(anomaly, 2)"] * climate.gmsl * np.power(temp, 2) +
            betas["np.power(anomaly, 2):np.power(gmsl, 2)"] * np.power(temp, 2) * np.power(climate.gmsl, 2)
        )

    else:
        raise NotImplementedError(
            f"Formula '{formula}' is not implemented. "
            "Supported formulas are quadratic and interaction terms for anomaly and/or gmsl."
        )

    return damages


def calculate_marginal_damages_from_fair(
    fair_control: xr.Dataset,
    fair_pulse: xr.Dataset,
    damage_function_coefficients: xr.Dataset,
    formula: str,
    global_consumption: Optional[xr.DataArray] = None,
    pulse_conversion_factor: float = 1.0,
    fair_aggregation: Optional[list] = None,
    fair_dims: Optional[list] = None,
    weitzman_parameters: Optional[list] = None,
    eta: float = 1.421158116
) -> xr.DataArray:
    """
    Calculate marginal damages from FAIR climate projections.

    This computes the difference in damages between pulse and control scenarios,
    with proper aggregation over climate uncertainty dimensions following the
    original dscim approach.

    Parameters
    ----------
    fair_control : xr.Dataset
        FAIR control scenario (no pulse) with 'temperature' and/or 'gmsl'
    fair_pulse : xr.Dataset
        FAIR pulse scenario with 'temperature' and/or 'gmsl'
    damage_function_coefficients : xr.Dataset
        Damage function coefficients with dimensions (discount_type, ssp, model, year)
    formula : str
        Damage function formula
    global_consumption : xr.DataArray, optional
        Global consumption without climate change. Required for 'ce' aggregation.
    pulse_conversion_factor : float, default 1.0
        Conversion factor for pulse (e.g., to convert to per-tonne basis)
    fair_aggregation : list, optional
        List of aggregation methods: ['ce', 'mean', 'gwr_mean', 'median', 'median_params']
        If None, returns uncollapsed marginal damages.
    fair_dims : list, optional
        List of FAIR dimensions to collapse (e.g., ['simulation', 'gcm'])
        Default is ['simulation']
    weitzman_parameters : list, optional
        List of Weitzman parameters for bottom-coding (used with 'ce' aggregation)
    eta : float, default 1.421158116
        Risk aversion parameter for certainty equivalent calculation

    Returns
    -------
    xr.DataArray
        Marginal damages with 'fair_aggregation' dimension if aggregation methods
        are specified, otherwise uncollapsed marginal damages

    Examples
    --------
    >>> md = calculate_marginal_damages_from_fair(
    ...     fair_control,
    ...     fair_pulse,
    ...     damage_function_coefficients,
    ...     "damages ~ -1 + anomaly + np.power(anomaly, 2)",
    ...     global_consumption=global_cons,
    ...     fair_aggregation=['ce', 'mean'],
    ...     fair_dims=['simulation'],
    ...     weitzman_parameters=[0.1, 1.0],
    ...     pulse_conversion_factor=1e12
    ... )

    Notes
    -----
    The marginal damages represent the additional damages caused by the pulse
    relative to the control scenario. This is the key input for SCC calculation.

    When fair_aggregation is specified:
    - 'ce': Certainty equivalent using Weitzman bottom-coding and CRRA utility
    - 'mean': Simple mean over fair_dims
    - 'gwr_mean': Mean over fair_dims (alternative name)
    - 'median_params': Uses median climate parameters (not yet implemented)
    - 'median': Median over fair_dims (computed post-SCC in original)
    """
    from dscim_new.core.utils import crra_certainty_equivalent

    # Auto-detect FAIR dimensions if not specified
    if fair_dims is None:
        fair_dims = []
        # Check for common FAIR uncertainty dimensions
        for potential_dim in ['simulation', 'gcm', 'rcp', 'ssp']:
            # Check in either control or pulse datasets
            if (potential_dim in fair_control.dims or
                potential_dim in fair_pulse.dims):
                # Don't include ssp/rcp as they're scenario dimensions, not uncertainty
                if potential_dim not in ['ssp', 'rcp']:
                    fair_dims.append(potential_dim)

        # If no uncertainty dims found, default to empty list (no collapsing)
        if not fair_dims:
            fair_dims = []

        print(f"Auto-detected FAIR uncertainty dimensions: {fair_dims}")

    # If no aggregation specified, return simple uncollapsed marginal damages
    if fair_aggregation is None:
        # Compute damages for both scenarios
        damages_pulse = compute_damages_from_climate(
            fair_pulse,
            damage_function_coefficients,
            formula
        )

        damages_control = compute_damages_from_climate(
            fair_control,
            damage_function_coefficients,
            formula
        )

        # Calculate marginal damages (difference)
        marginal_damages = damages_pulse - damages_control

        # Apply conversion factor
        marginal_damages = marginal_damages * pulse_conversion_factor

        return marginal_damages

    # Otherwise, compute marginal damages for each aggregation method
    marginal_damages_list = []

    for agg in [a for a in fair_aggregation if a != 'median']:
        if agg == 'ce':
            # Certainty equivalent aggregation requires consumption
            if global_consumption is None:
                raise ValueError("global_consumption is required for 'ce' aggregation")
            if weitzman_parameters is None:
                raise ValueError("weitzman_parameters is required for 'ce' aggregation")

            # Compute damages for both scenarios
            damages_pulse = compute_damages_from_climate(
                fair_pulse,
                damage_function_coefficients,
                formula
            )
            damages_control = compute_damages_from_climate(
                fair_control,
                damage_function_coefficients,
                formula
            )

            # For each Weitzman parameter, compute CE-aggregated marginal damages
            ce_md_list = []
            for wp in weitzman_parameters:
                # Calculate consumption with climate change
                cc_cons_control = global_consumption - damages_control
                cc_cons_pulse = global_consumption - damages_pulse

                # Apply Weitzman bottom-coding
                cc_cons_control_coded = _apply_weitzman_bottom_coding(
                    cc_cons_control, global_consumption, wp, eta
                )
                cc_cons_pulse_coded = _apply_weitzman_bottom_coding(
                    cc_cons_pulse, global_consumption, wp, eta
                )

                # Determine which dims to collapse
                dims_to_collapse = [d for d in fair_dims if d in cc_cons_control_coded.dims]

                # Apply certainty equivalent
                if dims_to_collapse:
                    ce_control = crra_certainty_equivalent(
                        cc_cons_control_coded, eta, dims_to_collapse
                    )
                    ce_pulse = crra_certainty_equivalent(
                        cc_cons_pulse_coded, eta, dims_to_collapse
                    )
                else:
                    ce_control = cc_cons_control_coded
                    ce_pulse = cc_cons_pulse_coded

                # Marginal damages = CE_control - CE_pulse
                md_ce = ce_control - ce_pulse

                # Add weitzman_parameter coordinate
                md_ce = md_ce.assign_coords({"weitzman_parameter": str(wp)})
                ce_md_list.append(md_ce)

            # Concatenate over weitzman_parameter dimension
            md = xr.concat(ce_md_list, dim="weitzman_parameter")

        elif agg in ['mean', 'gwr_mean']:
            # Simple mean aggregation
            # Compute damages for both scenarios
            damages_pulse = compute_damages_from_climate(
                fair_pulse,
                damage_function_coefficients,
                formula
            )
            damages_control = compute_damages_from_climate(
                fair_control,
                damage_function_coefficients,
                formula
            )

            # Calculate marginal damages
            md = damages_pulse - damages_control

            # Determine which dims to collapse
            dims_to_collapse = [d for d in fair_dims if d in md.dims]

            # Mean over fair_dims
            if dims_to_collapse:
                md = md.mean(dim=dims_to_collapse)

            # Add weitzman_parameter dimension to match 'ce' structure
            if weitzman_parameters:
                md = md.expand_dims(
                    {"weitzman_parameter": [str(wp) for wp in weitzman_parameters]}
                )

        elif agg == 'median_params':
            # This would require median climate parameters scenario
            # For now, raise not implemented
            raise NotImplementedError(
                "median_params aggregation requires separate FAIR median params scenario"
            )

        # Add fair_aggregation coordinate and append
        md = md.assign_coords({"fair_aggregation": agg})
        marginal_damages_list.append(md)

    # Concatenate all aggregation methods
    marginal_damages = xr.concat(marginal_damages_list, dim="fair_aggregation")

    # Apply conversion factor
    marginal_damages = marginal_damages * pulse_conversion_factor

    return marginal_damages


def _apply_weitzman_bottom_coding(
    cc_consumption: xr.DataArray,
    no_cc_consumption: xr.DataArray,
    parameter: float,
    eta: float
) -> xr.DataArray:
    """
    Apply Weitzman bottom-coding to consumption.

    This implements bottom coding that fixes marginal utility below a threshold
    to the marginal utility at that threshold.

    Parameters
    ----------
    cc_consumption : xr.DataArray
        Consumption with climate change
    no_cc_consumption : xr.DataArray
        Consumption without climate change (used to calculate threshold)
    parameter : float
        Weitzman parameter (share or absolute value)
    eta : float
        Risk aversion parameter

    Returns
    -------
    xr.DataArray
        Bottom-coded consumption
    """
    from dscim_new.core.utils import power

    # If parameter <= 1, treat as share of no-cc consumption
    if parameter <= 1:
        threshold = parameter * no_cc_consumption
    else:
        threshold = parameter

    if eta == 1:
        # Log utility case
        w_utility = np.log(threshold)
        bottom_utility = np.power(threshold, -1) * (threshold - cc_consumption)
        bottom_coded_cons = np.exp(w_utility - bottom_utility)

        clipped_cons = xr.where(
            cc_consumption > threshold, cc_consumption, bottom_coded_cons
        )
    else:
        # Power utility case
        w_utility = np.power(threshold, (1 - eta)) / (1 - eta)
        bottom_utility = np.power(threshold, -eta) * (threshold - cc_consumption)
        bottom_coded_cons = power(
            ((1 - eta) * (w_utility - bottom_utility)), (1 / (1 - eta))
        )

        clipped_cons = xr.where(
            cc_consumption > threshold, cc_consumption, bottom_coded_cons
        )

    return clipped_cons
