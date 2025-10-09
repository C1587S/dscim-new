"""
Damage function preprocessing and generation.

Orchestrates damage function fitting workflow by coordinating core functions and I/O.
"""

import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

from ..core.damage_functions import (
    fit_damage_function_ols,
    evaluate_damage_function,
    calculate_marginal_damages,
    extrapolate_damages,
    compute_damage_function_points,
    compute_damage_function_grid,  # New: for original dscim format
    fit_damage_function_rolling_window,  # New: for original dscim per-year fitting
    compute_damages_from_climate,  # New: compute damages from FAIR projections
    calculate_marginal_damages_from_fair,  # New: per-scenario marginal damages
)
from ..io.climate_io import (
    load_gmst_data,
    load_gmsl_data,
    get_climate_variable_for_sector,
    match_climate_to_damages,
)
from ..io.damage_function_io import save_full_damage_function
from ..config.schemas import DamageFunctionConfig, ClimateDataConfig

try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.panel import Panel
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


@dataclass
class DamageFunctionResult:
    """
    Result of damage function generation.

    Attributes
    ----------
    coefficients : xr.DataArray
        Fitted regression coefficients (new format - for comparison)
    coefficients_grid : xr.Dataset
        Predicted damage values on climate grid (original dscim format)
        Variables are formula terms (e.g., 'anomaly', 'np.power(anomaly, 2)')
    marginal_damages : xr.DataArray, optional
        Marginal damages evaluated at climate values
    fit_stats : xr.Dataset
        Fit statistics (R-squared, n_obs, etc.)
    points : xr.Dataset, optional
        Evaluation points for visualization
    climate_range : tuple, optional
        (min, max) range of climate variable used
    """
    coefficients: xr.DataArray
    coefficients_grid: xr.Dataset  # New: grid-based predictions (original format)
    marginal_damages: Optional[xr.DataArray]
    fit_stats: xr.Dataset
    points: Optional[xr.Dataset]
    climate_range: Optional[Tuple[float, float]]


class DamageFunctionProcessor:
    """
    Processor for generating damage functions from reduced damages and climate data.

    This class orchestrates the damage function fitting workflow:
    1. Load reduced damages and climate data
    2. Fit damage function using OLS or quantile regression
    3. Calculate marginal damages
    4. Extrapolate damages beyond projection period
    5. Save outputs in various formats

    Parameters
    ----------
    config : DamageFunctionConfig
        Damage function configuration
    climate_config : ClimateDataConfig
        Climate data configuration
    verbose : bool, default True
        Whether to print progress messages

    Examples
    --------
    >>> processor = DamageFunctionProcessor(df_config, climate_config)
    >>> result = processor.generate_damage_function(
    ...     damages=reduced_damages,
    ...     sector="mortality",
    ...     pulse_year=2020
    ... )
    >>> print(result.coefficients)
    """

    def __init__(
        self,
        config: DamageFunctionConfig,
        climate_config: ClimateDataConfig,
        verbose: bool = True
    ):
        self.config = config
        self.climate_config = climate_config
        self.verbose = verbose
        self.console = Console() if RICH_AVAILABLE and verbose else None

    def _log(self, message: str, style: str = ""):
        """Print message if verbose."""
        if self.verbose:
            if self.console:
                self.console.print(message, style=style)
            else:
                print(message)

    def load_climate_data(
        self,
        sector: str,
        use_dask: bool = True
    ) -> Tuple[Optional[pd.DataFrame], Optional[xr.Dataset]]:
        """
        Load climate data based on sector requirements.

        Parameters
        ----------
        sector : str
            Sector name
        use_dask : bool, default True
            Whether to use Dask for loading

        Returns
        -------
        tuple
            (gmst_data, gmsl_data) - at least one will be non-None

        Raises
        ------
        ValueError
            If required climate data is not configured
        """
        gmst_data = None
        gmsl_data = None

        # Determine if coastal sector
        is_coastal = "coastal" in sector.lower()

        if is_coastal:
            # Load GMSL for coastal sectors
            if not self.climate_config.gmsl_path:
                raise ValueError(f"GMSL data required for coastal sector: {sector}")
            self._log(f"Loading GMSL data from {self.climate_config.gmsl_path}")
            gmsl_data = load_gmsl_data(self.climate_config.gmsl_path, use_dask=use_dask)
        else:
            # Load GMST for non-coastal sectors
            if not self.climate_config.gmst_path:
                raise ValueError(f"GMST data required for sector: {sector}")
            self._log(f"Loading GMST data from {self.climate_config.gmst_path}")
            gmst_data = load_gmst_data(self.climate_config.gmst_path, use_dask=use_dask)

        return gmst_data, gmsl_data

    def fit_damage_function(
        self,
        damages: xr.DataArray,
        climate_var: xr.DataArray,
        coords_to_stack: Optional[list] = None
    ) -> Tuple[xr.DataArray, xr.Dataset]:
        """
        Fit damage function to damages and climate data.

        Parameters
        ----------
        damages : xr.DataArray
            Damage values
        climate_var : xr.DataArray
            Climate variable (temperature or sea level)
        coords_to_stack : list, optional
            Coordinates to stack for regression

        Returns
        -------
        tuple
            (coefficients, fit_stats)
        """
        self._log("Fitting damage function using OLS regression", style="bold blue")

        # Fit using core function
        result = fit_damage_function_ols(
            damages=damages,
            climate_var=climate_var,
            formula=self.config.formula,
            coords_to_stack=coords_to_stack
        )

        coefficients = result['coefficients']
        fit_stats = result

        # Log fit quality
        if 'rsquared' in fit_stats.data_vars:
            r2 = float(fit_stats['rsquared'].values)
            self._log(f"R-squared: {r2:.4f}", style="green")
        if 'n_obs' in fit_stats.data_vars:
            n_obs = int(fit_stats['n_obs'].values)
            self._log(f"Number of observations: {n_obs}", style="green")

        return coefficients, fit_stats

    def compute_marginal_damages(
        self,
        coefficients: xr.DataArray,
        climate_values: xr.DataArray
    ) -> xr.DataArray:
        """
        Calculate marginal damages at specified climate values.

        Parameters
        ----------
        coefficients : xr.DataArray
            Fitted coefficients
        climate_values : xr.DataArray
            Climate values at which to evaluate marginal damages

        Returns
        -------
        xr.DataArray
            Marginal damages
        """
        self._log("Calculating marginal damages", style="bold blue")

        marginal_damages = calculate_marginal_damages(
            coefficients=coefficients,
            climate_values=climate_values,
            formula=self.config.formula
        )

        return marginal_damages

    def generate_evaluation_points(
        self,
        coefficients: xr.DataArray,
        climate_range: Tuple[float, float]
    ) -> xr.Dataset:
        """
        Generate evaluation points for damage function visualization.

        Parameters
        ----------
        coefficients : xr.DataArray
            Fitted coefficients
        climate_range : tuple
            (min, max) range of climate variable

        Returns
        -------
        xr.Dataset
            Evaluation points with climate values and predicted damages
        """
        if not self.config.save_points:
            return None

        self._log(f"Generating {self.config.n_points} evaluation points for visualization")

        points = compute_damage_function_points(
            coefficients=coefficients,
            climate_range=climate_range,
            formula=self.config.formula,
            n_points=self.config.n_points
        )

        return points

    def generate_damage_function(
        self,
        damages: xr.DataArray,
        sector: str,
        pulse_year: int,
        climate_data: Optional[Tuple[pd.DataFrame, xr.Dataset]] = None,
        discount_type: str = "constant",
        save_outputs: bool = True,
        output_dir: Optional[str] = None,
        output_format: str = "zarr"
    ) -> DamageFunctionResult:
        """
        Generate complete damage function from reduced damages.

        This is the main method that orchestrates the entire workflow.

        Parameters
        ----------
        damages : xr.DataArray
            Reduced damage values
        sector : str
            Sector name
        pulse_year : int
            Year of carbon pulse
        climate_data : tuple, optional
            Pre-loaded (gmst_data, gmsl_data). If None, loads from config.
        discount_type : str, default "constant"
            Type of discounting to use (e.g., "constant", "ramsey", "gwr")
            This dimension is added to outputs to match original dscim format
        save_outputs : bool, default True
            Whether to save outputs to disk
        output_dir : str, optional
            Output directory (overrides config paths)
        output_format : str, default "zarr"
            Output format for saved files

        Returns
        -------
        DamageFunctionResult
            Complete damage function result

        Examples
        --------
        >>> processor = DamageFunctionProcessor(df_config, climate_config)
        >>> result = processor.generate_damage_function(
        ...     damages=reduced_damages,
        ...     sector="mortality",
        ...     pulse_year=2020,
        ...     discount_type="constant"
        ... )
        """
        if self.console and RICH_AVAILABLE:
            self.console.print(
                Panel(
                    f"Generating damage function for [bold]{sector}[/bold], pulse year [bold]{pulse_year}[/bold]",
                    style="blue"
                )
            )

        # Load climate data if not provided
        if climate_data is None:
            gmst_data, gmsl_data = self.load_climate_data(sector)
        else:
            gmst_data, gmsl_data = climate_data

        # Get appropriate climate variable for sector
        climate_var = get_climate_variable_for_sector(
            sector=sector,
            gmst_data=gmst_data,
            gmsl_data=gmsl_data
        )

        # Match climate data to damage coordinates
        damage_coords = {dim: damages.coords[dim] for dim in damages.dims}
        climate_matched = match_climate_to_damages(
            climate_data=climate_var,
            damage_coords=damage_coords
        )

        # Calculate climate range for evaluation points
        climate_range = (
            float(climate_matched.min().values),
            float(climate_matched.max().values)
        )

        # Choose fitting method based on configuration
        if self.config.fit_method == "rolling_window":
            # Original dscim approach: fit separately for each (ssp, model, year)
            self._log("Fitting damage functions with rolling window approach (per year/scenario)", style="bold blue")

            # Auto-detect dimension names from the CLIMATE data (not damages)
            # The climate data determines what dimensions we can actually fit over
            ssp_dim = None
            model_dim = None
            year_dim = None
            region_dim = None

            # Check climate data dimensions
            for dim in climate_matched.dims:
                if 'ssp' in dim.lower() or 'rcp' in dim.lower() or 'scenario' in dim.lower():
                    ssp_dim = dim
                if 'model' in dim.lower() or 'gcm' in dim.lower():
                    model_dim = dim
                if 'year' in dim.lower():
                    year_dim = dim

            # Check damage data for region dimension
            for dim in damages.dims:
                if 'region' in dim.lower() or 'hierid' in dim.lower():
                    region_dim = dim

            # Log detected dimensions
            self._log(f"Detected dimensions from climate data: ssp={ssp_dim}, model={model_dim}, year={year_dim}")
            self._log(f"Detected region dimension from damages: region={region_dim}")

            # Use detected dimensions (fallback to defaults if not found)
            coefficients_grid = fit_damage_function_rolling_window(
                damages=damages,
                climate_var=climate_matched,
                formula=self.config.formula,
                year_range=range(self.config.year_range[0], self.config.year_range[1]),
                window_size=self.config.window_size,
                ssp_dim=ssp_dim or 'ssp',
                model_dim=model_dim or 'model',
                year_dim=year_dim or 'year',
                region_dim=region_dim or 'region',
            )

            # For rolling window, coefficients_grid IS the main output
            # Extract a "representative" coefficient for backwards compatibility
            # (e.g., mean across ssp/model/year)
            coefficients = None
            for var_name in coefficients_grid.data_vars:
                if coefficients is None:
                    coefficients = xr.DataArray(
                        [coefficients_grid[var_name].mean().values],
                        dims=['coefficient'],
                        coords={'coefficient': [var_name]}
                    )
                else:
                    new_coef = xr.DataArray(
                        [coefficients_grid[var_name].mean().values],
                        dims=['coefficient'],
                        coords={'coefficient': [var_name]}
                    )
                    coefficients = xr.concat([coefficients, new_coef], dim='coefficient')

            fit_stats = xr.Dataset({
                'method': 'rolling_window',
                'window_size': self.config.window_size,
            })

            self._log(f"Rolling window fitting complete: {coefficients_grid.dims}", style="green")

        else:
            # New approach: single global fit
            self._log("Fitting damage function using global OLS regression", style="bold blue")

            coefficients, fit_stats = self.fit_damage_function(
                damages=damages,
                climate_var=climate_matched
            )

            # Generate grid-based predictions (original dscim format)
            # This creates the "damage_function_coefficients" structure that matches original
            self._log("Generating grid-based predictions (original dscim format)", style="bold blue")
            coefficients_grid = compute_damage_function_grid(
                coefficients=coefficients,
                formula=self.config.formula,
                min_anomaly=0.0,
                max_anomaly=20.0,
                step_anomaly=0.2,
                min_gmsl=0.0,
                max_gmsl=300.0,
                step_gmsl=3.0,
            )
            self._log(f"Grid predictions generated: {list(coefficients_grid.data_vars)}", style="green")

        # Add discount_type dimension to match original dscim format
        # This is done AFTER fitting to maintain the same structure as original
        self._log(f"Adding discount_type dimension: {discount_type}", style="bold blue")
        coefficients_grid = coefficients_grid.expand_dims(
            {"discount_type": [discount_type]}
        )
        self._log(f"Final dimensions: {coefficients_grid.dims}", style="green")

        # Calculate marginal damages
        marginal_damages = self.compute_marginal_damages(
            coefficients=coefficients,
            climate_values=climate_matched
        )

        # Generate evaluation points for visualization
        points = self.generate_evaluation_points(
            coefficients=coefficients,
            climate_range=climate_range
        )

        # Save outputs if requested
        if save_outputs and output_dir:
            self._log(f"Saving damage function outputs to {output_dir}", style="bold green")
            saved_paths = save_full_damage_function(
                coefficients=coefficients,
                coefficients_grid=coefficients_grid,  # Add grid predictions
                marginal_damages=marginal_damages,
                points=points,
                fit_stats=fit_stats,
                output_dir=output_dir,
                sector=sector,
                pulse_year=pulse_year,
                formula=self.config.formula,
                format=output_format
            )
            self._log(f"Saved {len(saved_paths)} files", style="green")

        # Create result
        result = DamageFunctionResult(
            coefficients=coefficients,
            coefficients_grid=coefficients_grid,  # Add grid predictions
            marginal_damages=marginal_damages,
            fit_stats=fit_stats,
            points=points,
            climate_range=climate_range
        )

        if self.console:
            self.console.print("[bold green]✓[/bold green] Damage function generation complete")

        return result

    def batch_generate_damage_functions(
        self,
        damages_dict: Dict[Tuple[str, int], xr.DataArray],
        output_dir: str,
        output_format: str = "zarr"
    ) -> Dict[Tuple[str, int], DamageFunctionResult]:
        """
        Generate damage functions for multiple sectors and pulse years.

        Parameters
        ----------
        damages_dict : dict
            Dictionary mapping (sector, pulse_year) to damage DataArrays
        output_dir : str
            Output directory for all results
        output_format : str, default "zarr"
            Output format

        Returns
        -------
        dict
            Dictionary mapping (sector, pulse_year) to DamageFunctionResult

        Examples
        --------
        >>> damages_dict = {
        ...     ("mortality", 2020): mortality_damages,
        ...     ("coastal", 2020): coastal_damages
        ... }
        >>> results = processor.batch_generate_damage_functions(
        ...     damages_dict,
        ...     output_dir="outputs/damage_functions"
        ... )
        """
        results = {}

        total = len(damages_dict)
        self._log(f"Processing {total} damage function(s)", style="bold")

        for i, ((sector, pulse_year), damages) in enumerate(damages_dict.items(), 1):
            self._log(f"\n[{i}/{total}] Processing {sector}, pulse year {pulse_year}")

            result = self.generate_damage_function(
                damages=damages,
                sector=sector,
                pulse_year=pulse_year,
                save_outputs=True,
                output_dir=output_dir,
                output_format=output_format
            )

            results[(sector, pulse_year)] = result

        if self.console:
            self.console.print(
                f"\n[bold green]✓ Completed all {total} damage functions[/bold green]"
            )

        return results
