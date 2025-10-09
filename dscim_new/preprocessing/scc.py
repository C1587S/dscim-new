"""
Social Cost of Carbon (SCC) calculation preprocessing.

Orchestrates SCC calculation workflow by coordinating core functions and I/O.
"""

import xarray as xr
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass

from ..core.discounting import calculate_discount_factors
from ..core.scc_calculation import (
    calculate_scc,
    calculate_global_consumption,
    aggregate_scc_over_fair,
    calculate_scc_quantiles,
    calculate_uncollapsed_scc,
)
from ..io.scc_io import save_full_scc
from ..config.schemas import DiscountingConfig, SCCConfig

try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.panel import Panel
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


@dataclass
class SCCResult:
    """
    Result of SCC calculation.

    Attributes
    ----------
    scc : xr.DataArray
        Aggregated SCC values
    scc_uncollapsed : xr.DataArray, optional
        Full SCC distribution (before aggregation)
    discount_factors : xr.DataArray
        Discount factors used
    quantiles : xr.Dataset, optional
        SCC quantiles for uncertainty bounds
    scc_mean : float
        Mean SCC value
    scc_median : float
        Median SCC value
    """
    scc: xr.DataArray
    scc_uncollapsed: Optional[xr.DataArray]
    discount_factors: xr.DataArray
    quantiles: Optional[xr.Dataset]
    scc_mean: float
    scc_median: float


class SCCCalculator:
    """
    Calculator for Social Cost of Carbon from marginal damages and consumption.

    This class orchestrates the SCC calculation workflow:
    1. Calculate discount factors (constant, Ramsey, or GWR)
    2. Multiply marginal damages by discount factors
    3. Sum over time to get SCC
    4. Aggregate over climate uncertainty
    5. Calculate quantiles and summary statistics
    6. Save outputs

    Parameters
    ----------
    scc_config : SCCConfig
        SCC calculation configuration
    verbose : bool, default True
        Whether to print progress messages

    Examples
    --------
    >>> calculator = SCCCalculator(scc_config)
    >>> result = calculator.calculate_scc(
    ...     marginal_damages=md,
    ...     consumption=consumption,
    ...     discount_config=discount_config,
    ...     pulse_year=2020
    ... )
    >>> print(result.scc_mean)
    """

    def __init__(
        self,
        scc_config: SCCConfig,
        verbose: bool = True
    ):
        self.config = scc_config
        self.verbose = verbose
        self.console = Console() if RICH_AVAILABLE and verbose else None

    def _log(self, message: str, style: str = ""):
        """Print message if verbose."""
        if self.verbose:
            if self.console:
                self.console.print(message, style=style)
            else:
                print(message)

    def calculate_discount_factors_for_scc(
        self,
        discount_config: DiscountingConfig,
        consumption: Optional[xr.DataArray] = None,
        years: Optional[xr.DataArray] = None,
        pulse_year: int = 2020
    ) -> xr.DataArray:
        """
        Calculate discount factors based on discounting configuration.

        Parameters
        ----------
        discount_config : DiscountingConfig
            Discounting configuration
        consumption : xr.DataArray, optional
            Consumption data (required for Ramsey/GWR)
        years : xr.DataArray, optional
            Years (required for constant discounting)
        pulse_year : int, default 2020
            Base year for discounting

        Returns
        -------
        xr.DataArray
            Discount factors

        Raises
        ------
        ValueError
            If required data is missing for discount type
        """
        self._log(
            f"Calculating {discount_config.discount_type} discount factors",
            style="bold blue"
        )

        if discount_config.discount_type == "constant":
            if years is None:
                raise ValueError("years required for constant discounting")
            discount_factors = calculate_discount_factors(
                discount_type="constant",
                years=years,
                discount_rate=discount_config.discount_rate,
                pulse_year=pulse_year,
                discrete=discount_config.discrete
            )
            self._log(
                f"Using constant discount rate: {discount_config.discount_rate:.4f}",
                style="green"
            )

        elif discount_config.discount_type == "ramsey":
            if consumption is None:
                raise ValueError("consumption required for Ramsey discounting")
            discount_factors = calculate_discount_factors(
                discount_type="ramsey",
                consumption=consumption,
                eta=discount_config.eta,
                rho=discount_config.rho,
                pulse_year=pulse_year,
                discrete=discount_config.discrete
            )
            self._log(
                f"Using Ramsey discounting: eta={discount_config.eta}, rho={discount_config.rho}",
                style="green"
            )

        elif discount_config.discount_type == "gwr":
            if consumption is None:
                raise ValueError("consumption required for GWR discounting")
            discount_factors = calculate_discount_factors(
                discount_type="gwr",
                consumption=consumption,
                eta=discount_config.eta,
                rho=discount_config.rho,
                pulse_year=pulse_year,
                gwr_method=discount_config.gwr_method
            )
            self._log(
                f"Using GWR discounting ({discount_config.gwr_method}): "
                f"eta={discount_config.eta}, rho={discount_config.rho}",
                style="green"
            )

        else:
            raise ValueError(f"Unknown discount type: {discount_config.discount_type}")

        return discount_factors

    def aggregate_and_summarize(
        self,
        scc_uncollapsed: xr.DataArray,
        fair_dims: Optional[List[str]] = None
    ) -> Tuple[xr.DataArray, Optional[xr.Dataset], float, float]:
        """
        Aggregate SCC over FAIR dimensions and calculate summary statistics.

        Parameters
        ----------
        scc_uncollapsed : xr.DataArray
            Uncollapsed SCC values
        fair_dims : list of str, optional
            Dimensions to aggregate over. If None, uses ["simulation"]

        Returns
        -------
        tuple
            (scc_aggregated, quantiles, mean, median)
        """
        if fair_dims is None:
            # Default to aggregating over simulation dimension if it exists
            fair_dims = [d for d in ["simulation", "sim"] if d in scc_uncollapsed.dims]
            if not fair_dims:
                # No simulation dims, aggregate over other uncertainty dims
                uncertainty_dims = ['rcp', 'gcm', 'model', 'scenario']
                fair_dims = [d for d in scc_uncollapsed.dims if d in uncertainty_dims]
            if not fair_dims:
                fair_dims = None  # Let aggregate function decide

        # Aggregate SCC
        self._log(f"Aggregating SCC using method: {self.config.fair_aggregation}")
        scc_aggregated = aggregate_scc_over_fair(
            scc=scc_uncollapsed,
            method=self.config.fair_aggregation,
            dims=fair_dims
        )

        # Calculate quantiles if requested
        quantiles = None
        if self.config.calculate_quantiles:
            self._log(f"Calculating quantiles: {self.config.quantile_levels}")
            quantiles = calculate_scc_quantiles(
                scc=scc_uncollapsed,
                quantiles=self.config.quantile_levels,
                dims=fair_dims
            )

        # Note: Summary statistics are not computed here to avoid Dask issues
        # Users can compute statistics after loading results if needed
        self._log("SCC calculation complete", style="bold green")

        return scc_aggregated, quantiles

    def calculate_scc(
        self,
        marginal_damages: xr.DataArray,
        discount_config: DiscountingConfig,
        pulse_year: int,
        consumption: Optional[xr.DataArray] = None,
        sector: Optional[str] = None,
        save_outputs: bool = True,
        output_dir: Optional[str] = None,
        output_format: str = "zarr"
    ) -> SCCResult:
        """
        Calculate Social Cost of Carbon from marginal damages.

        This is the main method that orchestrates the entire SCC calculation.

        Parameters
        ----------
        marginal_damages : xr.DataArray
            Marginal damages ($/tCO2) with year dimension
        discount_config : DiscountingConfig
            Discounting configuration
        pulse_year : int
            Year of carbon pulse
        consumption : xr.DataArray, optional
            Consumption data (required for Ramsey/GWR discounting)
        sector : str, optional
            Sector name (for output filenames)
        save_outputs : bool, default True
            Whether to save outputs to disk
        output_dir : str, optional
            Output directory
        output_format : str, default "zarr"
            Output format for saved files

        Returns
        -------
        SCCResult
            Complete SCC calculation result

        Examples
        --------
        >>> calculator = SCCCalculator(scc_config)
        >>> result = calculator.calculate_scc(
        ...     marginal_damages=md,
        ...     discount_config=ramsey_config,
        ...     pulse_year=2020,
        ...     consumption=consumption,
        ...     sector="mortality"
        ... )
        """
        if self.console and RICH_AVAILABLE:
            sector_str = f" for [bold]{sector}[/bold]" if sector else ""
            self.console.print(
                Panel(
                    f"Calculating SCC{sector_str}, pulse year [bold]{pulse_year}[/bold]",
                    style="blue"
                )
            )

        # Get years from marginal damages
        years = marginal_damages.year if 'year' in marginal_damages.dims else None

        # Calculate discount factors
        discount_factors = self.calculate_discount_factors_for_scc(
            discount_config=discount_config,
            consumption=consumption,
            years=years,
            pulse_year=pulse_year
        )

        # Calculate uncollapsed SCC (full distribution)
        self._log("Computing discounted marginal damages and summing over time")
        scc_uncollapsed = calculate_uncollapsed_scc(
            marginal_damages=marginal_damages,
            discount_factors=discount_factors,
            pulse_year=pulse_year,
            keep_year=False
        )

        # Aggregate and summarize
        scc_aggregated, quantiles = self.aggregate_and_summarize(
            scc_uncollapsed=scc_uncollapsed
        )

        # Decide what to keep
        scc_to_save = scc_aggregated
        scc_uncollapsed_to_save = scc_uncollapsed if self.config.save_uncollapsed else None

        # Save outputs if requested
        if save_outputs and output_dir and sector:
            self._log(f"Saving SCC outputs to {output_dir}", style="bold green")

            # Build discount parameters dict
            discount_params = {
                'discount_type': discount_config.discount_type
            }
            if discount_config.eta is not None:
                discount_params['eta'] = discount_config.eta
            if discount_config.rho is not None:
                discount_params['rho'] = discount_config.rho
            if discount_config.discount_rate is not None:
                discount_params['discount_rate'] = discount_config.discount_rate

            discount_factors_to_save = discount_factors if self.config.save_discount_factors else None

            saved_paths = save_full_scc(
                scc=scc_to_save,
                discount_factors=discount_factors_to_save,
                quantiles=quantiles,
                output_dir=output_dir,
                sector=sector,
                pulse_year=pulse_year,
                format=output_format,
                **discount_params
            )
            self._log(f"Saved {len(saved_paths)} files", style="green")

        # Create result
        result = SCCResult(
            scc=scc_aggregated,
            scc_uncollapsed=scc_uncollapsed_to_save,
            discount_factors=discount_factors,
            quantiles=quantiles,
            scc_mean=None,  # Not computed to avoid Dask issues
            scc_median=None  # Not computed to avoid Dask issues
        )

        if self.console:
            self.console.print("[bold green]✓[/bold green] SCC calculation complete")

        return result

    def batch_calculate_scc(
        self,
        marginal_damages_dict: Dict[Tuple[str, int], xr.DataArray],
        discount_configs: List[DiscountingConfig],
        consumption: Optional[xr.DataArray],
        output_dir: str,
        output_format: str = "zarr"
    ) -> Dict[Tuple[str, int, str], SCCResult]:
        """
        Calculate SCC for multiple sectors, pulse years, and discount configurations.

        Parameters
        ----------
        marginal_damages_dict : dict
            Dictionary mapping (sector, pulse_year) to marginal damage DataArrays
        discount_configs : list of DiscountingConfig
            List of discounting configurations to apply
        consumption : xr.DataArray, optional
            Consumption data (required for Ramsey/GWR)
        output_dir : str
            Output directory for all results
        output_format : str, default "zarr"
            Output format

        Returns
        -------
        dict
            Dictionary mapping (sector, pulse_year, discount_type) to SCCResult

        Examples
        --------
        >>> md_dict = {
        ...     ("mortality", 2020): mortality_md,
        ...     ("coastal", 2020): coastal_md
        ... }
        >>> discount_configs = [ramsey_config, constant_config]
        >>> results = calculator.batch_calculate_scc(
        ...     md_dict, discount_configs, consumption,
        ...     output_dir="outputs/scc"
        ... )
        """
        results = {}

        total = len(marginal_damages_dict) * len(discount_configs)
        self._log(f"Processing {total} SCC calculation(s)", style="bold")

        count = 0
        for (sector, pulse_year), marginal_damages in marginal_damages_dict.items():
            for discount_config in discount_configs:
                count += 1
                self._log(
                    f"\n[{count}/{total}] Processing {sector}, pulse year {pulse_year}, "
                    f"{discount_config.discount_type} discounting"
                )

                result = self.calculate_scc(
                    marginal_damages=marginal_damages,
                    discount_config=discount_config,
                    pulse_year=pulse_year,
                    consumption=consumption,
                    sector=sector,
                    save_outputs=True,
                    output_dir=output_dir,
                    output_format=output_format
                )

                key = (sector, pulse_year, discount_config.discount_type)
                results[key] = result

        if self.console:
            self.console.print(
                f"\n[bold green]✓ Completed all {total} SCC calculations[/bold green]"
            )

        return results

    def calculate_scc_from_damage_function(
        self,
        coefficients: xr.DataArray,
        climate_pulse: xr.DataArray,
        climate_control: xr.DataArray,
        discount_config: DiscountingConfig,
        pulse_year: int,
        consumption: Optional[xr.DataArray] = None,
        sector: Optional[str] = None,
        save_outputs: bool = True,
        output_dir: Optional[str] = None,
        output_format: str = "zarr"
    ) -> SCCResult:
        """
        Calculate SCC directly from damage function coefficients and FAIR climate data.

        This method combines damage function evaluation and SCC calculation in one step.

        Parameters
        ----------
        coefficients : xr.DataArray
            Fitted damage function coefficients
        climate_pulse : xr.DataArray
            FAIR climate with pulse (temperature or sea level)
        climate_control : xr.DataArray
            FAIR climate without pulse
        discount_config : DiscountingConfig
            Discounting configuration
        pulse_year : int
            Year of carbon pulse
        consumption : xr.DataArray, optional
            Consumption data
        sector : str, optional
            Sector name
        save_outputs : bool, default True
            Whether to save outputs
        output_dir : str, optional
            Output directory
        output_format : str, default "zarr"
            Output format

        Returns
        -------
        SCCResult
            SCC calculation result

        Notes
        -----
        This method evaluates the damage function at pulse and control climates,
        takes the difference to get marginal damages, then calculates SCC.
        """
        from ..core.damage_functions import evaluate_damage_function, calculate_marginal_damages
        from ..config.schemas import DamageFunctionConfig

        self._log("Evaluating damage function at pulse and control climates")

        # For now, we'll need to extract the formula from coefficients attrs
        # Or require it to be passed separately
        # This is a simplified implementation

        # Calculate climate difference (pulse - control)
        climate_diff = climate_pulse - climate_control

        # Calculate marginal damages using the damage function
        # Note: This requires the formula, which should be stored in coefficients.attrs
        if 'formula' not in coefficients.attrs:
            raise ValueError("Damage function formula not found in coefficients attributes")

        formula = coefficients.attrs['formula']

        marginal_damages = calculate_marginal_damages(
            coefficients=coefficients,
            climate_values=climate_control,  # Evaluate at control climate
            formula=formula
        )

        # Now calculate SCC using the marginal damages
        return self.calculate_scc(
            marginal_damages=marginal_damages,
            discount_config=discount_config,
            pulse_year=pulse_year,
            consumption=consumption,
            sector=sector,
            save_outputs=save_outputs,
            output_dir=output_dir,
            output_format=output_format
        )
