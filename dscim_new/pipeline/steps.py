"""
Pipeline step implementations.

Concrete implementations of pipeline steps for DSCIM workflows.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import xarray as xr

from .base import PipelineStep
from ..config import DSCIMConfig
from ..preprocessing import DamageProcessor, DamageFunctionProcessor, SCCCalculator
from ..io import OutputNaming


class ReduceDamagesStep(PipelineStep):
    """
    Reduce damages pipeline step.

    Processes sectoral damages with specified recipe and reduction type.

    Parameters
    ----------
    config : DSCIMConfig
        Validated configuration
    sector : str
        Sector name
    recipe : str
        Aggregation recipe ("adding_up" or "risk_aversion")
    reduction : str
        Reduction type ("cc" or "no_cc")
    eta : float, optional
        Risk aversion parameter (for risk_aversion recipe)
    bottom_coding_gdppc : float, optional
        Minimum GDP per capita threshold
    zero : bool, optional
        Whether to zero historical climate
    quantreg : bool, optional
        Whether to use quantile regression
    verbose : bool, optional
        Whether to print progress

    Examples
    --------
    >>> step = ReduceDamagesStep(
    ...     config=config,
    ...     sector="coastal",
    ...     recipe="adding_up",
    ...     reduction="cc"
    ... )
    >>> inputs = {
    ...     "sector_damages_path": "data/coastal.zarr",
    ...     "socioec_path": "data/econ.zarr"
    ... }
    >>> outputs = step.run(inputs, save=True)
    """

    def __init__(
        self,
        config: DSCIMConfig,
        sector: str,
        recipe: str,
        reduction: str,
        eta: Optional[float] = None,
        bottom_coding_gdppc: float = 39.39265060424805,
        zero: bool = False,
        quantreg: bool = False,
        verbose: bool = True,
    ):
        super().__init__(config, verbose)
        self.sector = sector
        self.recipe = recipe
        self.reduction = reduction
        self.eta = eta
        self.bottom_coding_gdppc = bottom_coding_gdppc
        self.zero = zero
        self.quantreg = quantreg

        # Validate step-specific configuration
        self._validate_step_config()

    def _validate_step_config(self):
        """Validate step-specific configuration."""
        # Sector exists
        if self.sector not in self.config.sectors:
            available = list(self.config.sectors.keys())
            raise ValueError(
                f"Sector '{self.sector}' not found in configuration.\n"
                f"Available sectors: {available}"
            )

        # Validate recipe/eta combination
        if self.recipe == "adding_up" and self.eta is not None:
            raise ValueError("adding_up recipe does not accept eta parameter")
        if self.recipe == "risk_aversion" and self.eta is None:
            raise ValueError("risk_aversion recipe requires eta parameter")

    def required_inputs(self) -> List[str]:
        """Inputs needed for reduce_damages."""
        return [
            "sector_damages_path",  # Path to sector damages
            "socioec_path",  # Path to economic data
        ]

    def output_keys(self) -> List[str]:
        """Outputs produced by reduce_damages."""
        return ["reduced_damages"]

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute reduce_damages processing."""
        # Get inputs (can be paths or use defaults from config)
        sector_path = inputs.get("sector_damages_path") or self.config.sectors[
            self.sector
        ].sector_path
        socioec_path = inputs.get("socioec_path") or self.config.econdata.global_ssp

        # Create processor
        processor = DamageProcessor(
            config=self.config,
            sector=self.sector,
            recipe=self.recipe,
            reduction=self.reduction,
            socioec_path=socioec_path,
            eta=self.eta,
            bottom_coding_gdppc=self.bottom_coding_gdppc,
            zero=self.zero,
            quantreg=self.quantreg,
            use_dask=self.config.processing.use_dask,
            verbose=self.verbose,
        )

        # Process (returns dataset, doesn't save)
        result = processor.process(save=False)

        return {"reduced_damages": result}

    def _get_output_path(
        self, key: str, output_dir: Optional[str] = None
    ) -> Path:
        """Get output path for reduced damages using OutputNaming."""
        if output_dir:
            base = Path(output_dir)
        else:
            base = Path(self.config.paths.reduced_damages_library)

        # Use OutputNaming for consistency
        naming = OutputNaming(
            recipe=self.recipe,
            discount_type=self.reduction,  # "cc" or "no_cc" as discount_type
            eta=self.eta,
            sector=self.sector,
        )

        # Extension based on format
        ext_map = {"zarr": ".zarr", "netcdf": ".nc", "csv": ".csv"}
        ext = ext_map.get(self.config.processing.output_format, ".zarr")

        # Get filename
        filename = naming.get_filename("reduced_damages", ext=ext)
        return base / self.sector / filename


class GenerateDamageFunctionStep(PipelineStep):
    """
    Generate damage function pipeline step.

    Fits damage functions from reduced damages and climate data.

    Parameters
    ----------
    config : DSCIMConfig
        Validated configuration
    sector : str
        Sector name
    pulse_year : int
        Year of carbon pulse
    verbose : bool, optional
        Whether to print progress

    Examples
    --------
    >>> step = GenerateDamageFunctionStep(
    ...     config=config,
    ...     sector="mortality",
    ...     pulse_year=2020
    ... )
    >>> inputs = {"reduced_damages": damages_data}
    >>> outputs = step.run(inputs, save=True)
    """

    def __init__(
        self,
        config: DSCIMConfig,
        sector: str,
        pulse_year: int,
        verbose: bool = True,
    ):
        super().__init__(config, verbose)
        self.sector = sector
        self.pulse_year = pulse_year

        # Validate configuration
        self.config.validate_for_damage_functions(sector)

    def required_inputs(self) -> List[str]:
        """Inputs needed for damage function generation."""
        return ["reduced_damages"]

    def output_keys(self) -> List[str]:
        """Outputs produced by damage function generation."""
        return [
            "damage_function_coefficients",
            "damage_function_coefficients_grid",  # New: original dscim format
            "marginal_damages",
            "damage_function_points",
            "damage_function_stats",
        ]

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute damage function generation."""
        reduced_damages = inputs["reduced_damages"]

        # Create processor
        processor = DamageFunctionProcessor(
            config=self.config.damage_function,
            climate_config=self.config.climate_data,
            verbose=self.verbose,
        )

        # Generate damage function
        result = processor.generate_damage_function(
            damages=reduced_damages,
            sector=self.sector,
            pulse_year=self.pulse_year,
            save_outputs=False,  # We'll handle saving in the step
        )

        return {
            "damage_function_coefficients": result.coefficients,
            "damage_function_coefficients_grid": result.coefficients_grid,  # New: grid format
            "marginal_damages": result.marginal_damages,
            "damage_function_points": result.points,
            "damage_function_stats": result.fit_stats,
        }

    def _get_output_path(
        self, key: str, output_dir: Optional[str] = None
    ) -> Path:
        """Get output path for damage function outputs using OutputNaming."""
        if output_dir:
            base = Path(output_dir)
        else:
            base = Path(self.config.paths.ssp_damage_function_library)

        # Note: For damage functions, we don't have recipe/discount info yet
        # These are intermediate outputs that precede recipe-discount combinations
        # So we keep the simple naming for now
        sector_dir = base / self.sector / str(self.pulse_year)

        # Extension based on format
        ext_map = {"zarr": ".zarr", "netcdf": ".nc", "csv": ".csv"}
        ext = ext_map.get(self.config.processing.output_format, ".zarr")

        # Map output keys to filenames
        if key == "damage_function_coefficients":
            filename = f"damage_function_coefficients{ext}"
        elif key == "marginal_damages":
            filename = f"marginal_damages{ext}"
        elif key == "damage_function_points":
            filename = "damage_function_points.csv"
        elif key == "damage_function_stats":
            filename = f"damage_function_fit{ext}"
        else:
            filename = f"{key}{ext}"

        return sector_dir / filename


class CalculateSCCStep(PipelineStep):
    """
    Calculate SCC pipeline step.

    Calculates Social Cost of Carbon from marginal damages.

    Parameters
    ----------
    config : DSCIMConfig
        Validated configuration
    sector : str
        Sector name
    pulse_year : int
        Year of carbon pulse
    discount_config_index : int, optional
        Index of discount configuration to use (default: 0)
    verbose : bool, optional
        Whether to print progress

    Examples
    --------
    >>> step = CalculateSCCStep(
    ...     config=config,
    ...     sector="mortality",
    ...     pulse_year=2020
    ... )
    >>> inputs = {
    ...     "marginal_damages": md_data,
    ...     "consumption": consumption_data
    ... }
    >>> outputs = step.run(inputs, save=True)
    """

    def __init__(
        self,
        config: DSCIMConfig,
        sector: str,
        pulse_year: int,
        recipe: str = "adding_up",
        discount_config_index: int = 0,
        verbose: bool = True,
    ):
        super().__init__(config, verbose)
        self.sector = sector
        self.pulse_year = pulse_year
        self.recipe = recipe
        self.discount_config_index = discount_config_index

        # Validate configuration
        self.config.validate_for_scc(sector)

        # Get discount configuration
        discount_configs = self.config.get_discounting_configs()
        if discount_config_index >= len(discount_configs):
            raise ValueError(
                f"Discount config index {discount_config_index} out of range. "
                f"Available configs: {len(discount_configs)}"
            )
        self.discount_config = discount_configs[discount_config_index]

    def required_inputs(self) -> List[str]:
        """Inputs needed for SCC calculation."""
        required = ["marginal_damages"]
        # Consumption required for Ramsey/GWR
        if self.discount_config.discount_type in ["ramsey", "gwr"]:
            required.append("consumption")
        return required

    def output_keys(self) -> List[str]:
        """Outputs produced by SCC calculation."""
        outputs = ["scc", "discount_factors"]
        if self.config.scc.calculate_quantiles:
            outputs.append("scc_quantiles")
        if self.config.scc.save_uncollapsed:
            outputs.append("scc_uncollapsed")
        return outputs

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute SCC calculation."""
        marginal_damages = inputs["marginal_damages"]
        consumption = inputs.get("consumption")

        # Create calculator
        calculator = SCCCalculator(
            scc_config=self.config.scc,
            verbose=self.verbose,
        )

        # Calculate SCC
        result = calculator.calculate_scc(
            marginal_damages=marginal_damages,
            discount_config=self.discount_config,
            pulse_year=self.pulse_year,
            consumption=consumption,
            sector=self.sector,
            save_outputs=False,  # We'll handle saving in the step
        )

        outputs = {
            "scc": result.scc,
            "discount_factors": result.discount_factors,
        }

        if result.quantiles is not None:
            outputs["scc_quantiles"] = result.quantiles

        if result.scc_uncollapsed is not None:
            outputs["scc_uncollapsed"] = result.scc_uncollapsed

        return outputs

    def _get_output_path(
        self, key: str, output_dir: Optional[str] = None
    ) -> Path:
        """Get output path for SCC outputs using OutputNaming."""
        if output_dir:
            base = Path(output_dir)
        else:
            base = Path(self.config.paths.AR6_ssp_results)

        # Create OutputNaming instance
        naming = OutputNaming(
            recipe=self.recipe,
            discount_type=self.discount_config.discount_type,
            eta=self.discount_config.eta,
            rho=self.discount_config.rho,
            sector=self.sector,
            pulse_year=self.pulse_year,
        )

        # Extension based on format
        ext_map = {"zarr": ".zarr", "netcdf": ".nc4", "csv": ".csv"}
        ext = ext_map.get(self.config.processing.output_format, ".nc4")

        # Map output keys to output types and use OutputNaming
        output_type_map = {
            "scc": "scc",
            "discount_factors": "discount_factors",
            "scc_quantiles": "scc_quantiles",
            "scc_uncollapsed": "uncollapsed_sccs",
        }

        output_type = output_type_map.get(key, key)

        # Handle uncollapsed outputs
        collapsed = key != "scc_uncollapsed"

        # Get full path from OutputNaming
        return naming.get_output_path(
            base_dir=base,
            output_type=output_type,
            ext=ext,
            collapsed=collapsed
        )
