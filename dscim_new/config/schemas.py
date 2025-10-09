"""
Pydantic schemas for DSCIM configuration validation.

Provides automatic validation of configuration files and parameters.
"""

from pydantic import BaseModel, Field, validator, root_validator, model_validator
from pathlib import Path
from typing import Optional, List, Literal, Dict, Any, Union
import yaml
import xarray as xr

try:
    from rich.console import Console
    from rich.panel import Panel
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class PathsConfig(BaseModel):
    """
    Paths configuration with validation.

    Attributes
    ----------
    reduced_damages_library : str
        Output directory for reduced damages
    ssp_damage_function_library : str, optional
        Output directory for damage functions
    AR6_ssp_results : str, optional
        Output directory for SCC results
    """
    reduced_damages_library: str
    ssp_damage_function_library: Optional[str] = None
    AR6_ssp_results: Optional[str] = None
    AR5_ssp_results: Optional[str] = None

    @validator('reduced_damages_library')
    def validate_output_dir_creatable(cls, v):
        """Validate that output directory can be created."""
        path = Path(v)
        # Check if path exists or parent exists (so we can create it)
        if not path.exists():
            if path.parent.exists():
                # Parent exists, we can create this directory
                return v
            else:
                raise ValueError(
                    f"Cannot create output directory '{v}': parent directory does not exist"
                )
        return v


class SectorConfig(BaseModel):
    """
    Sector configuration with validation.

    Validates that sector data files exist and contain required variables.
    """
    sector_path: str
    histclim: str
    delta: str
    formula: str

    @validator('sector_path')
    def validate_sector_path_exists(cls, v):
        """Validate sector data file exists."""
        path = Path(v)
        if not path.exists():
            raise FileNotFoundError(
                f"Sector data file not found: {v}\n"
                f"Please ensure the data file exists or generate synthetic data."
            )
        return v

    @model_validator(mode='after')
    def validate_sector_data_variables(self):
        """Validate that sector data contains required variables."""
        sector_path = self.sector_path
        histclim = self.histclim
        delta = self.delta

        if sector_path and Path(sector_path).exists():
            try:
                # Try to open and check variables
                ds = xr.open_zarr(sector_path, chunks=None)

                missing_vars = []
                if histclim and histclim not in ds.data_vars:
                    missing_vars.append(histclim)
                if delta and delta not in ds.data_vars:
                    missing_vars.append(delta)

                if missing_vars:
                    raise ValueError(
                        f"Variables {missing_vars} not found in {sector_path}\n"
                        f"Available variables: {list(ds.data_vars)}"
                    )

                # Close dataset
                ds.close()

            except Exception as e:
                if "not found" in str(e).lower():
                    raise
                # Don't fail on other errors (e.g., missing dependencies)
                # Just warn
                import warnings
                warnings.warn(f"Could not validate sector data: {e}")

        return self


class EconDataConfig(BaseModel):
    """
    Economic data configuration with validation.

    Validates that economic data exists and contains required variables.
    """
    global_ssp: str

    @validator('global_ssp')
    def validate_econ_path_exists(cls, v):
        """Validate economic data exists."""
        path = Path(v)
        if not path.exists():
            raise FileNotFoundError(
                f"Economic data file not found: {v}\n"
                f"Please ensure the data file exists or generate synthetic data."
            )
        return v

    @validator('global_ssp')
    def validate_econ_data_variables(cls, v):
        """Validate economic data contains required variables."""
        try:
            ds = xr.open_zarr(v, chunks=None)

            required_vars = ['gdppc']  # Minimum required
            recommended_vars = ['gdppc', 'pop', 'gdp']

            missing_required = [var for var in required_vars if var not in ds.data_vars]
            missing_recommended = [var for var in recommended_vars if var not in ds.data_vars]

            if missing_required:
                raise ValueError(
                    f"Required variables {missing_required} not found in {v}\n"
                    f"Available variables: {list(ds.data_vars)}"
                )

            if missing_recommended:
                import warnings
                warnings.warn(
                    f"Recommended variables {missing_recommended} not found in {v}"
                )

            # Close dataset
            ds.close()

        except Exception as e:
            if "not found" in str(e).lower() or "required" in str(e).lower():
                raise
            # Don't fail on other errors
            import warnings
            warnings.warn(f"Could not fully validate economic data: {e}")

        return v


class ProcessingConfig(BaseModel):
    """
    Processing options configuration.

    Attributes
    ----------
    use_dask : bool
        Whether to use Dask for distributed processing
    verbose : bool
        Whether to print progress messages
    output_format : str
        Default output format for saved files
    save_intermediate : bool
        Whether to automatically save intermediate results
    """
    use_dask: bool = True
    verbose: bool = True
    output_format: Literal["zarr", "netcdf", "csv"] = "zarr"
    save_intermediate: bool = False


class ClimateDataConfig(BaseModel):
    """
    Climate data configuration with validation.

    Attributes
    ----------
    gmst_path : str, optional
        Path to GMST (temperature anomaly) CSV file
    gmsl_path : str, optional
        Path to GMSL (sea level) Zarr file
    fair_temperature_path : str, optional
        Path to FAIR temperature NetCDF file
    fair_gmsl_path : str, optional
        Path to FAIR GMSL NetCDF file
    pulse_conversion_path : str, optional
        Path to pulse conversion factors NetCDF file
    """
    gmst_path: Optional[str] = None
    gmsl_path: Optional[str] = None
    fair_temperature_path: Optional[str] = None
    fair_gmsl_path: Optional[str] = None
    pulse_conversion_path: Optional[str] = None

    @validator('gmst_path', 'gmsl_path', 'fair_temperature_path',
               'fair_gmsl_path', 'pulse_conversion_path')
    def validate_path_exists(cls, v):
        """Validate that path exists if provided."""
        if v is not None:
            path = Path(v)
            if not path.exists():
                raise FileNotFoundError(f"Climate data file not found: {v}")
        return v


class DamageFunctionConfig(BaseModel):
    """
    Damage function configuration.

    Attributes
    ----------
    formula : str
        Patsy formula for damage function (e.g., "damages ~ -1 + anomaly + np.power(anomaly, 2)")
    fit_type : str
        Type of regression to use ("ols" or "quantile")
    quantiles : list of float, optional
        Quantiles for quantile regression (e.g., [0.5])
    fit_method : str
        Fitting approach: "global" (single fit) or "rolling_window" (per year/scenario)
    window_size : int
        Size of rolling window for rolling_window method (default: 5 years)
    year_range : tuple of int
        (start_year, end_year) for rolling window fitting (default: (2020, 2101))
    extrapolation_method : str
        Method for extrapolating damages beyond projection period
    extrapolation_years : tuple of int
        (start_year, end_year) for extrapolation calculation
    save_points : bool
        Whether to save evaluation points for visualization
    n_points : int
        Number of evaluation points for visualization
    """
    formula: str = "damages ~ -1 + anomaly + np.power(anomaly, 2)"
    fit_type: Literal["ols", "quantile"] = "ols"
    quantiles: Optional[List[float]] = None
    fit_method: Literal["global", "rolling_window"] = "global"
    window_size: int = 5
    year_range: tuple = (2020, 2101)
    extrapolation_method: Literal["global_c_ratio", "constant", "linear"] = "global_c_ratio"
    extrapolation_years: tuple = (2085, 2099)
    save_points: bool = True
    n_points: int = 100

    @validator('quantiles')
    def validate_quantiles(cls, v, values):
        """Validate quantiles for quantile regression."""
        if values.get('fit_type') == 'quantile' and not v:
            raise ValueError("quantiles must be specified for quantile regression")
        if v:
            for q in v:
                if not 0 < q < 1:
                    raise ValueError(f"Quantile {q} must be between 0 and 1")
        return v


class DiscountingConfig(BaseModel):
    """
    Discounting configuration.

    Attributes
    ----------
    discount_type : str
        Type of discounting: "constant", "ramsey" variants, or "gwr" variants
        Ramsey variants: "naive_ramsey", "euler_ramsey"
        GWR variants: "naive_gwr", "euler_gwr", "gwr_gwr"
    discount_rate : float, optional
        Discount rate for constant discounting (e.g., 0.02 for 2%)
    eta : float, optional
        Elasticity of marginal utility (for Ramsey/GWR)
    rho : float, optional
        Pure rate of time preference (for Ramsey/GWR)
    discrete : bool
        Whether to use discrete discounting (vs continuous)
    gwr_method : str
        GWR method (deprecated - use discount_type instead)
    """
    discount_type: Literal[
        "constant", "constant_model_collapsed", "constant_gwr",
        "ramsey", "naive_ramsey", "euler_ramsey",
        "gwr", "naive_gwr", "euler_gwr", "gwr_gwr"
    ]
    discount_rate: Optional[float] = None
    eta: Optional[float] = None
    rho: Optional[float] = None
    discrete: bool = False
    gwr_method: Optional[Literal["naive_gwr", "gwr_gwr", "euler_gwr"]] = None

    @model_validator(mode='after')
    def validate_discount_params(self):
        """Validate that required parameters are provided for discount type."""
        if self.discount_type in ['constant', 'constant_model_collapsed', 'constant_gwr']:
            if self.discount_rate is None:
                # For constant discounting, discount_rate is optional if using default
                pass  # Allow None, will use defaults

        elif 'ramsey' in self.discount_type or 'gwr' in self.discount_type:
            # All Ramsey and GWR variants need eta and rho
            if self.eta is None:
                raise ValueError(f"eta required for {self.discount_type} discounting")
            if self.rho is None:
                raise ValueError(f"rho required for {self.discount_type} discounting")

        return self


class SCCConfig(BaseModel):
    """
    Social Cost of Carbon calculation configuration.

    Attributes
    ----------
    pulse_years : list of int
        Years for carbon pulse (e.g., [2020])
    fair_aggregation : str
        How to aggregate over FAIR simulations: "mean", "median", "ce"
    calculate_quantiles : bool
        Whether to calculate uncertainty quantiles
    quantile_levels : list of float
        Quantile levels to calculate (e.g., [0.05, 0.5, 0.95])
    save_discount_factors : bool
        Whether to save discount factors separately
    save_uncollapsed : bool
        Whether to save uncollapsed SCC (full distribution)
    """
    pulse_years: List[int] = [2020]
    fair_aggregation: Literal["mean", "median", "ce"] = "mean"
    calculate_quantiles: bool = True
    quantile_levels: List[float] = [0.05, 0.25, 0.5, 0.75, 0.95]
    save_discount_factors: bool = True
    save_uncollapsed: bool = False

    @validator('quantile_levels', each_item=True)
    def validate_quantile_level(cls, v):
        """Validate quantile levels are between 0 and 1."""
        if not 0 < v < 1:
            raise ValueError(f"Quantile level {v} must be between 0 and 1")
        return v


class PipelineConfig(BaseModel):
    """
    Pipeline execution configuration.

    Defines which sectors, recipes, and scenarios to process.
    """
    sectors_to_process: Optional[List[str]] = None
    recipes: List[str] = ["adding_up", "risk_aversion"]
    reductions: List[str] = ["cc", "no_cc"]
    eta_values: List[float] = [2.0]

    @validator('recipes', each_item=True)
    def validate_recipe(cls, v):
        """Validate recipe names."""
        valid_recipes = ["adding_up", "risk_aversion", "equity"]
        if v not in valid_recipes:
            raise ValueError(f"Invalid recipe '{v}'. Must be one of {valid_recipes}")
        return v

    @validator('reductions', each_item=True)
    def validate_reduction(cls, v):
        """Validate reduction types."""
        valid_reductions = ["cc", "no_cc"]
        if v not in valid_reductions:
            raise ValueError(f"Invalid reduction '{v}'. Must be one of {valid_reductions}")
        return v


class DSCIMConfig(BaseModel):
    """
    Complete DSCIM configuration with full validation.

    This is the main configuration class that validates all inputs
    when loaded from YAML or dictionary.

    Examples
    --------
    >>> # From YAML file
    >>> config = DSCIMConfig.from_yaml("config.yaml")

    >>> # From dictionary
    >>> config = DSCIMConfig(**config_dict)

    >>> # Access validated fields
    >>> print(config.paths.reduced_damages_library)
    >>> print(config.sectors["coastal"].sector_path)
    """
    paths: PathsConfig
    econdata: EconDataConfig
    sectors: Dict[str, SectorConfig]
    processing: ProcessingConfig = ProcessingConfig()
    pipeline: Optional[PipelineConfig] = None

    # New configs for damage functions and SCC
    climate_data: Optional[ClimateDataConfig] = None
    damage_function: Optional[DamageFunctionConfig] = DamageFunctionConfig()
    discounting: Optional[List[DiscountingConfig]] = None
    scc: Optional[SCCConfig] = SCCConfig()

    # Optional fields from original config
    mortality_version: Optional[int] = None
    coastal_version: Optional[str] = None
    AR6_ssp_climate: Optional[Dict[str, Any]] = None
    global_parameters: Optional[Dict[str, Any]] = None

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"  # Allow extra fields for forward compatibility

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "DSCIMConfig":
        """
        Load and validate configuration from YAML file.

        Parameters
        ----------
        yaml_path : str
            Path to YAML configuration file

        Returns
        -------
        DSCIMConfig
            Validated configuration object

        Raises
        ------
        FileNotFoundError
            If YAML file doesn't exist
        ValidationError
            If configuration is invalid

        Examples
        --------
        >>> config = DSCIMConfig.from_yaml("config.yaml")
        >>> # Automatically validates all fields
        """
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        return cls(**data)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "DSCIMConfig":
        """
        Load and validate configuration from dictionary.

        Parameters
        ----------
        config_dict : dict
            Configuration dictionary

        Returns
        -------
        DSCIMConfig
            Validated configuration object
        """
        return cls(**config_dict)

    def get_sector_config(self, sector: str) -> SectorConfig:
        """
        Get configuration for a specific sector.

        Parameters
        ----------
        sector : str
            Sector name

        Returns
        -------
        SectorConfig
            Sector configuration

        Raises
        ------
        KeyError
            If sector not found in configuration
        """
        if sector not in self.sectors:
            available = list(self.sectors.keys())
            raise KeyError(
                f"Sector '{sector}' not found in configuration.\n"
                f"Available sectors: {available}"
            )
        return self.sectors[sector]

    def validate_for_reduce_damages(self, sector: str):
        """
        Validate that configuration has everything needed for reduce_damages.

        Parameters
        ----------
        sector : str
            Sector to validate

        Raises
        ------
        ValueError
            If configuration is missing required fields
        """
        # Check sector exists
        self.get_sector_config(sector)

        # Check economic data is configured
        if not self.econdata.global_ssp:
            raise ValueError("Economic data path not configured (econdata.global_ssp)")

        # Check output path is configured
        if not self.paths.reduced_damages_library:
            raise ValueError("Output path not configured (paths.reduced_damages_library)")

    def get_pipeline_sectors(self) -> List[str]:
        """
        Get list of sectors to process in pipeline.

        Returns
        -------
        list
            Sector names to process
        """
        if self.pipeline and self.pipeline.sectors_to_process:
            return self.pipeline.sectors_to_process
        else:
            return list(self.sectors.keys())

    def validate_for_damage_functions(self, sector: str):
        """
        Validate that configuration has everything needed for damage function generation.

        Parameters
        ----------
        sector : str
            Sector to validate

        Raises
        ------
        ValueError
            If configuration is missing required fields
        """
        # Check reduce_damages requirements first
        self.validate_for_reduce_damages(sector)

        # Check climate data is configured
        if not self.climate_data:
            raise ValueError("Climate data not configured (climate_data section)")

        # Check damage function config
        if not self.damage_function:
            raise ValueError("Damage function configuration not found")

        # Check output path for damage functions
        if not self.paths.ssp_damage_function_library:
            raise ValueError("Damage function output path not configured (paths.ssp_damage_function_library)")

    def validate_for_scc(self, sector: str):
        """
        Validate that configuration has everything needed for SCC calculation.

        Parameters
        ----------
        sector : str
            Sector to validate

        Raises
        ------
        ValueError
            If configuration is missing required fields
        """
        # Check damage function requirements first
        self.validate_for_damage_functions(sector)

        # Check discounting is configured
        if not self.discounting:
            raise ValueError("Discounting configuration not found (discounting section)")

        # Check SCC config
        if not self.scc:
            raise ValueError("SCC configuration not found")

        # Check output path for SCC results
        if not self.paths.AR6_ssp_results:
            raise ValueError("SCC output path not configured (paths.AR6_ssp_results)")

    def get_discounting_configs(self) -> List[DiscountingConfig]:
        """
        Get list of discounting configurations to process.

        Returns
        -------
        list
            Discounting configurations
        """
        if self.discounting:
            return self.discounting
        else:
            # Default: constant discounting at 2%
            return [DiscountingConfig(discount_type="constant", discount_rate=0.02)]
