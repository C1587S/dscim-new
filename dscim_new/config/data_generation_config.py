"""
Hydra Configuration Schemas for DSCIM Data Generation

This module defines structured configuration schemas using dataclasses and Hydra
for generating synthetic test data and managing computation parameters.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Union
from omegaconf import MISSING


@dataclass
class TimeConfig:
    """Time-related configuration"""
    start_year: int = 2020
    end_year: int = 2030
    extrap_start_year: int = 2001
    extrap_end_year: int = 2099
    pulse_years: List[int] = field(default_factory=lambda: [2020])


@dataclass
class ClimateScenarioConfig:
    """Climate scenario configuration"""
    rcps: List[str] = field(default_factory=lambda: ['dummy1', 'dummy2'])
    gcms: List[str] = field(default_factory=lambda: ['dummy1', 'dummy2'])
    gases: List[str] = field(default_factory=lambda: ['dummy_gas'])
    slrs: List[int] = field(default_factory=lambda: [0, 1])
    n_simulations: int = 4


@dataclass
class EconomicScenarioConfig:
    """Economic scenario configuration"""
    ssps: List[str] = field(default_factory=lambda: ['ssp1', 'ssp2', 'ssp3', 'ssp4', 'ssp5'])
    regions: List[str] = field(default_factory=lambda: ['dummy1', 'dummy2'])
    models: List[str] = field(default_factory=lambda: ['dummy1', 'dummy2'])  # IAMs
    n_batches: int = 15


@dataclass
class DataRangesConfig:
    """Configuration for data value ranges"""
    temperature_range: List[float] = field(default_factory=lambda: [0.0, 10.0])
    gmsl_range: List[float] = field(default_factory=lambda: [0.0, 10.0])
    population_range: List[float] = field(default_factory=lambda: [20.0, 100.0])
    gdppc_range: List[float] = field(default_factory=lambda: [50.0, 100.0])
    delta_damage_range: List[float] = field(default_factory=lambda: [5.0, 15.0])
    histclim_damage_range: List[float] = field(default_factory=lambda: [1.0, 10.0])
    emission_conversion_factor: float = 0.1


@dataclass
class SectorConfig:
    """Configuration for individual sectors"""
    name: str = MISSING
    delta_variable: str = "delta_dummy"
    histclim_variable: str = "histclim_dummy"
    sector_path: str = MISSING
    formula: str = MISSING
    is_coastal: bool = False


@dataclass
class PathConfig:
    """File path configuration"""
    output_dir: str = "./synthetic_data"

    # Climate data paths
    gmst_path: str = "./climate/GMTanom_all_temp_2001_2010_smooth.csv"
    gmsl_path: str = "./climate/coastal_gmsl_v0.20.zarr"
    gmst_fair_path: str = "./climate/ar6_fair162_sim.nc"
    gmsl_fair_path: str = "./climate/scenario_gmsl.nc4"
    conversion_path: str = "./climate/conversion.nc4"
    ecs_mask_path: str = "./climate/parameter_filters_truncate_ECS.nc"

    # Economic data paths
    economic_path: str = "./econ/integration-econ-bc39.zarr"

    # Result paths
    reduced_damages_path: str = "./reduced_damages"
    ssp_damage_functions_path: str = "./ssp_damage_functions"
    results_path: str = "./results"


@dataclass
class DataGenerationConfig:
    """Main configuration for synthetic data generation"""
    seed: int = 42
    time: TimeConfig = field(default_factory=TimeConfig)
    climate: ClimateScenarioConfig = field(default_factory=ClimateScenarioConfig)
    economic: EconomicScenarioConfig = field(default_factory=EconomicScenarioConfig)
    data_ranges: DataRangesConfig = field(default_factory=DataRangesConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    sectors: Dict[str, SectorConfig] = field(default_factory=dict)


@dataclass
class DamageModelConfig:
    """Configuration for damage function modeling"""
    formula: str = "damages ~ -1 + anomaly + np.power(anomaly, 2)"
    fit_type: str = "ols"  # 'ols' or 'quantreg'
    quantreg_quantiles: Optional[List[float]] = None
    quantreg_weights: Optional[List[float]] = None
    extrapolation_method: str = "global_c_ratio"
    extrapolation_start_year: int = 2085
    extrapolation_end_year: int = 2099
    clip_gmsl: bool = False


@dataclass
class DiscountingConfig:
    """Configuration for discounting calculations"""
    discount_type: str = "euler_ramsey"  # constant, naive_ramsey, euler_ramsey, etc.
    discount_rates: List[float] = field(default_factory=lambda: [0.01, 0.015, 0.02, 0.025, 0.03, 0.05])
    rho: float = 0.00461878399  # Pure rate of time preference
    eta: float = 1.421158116   # Elasticity of marginal utility
    discrete_discounting: bool = False


@dataclass
class FairAggregationConfig:
    """Configuration for FAIR uncertainty aggregation"""
    aggregation_methods: List[str] = field(default_factory=lambda: ["ce", "mean", "gwr_mean", "median", "median_params"])
    fair_dims: List[str] = field(default_factory=lambda: ["simulation"])
    weitzman_parameters: List[float] = field(default_factory=lambda: [0.1, 0.5])


@dataclass
class ComputationConfig:
    """Configuration for computational parameters"""
    damage_model: DamageModelConfig = field(default_factory=DamageModelConfig)
    discounting: DiscountingConfig = field(default_factory=DiscountingConfig)
    fair_aggregation: FairAggregationConfig = field(default_factory=FairAggregationConfig)

    # Performance settings
    use_dask: bool = True
    n_workers: int = 4
    memory_limit: str = "4GB"

    # Output settings
    save_intermediate: bool = True
    compress_output: bool = True


@dataclass
class TestingConfig:
    """Configuration for testing and validation"""
    enable_regression_tests: bool = True
    regression_tolerances: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "damage_function_coefficients": {"rtol": 1e-6, "atol": 1e-10},
        "scc": {"rtol": 1e-4, "atol": 1e-8},
        "marginal_damages": {"rtol": 1e-5, "atol": 1e-9}
    })

    # Validation settings
    validate_data_structure: bool = True
    validate_mathematical_properties: bool = True

    # Performance testing
    benchmark_performance: bool = False
    performance_baseline_path: Optional[str] = None


@dataclass
class FullConfig:
    """Complete configuration combining all aspects"""
    data_generation: DataGenerationConfig = field(default_factory=DataGenerationConfig)
    computation: ComputationConfig = field(default_factory=ComputationConfig)
    testing: TestingConfig = field(default_factory=TestingConfig)

    # Metadata
    version: str = "2.0.0"
    description: str = "DSCIM Refactoring Configuration"
    author: str = "DSCIM Refactoring Team"


# Pre-defined sector configurations
COASTAL_SECTOR_CONFIG = SectorConfig(
    name="dummy_coastal_sector",
    delta_variable="delta_dummy",
    histclim_variable="histclim_dummy",
    sector_path="./sectoral/coastal_damages.zarr",
    formula="damages ~ -1 + gmsl + np.power(gmsl, 2)",
    is_coastal=True
)

NON_COASTAL_SECTOR_CONFIG = SectorConfig(
    name="dummy_not_coastl_sector",
    delta_variable="delta_dummy",
    histclim_variable="histclim_dummy",
    sector_path="./sectoral/not_coastl_damages.zarr",
    formula="damages ~ -1 + anomaly + np.power(anomaly, 2)",
    is_coastal=False
)


def create_default_config() -> FullConfig:
    """Create a default configuration with common settings"""
    config = FullConfig()

    # Add default sectors
    config.data_generation.sectors = {
        "coastal": COASTAL_SECTOR_CONFIG,
        "non_coastal": NON_COASTAL_SECTOR_CONFIG
    }

    return config


def create_testing_config() -> FullConfig:
    """Create a configuration optimized for testing (smaller datasets)"""
    config = create_default_config()

    # Reduce data size for faster testing
    config.data_generation.time.end_year = 2025
    config.data_generation.time.extrap_end_year = 2050
    config.data_generation.climate.n_simulations = 2
    config.data_generation.economic.n_batches = 5
    config.data_generation.climate.rcps = ['dummy1']
    config.data_generation.climate.gcms = ['dummy1']
    config.data_generation.economic.regions = ['dummy1']
    config.data_generation.economic.models = ['dummy1']

    # Enable all testing features
    config.testing.enable_regression_tests = True
    config.testing.validate_data_structure = True
    config.testing.validate_mathematical_properties = True

    return config


def create_production_config() -> FullConfig:
    """Create a configuration for production-like data generation"""
    config = create_default_config()

    # Full time range
    config.data_generation.time.end_year = 2100
    config.data_generation.time.extrap_end_year = 2300

    # More realistic scenarios
    config.data_generation.climate.rcps = ['rcp26', 'rcp45', 'rcp60', 'rcp85']
    config.data_generation.climate.gcms = ['ACCESS1-0', 'GFDL-ESM2M', 'HadGEM2-ES', 'IPSL-CM5A-LR']
    config.data_generation.climate.n_simulations = 100

    # Performance optimizations
    config.computation.use_dask = True
    config.computation.n_workers = 8
    config.computation.memory_limit = "8GB"

    return config