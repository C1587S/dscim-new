# Configuration

## Overview

DSCIM-New uses Pydantic schemas for type-safe configuration validation. Configuration can be loaded from YAML files or created programmatically.

## Loading Configuration

### From YAML File

```python
from dscim_new.config.schemas import DSCIMConfig

config = DSCIMConfig.from_yaml("config.yaml")
```

### Programmatic Creation

**NEED TO INCLUDE A MORE SPECIFIC EXAMPLE**

```python
from dscim_new.config.schemas import (
    DSCIMConfig,
    ClimateDataConfig,
    DamageFunctionConfig,
    DiscountingConfig,
)

config = DSCIMConfig(
    climate_data=ClimateDataConfig(...),
    damage_function_config=DamageFunctionConfig(...),
    discounting_configs=[DiscountingConfig(...)]
)
```

## Configuration Schema

### Top-Level Configuration

`DSCIMConfig` is the main configuration object:

```yaml
paths:
  reduced_damages_library: "output/reduced_damages"
  ssp_damage_function_library: "output/damage_functions"
  AR6_ssp_results: "output/scc_results"

climate_data:
  gmst_path: "data/climate/gmst.nc4"
  gmsl_path: "data/climate/gmsl.nc4"
  fair_temperature_path: "data/climate/fair_temp.zarr"
  fair_gmsl_path: "data/climate/fair_gmsl.zarr"

econdata:
  global_ssp: "data/economic/global_ssp.zarr"

sectoral_config:
  mortality:
    sector_path: "data/damages/mortality.zarr"
    histclim: "histclim"
    delta: "delta"
    formula: "~ -1 + anomaly + np.power(anomaly, 2)"

damage_function_config:
  formula: "damages ~ -1 + anomaly + np.power(anomaly, 2)"
  fit_type: "ols"
  quantreg_quantiles: null
  subset_dict: null
  fair_dims: ["simulation"]

discounting_configs:
  - discount_type: "constant"
    discount_rate: 0.02
  - discount_type: "ramsey"
    eta: 1.45
    rho: 0.001

scc_config:
  ce_horizon_years: 300
  damage_function_fit_years: [2020, 2030, 2040, 2050, 2060, 2070, 2080, 2090, 2100]

processing:
  save_intermediate: true
  verbose: false
```

## Configuration Components

### Paths Configuration

Specifies output directories for intermediate and final results:

```python
class PathsConfig(BaseModel):
    reduced_damages_library: str  # Required
    ssp_damage_function_library: Optional[str] = None
    AR6_ssp_results: Optional[str] = None
```

Validation:
- Checks that parent directories exist
- Creates output directories if needed

### Climate Data Configuration

Paths to climate input data:

```python
class ClimateDataConfig(BaseModel):
    gmst_path: str  # GMST matching data
    gmsl_path: Optional[str] = None  # GMSL matching data (for coastal)
    fair_temperature_path: Optional[str] = None  # FAIR temperature projections
    fair_gmsl_path: Optional[str] = None  # FAIR GMSL projections
```

Validation:
- Verifies files exist
- Checks for required variables in datasets

### Sector Configuration

Sector-specific settings:

```python
class SectorConfig(BaseModel):
    sector_path: str  # Path to sector damages
    histclim: str  # Historical climate variable name
    delta: str  # Delta damages variable name
    formula: str  # Damage function formula
```

Example:
```yaml
sectoral_config:
  mortality:
    sector_path: "data/damages/mortality.zarr"
    histclim: "histclim"
    delta: "delta"
    formula: "~ -1 + anomaly + np.power(anomaly, 2)"
  coastal:
    sector_path: "data/damages/coastal.zarr"
    histclim: "histclim_gmsl"
    delta: "delta"
    formula: "~ -1 + anomaly + np.power(anomaly, 2)"
```

### Economic Data Configuration

Economic variables configuration:

```python
class EconDataConfig(BaseModel):
    global_ssp: str  # Path to economic data (GDP, population, consumption)
```

Required variables in dataset:
- `gdppc`: GDP per capita (required)
- `pop`: Population (recommended)
- `gdp`: GDP (recommended)

### Damage Function Configuration

Regression specification:

```python
class DamageFunctionConfig(BaseModel):
    formula: str = "damages ~ -1 + anomaly + np.power(anomaly, 2)"
    fit_type: Literal["ols", "quantreg"] = "ols"
    quantreg_quantiles: Optional[List[float]] = None
    subset_dict: Optional[Dict[str, Any]] = None
    fair_dims: List[str] = ["simulation"]
    save_files: bool = True
```

Formula syntax follows `statsmodels.formula.api`:
- `damages ~ -1 + anomaly`: Linear (no intercept)
- `damages ~ -1 + anomaly + np.power(anomaly, 2)`: Quadratic

Fit types:
- `ols`: Ordinary least squares
- `quantreg`: Quantile regression (requires `quantreg_quantiles`)

### Discounting Configuration

Discount method specifications:

```python
class DiscountingConfig(BaseModel):
    discount_type: Literal["constant", "ramsey", "gwr"]
    discount_rate: Optional[float] = None  # For constant
    eta: Optional[float] = None  # For ramsey/gwr
    rho: Optional[float] = None  # For ramsey/gwr
    gwr_method: Optional[Literal["naive_gwr", "euler_gwr"]] = "naive_gwr"
```

Examples:

**Constant discounting**:
```yaml
- discount_type: "constant"
  discount_rate: 0.02  # 2% discount rate
```

**Ramsey discounting**:
```yaml
- discount_type: "ramsey"
  eta: 1.45  # Elasticity of marginal utility
  rho: 0.001  # Pure rate of time preference
```

**GWR discounting**:
```yaml
- discount_type: "gwr"
  eta: 1.45
  rho: 0.001
  gwr_method: "naive_gwr"  # or "euler_gwr"
```

### SCC Configuration

SCC calculation parameters:

```python
class SCCConfig(BaseModel):
    ce_horizon_years: int = 300  # Certainty equivalent horizon
    damage_function_fit_years: List[int] = [2020, 2030, ..., 2100]
```

### Processing Configuration

Runtime behavior settings:

```python
class ProcessingConfig(BaseModel):
    save_intermediate: bool = True  # Save intermediate outputs
    verbose: bool = False  # Enable verbose logging
    n_workers: Optional[int] = None  # Dask workers
    threads_per_worker: Optional[int] = None  # Threads per worker
```

## Configuration Templates

### Minimal Configuration

```yaml
paths:
  reduced_damages_library: "output/reduced_damages"

climate_data:
  gmst_path: "data/climate/gmst.nc4"

econdata:
  global_ssp: "data/econ/global_ssp.zarr"

sectoral_config:
  mortality:
    sector_path: "data/damages/mortality.zarr"
    histclim: "histclim"
    delta: "delta"
    formula: "~ -1 + anomaly + np.power(anomaly, 2)"

discounting_configs:
  - discount_type: "constant"
    discount_rate: 0.02
```

### Full Configuration

See `examples/configs/full_config.yaml` for a complete example with all options.

## Validation

Configuration is validated automatically on load:

```python
config = DSCIMConfig.from_yaml("config.yaml")
# Raises ValidationError if:
# - Required fields missing
# - Files don't exist
# - Types incorrect
# - Values out of range
```

**Note**: Dependencies for each step must be included in the configuration to enable restarting from intermediate steps.

Common validation errors:

**Missing required fields**:
```
ValidationError: field required (type=value_error.missing)
```

**File not found**:
```
FileNotFoundError: Climate data file not found: data/climate/gmst.nc4
```

**Invalid type**:
```
ValidationError: value is not a valid float (type=type_error.float)
```

## Accessing Configuration

Configuration values are accessed via attributes:

```python
config = DSCIMConfig.from_yaml("config.yaml")

# Access paths
output_dir = config.paths.reduced_damages_library

# Access climate data paths
gmst_path = config.climate_data.gmst_path

# Access sector config
mortality_config = config.sectoral_config["mortality"]
sector_path = mortality_config.sector_path

# Access damage function config
formula = config.damage_function_config.formula

# Access discounting configs (list)
first_discount = config.discounting_configs[0]
discount_type = first_discount.discount_type
```

## Environment-Specific Configuration

Use different configurations for different environments:

```bash
# Development with synthetic data
dscim-pipeline --config configs/dev_config.yaml

# Production with real data
dscim-pipeline --config configs/prod_config.yaml

# Testing with minimal data
dscim-pipeline --config configs/test_config.yaml
```

## Configuration Inheritance

Create base configurations and override specific values:

```python
# Load base configuration
base_config = DSCIMConfig.from_yaml("base_config.yaml")

# Override specific values
base_config.processing.verbose = True
base_config.discounting_configs = [
    DiscountingConfig(discount_type="constant", discount_rate=0.03)
]
```

## Next Steps

- [Understand pipeline steps](pipeline-steps.md)
- [Review data requirements](data-requirements.md)
- [See configuration examples](../examples/custom-configs.md)
