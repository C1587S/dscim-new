# Complete Workflow Example

This example demonstrates the full DSCIM pipeline from data preparation to SCC calculation.

## Script Overview

The complete workflow script executes:

1. Synthetic data generation (climate, damages, economic)
2. Damage reduction with climate scenarios
3. Damage function estimation via OLS regression
4. SCC calculation with multiple discount methods

Examples are included in the `examples/` folder.

## Running the Example

```bash
cd examples/scripts
python full_pipeline_example.py --verbose --output-dir ../../test_output
```

Options:
- `--verbose`: Enable detailed progress output
- `--output-dir PATH`: Specify output directory
- `--sector {not_coastal,coastal}`: Choose sector
- `--pulse-year YEAR`: SCC pulse year (default: 2020)

**Resource Management**: **(EXPLAIN THIS BETTER)**

## Step-by-Step Walkthrough

### 1. Setup and Configuration

```python
from pathlib import Path
from dscim_new.config.schemas import DSCIMConfig
from dscim_new.utils import ClimateDataGenerator, DamagesDataGenerator

# Create output directories
output_dir = Path("test_output")
dirs = {
    "climate": output_dir / "climate_data",
    "damages": output_dir / "damages_data",
    "reduced": output_dir / "reduced_damages",
    "df": output_dir / "damage_functions",
    "scc": output_dir / "scc_results",
}

for dir_path in dirs.values():
    dir_path.mkdir(parents=True, exist_ok=True)
```

### 2. Generate Synthetic Data

```python
# Generate climate data
climate_gen = ClimateDataGenerator()
climate_paths = climate_gen.generate_all(dirs["climate"])

# Climate data includes:
# - GMST matching data
# - GMSL matching data
# - FAIR temperature projections
# - FAIR GMSL projections
# - Pulse conversion factors

# Generate damages and economic data
damages_gen = DamagesDataGenerator()
damages_paths = damages_gen.generate_all(output_dir / "damages_data")

# Damages data includes:
# - Sectoral damages (delta and histclim)
# - Economic variables (GDP, population, consumption)
```

### 3. Create Configuration

**NEED TO INCLUDE A MORE SPECIFIC EXAMPLE**

Using `DSCIMConfig` with a config file is recommended, but users can also modify settings on the fly programmatically.

```python
from dscim_new.config.schemas import (
    DSCIMConfig,
    PathsConfig,
    ClimateDataConfig,
    SectorConfig,
    EconDataConfig,
    DamageFunctionConfig,
    DiscountingConfig,
)

config = DSCIMConfig(
    paths=PathsConfig(
        reduced_damages_library=str(dirs["reduced"]),
        ssp_damage_function_library=str(dirs["df"]),
        AR6_ssp_results=str(dirs["scc"]),
    ),
    climate_data=ClimateDataConfig(
        gmst_path=str(climate_paths["gmst"]),
        gmsl_path=str(climate_paths["gmsl"]),
        fair_temperature_path=str(climate_paths["fair_temp"]),
        fair_gmsl_path=str(climate_paths["fair_gmsl"]),
    ),
    sectoral_config={
        "not_coastal": SectorConfig(
            sector_path=str(damages_paths["not_coastal"]),
            histclim="histclim",
            delta="delta",
            formula="~ -1 + anomaly + np.power(anomaly, 2)",
        )
    },
    econdata=EconDataConfig(
        global_ssp=str(damages_paths["econ"])
    ),
    damage_function_config=DamageFunctionConfig(
        formula="damages ~ -1 + anomaly + np.power(anomaly, 2)",
        fit_type="ols",
    ),
    discounting_configs=[
        DiscountingConfig(discount_type="constant", discount_rate=0.02),
        DiscountingConfig(discount_type="ramsey", eta=1.45, rho=0.001),
    ],
)
```

### 4. Execute Pipeline Steps

#### Step 1: Reduce Damages

```python
from dscim_new.pipeline import ReduceDamagesStep

reduce_step = ReduceDamagesStep(
    config=config,
    sector="not_coastal",
    recipe="adding_up",
    reduction="cc",
    verbose=True,
)

reduced_outputs = reduce_step.run(inputs={}, save=True)
# Output: reduced_damages.zarr
```

When using `save=True`, intermediate inputs are saved to the configured output directories. This includes reduced damages files (saved as .zarr format in the `reduced_damages_library` path), which can later be loaded to restart the pipeline from this step. Dependencies for each step must be included in the inputs dictionary when restarting from intermediate steps.

This step:
- Loads sectoral damage projections
- Loads climate matching data (GMST)
- Interpolates damages to climate scenarios
- Aggregates by recipe (adding_up, risk_aversion, equity)
- Saves reduced damages

#### Step 2: Generate Damage Function

```python
from dscim_new.pipeline import GenerateDamageFunctionStep

df_step = GenerateDamageFunctionStep(
    config=config,
    sector="not_coastal",
    pulse_year=2020,
    verbose=True,
)

df_outputs = df_step.run(
    inputs={"reduced_damages": reduced_outputs["reduced_damages"]},
    save=True,
)
# Outputs:
# - coefficients.zarr
# - marginal_damages.zarr
```

This step:
- Fits OLS regression: `damages ~ anomaly + anomaly^2`
- Extrapolates to FAIR temperature scenarios
- Calculates marginal damages (derivative of damage function)
- Saves coefficients and marginal damages

#### Step 3: Calculate SCC

```python
from dscim_new.pipeline import CalculateSCCStep
import xarray as xr

# Load consumption data
consumption = xr.open_zarr(damages_paths["econ"])["gdppc"]

scc_step = CalculateSCCStep(
    config=config,
    sector="not_coastal",
    pulse_year=2020,
    discount_config_index=0,  # First discount config (constant)
    verbose=True,
)

scc_outputs = scc_step.run(
    inputs={
        "marginal_damages": df_outputs["marginal_damages"],
        "consumption": consumption,
    },
    save=True,
)
# Outputs:
# - scc.zarr
# - discount_factors.zarr
```

This step:
- Calculates discount factors using specified method
- Integrates discounted marginal damages
- Computes SCC across all uncertainty dimensions
- Saves results

### 5. Access Results

```python
import xarray as xr

# Load SCC results
scc = xr.open_zarr(scc_outputs["scc_path"])

# Calculate summary statistics
mean_scc = float(scc.mean().values)
median_scc = float(scc.median().values)
p05 = float(scc.quantile(0.05).values)
p95 = float(scc.quantile(0.95).values)

print(f"Mean SCC: ${mean_scc:.2f}/tCO2")
print(f"Median SCC: ${median_scc:.2f}/tCO2")
print(f"90% CI: [${p05:.2f}, ${p95:.2f}]")
```

## Output Files

After running the complete workflow, the output directory contains:

```
test_output/
├── climate_data/
│   ├── gmst_matching.nc4
│   ├── gmsl_matching.nc4
│   ├── fair_temperature.zarr/
│   ├── fair_gmsl.zarr/
│   └── fair_conversion_factors.zarr/
├── damages_data/
│   ├── sectoral/
│   │   ├── not_coastal.zarr/
│   │   └── coastal.zarr/
│   └── econ/
│       └── global_ssp.zarr/
├── reduced_damages/
│   └── not_coastal_adding_up_cc.zarr/
├── damage_functions/
│   ├── not_coastal_2020_coefficients.zarr/
│   └── not_coastal_2020_marginal_damages.zarr/
└── scc_results/
    ├── not_coastal_2020_constant_scc.zarr/
    └── not_coastal_2020_ramsey_scc.zarr/
```

## Multiple Recipes and Discount Methods

To run with multiple configurations:

```python
recipes = ["adding_up", "risk_aversion"]
discount_configs = [
    DiscountingConfig(discount_type="constant", discount_rate=0.02),
    DiscountingConfig(discount_type="ramsey", eta=1.45, rho=0.001),
    DiscountingConfig(discount_type="ramsey", eta=2.0, rho=0.001),
]

for recipe in recipes:
    # Step 1: Reduce damages
    reduce_step = ReduceDamagesStep(
        config=config, sector="not_coastal", recipe=recipe, reduction="cc"
    )
    reduced = reduce_step.run(inputs={}, save=True)

    # Step 2: Generate damage function
    df_step = GenerateDamageFunctionStep(
        config=config, sector="not_coastal", pulse_year=2020
    )
    df_outputs = df_step.run(
        inputs={"reduced_damages": reduced["reduced_damages"]}, save=True
    )

    # Step 3: Calculate SCC for each discount method
    for i, disc_config in enumerate(discount_configs):
        scc_step = CalculateSCCStep(
            config=config,
            sector="not_coastal",
            pulse_year=2020,
            discount_config_index=i,
        )
        scc_outputs = scc_step.run(
            inputs={
                "marginal_damages": df_outputs["marginal_damages"],
                "consumption": consumption,
            },
            save=True,
        )
```

## Comparison with Original Implementation

The workflow replicates the original `dscim-testing/run_integration_result.py` script with equivalent outputs:

| Original | DSCIM-New | Status |
|----------|-----------|--------|
| `reduce_damages()` | `ReduceDamagesStep` | Equivalent |
| `run_ssps()` | `GenerateDamageFunctionStep` + `CalculateSCCStep` | Equivalent |
| Output files | Zarr format | Same structure |

Key differences:
- Modular steps vs less modular method/functions
- Explicit inputs/outputs vs implicit data flow
- Type-safe configuration vs dictionary parameters
- Validation at each step vs runtime errors

Results are numerically identical within floating-point tolerance.

## Next Steps

- [Review step-by-step tutorial](step-by-step.md)
- [Explore custom configurations](custom-configs.md)
- [Understand pipeline architecture](../user-guide/architecture.md)
- [See API reference](../api/pipeline.md)
