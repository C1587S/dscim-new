# Quick Start

## Complete Pipeline Example

Run the full DSCIM workflow with synthetic data:

```python
from dscim_new.pipeline import DSCIMPipeline

# Load configuration
pipeline = DSCIMPipeline("examples/configs/full_config.yaml")

# Run complete pipeline
results = pipeline.run_full_pipeline(
    sectors=["mortality"],
    recipes=["adding_up", "risk_aversion"],
    discount_types=["constant", "ramsey"],
    save=True
)

# Access results
scc = results["sccs"]["mortality"]["adding_up"]["constant"]["scc"]
```

## Step-by-Step Execution

Execute individual pipeline stages:

```python
from dscim_new.config.schemas import DSCIMConfig
from dscim_new.pipeline import (
    ReduceDamagesStep,
    GenerateDamageFunctionStep,
    CalculateSCCStep,
)

config = DSCIMConfig.from_yaml("config.yaml")

# Step 1: Reduce damages
reduce_step = ReduceDamagesStep(
    config=config,
    sector="mortality",
    recipe="adding_up",
    reduction="cc",
)
reduced = reduce_step.run(inputs={}, save=True)

# Step 2: Generate damage function
df_step = GenerateDamageFunctionStep(
    config=config,
    sector="mortality",
    pulse_year=2020,
)
df_outputs = df_step.run(
    inputs={"reduced_damages": reduced["reduced_damages"]},
    save=True
)

# Step 3: Calculate SCC
scc_step = CalculateSCCStep(
    config=config,
    sector="mortality",
    pulse_year=2020,
    discount_config_index=0,
)
scc_outputs = scc_step.run(
    inputs={
        "marginal_damages": df_outputs["marginal_damages"],
        "consumption": consumption_data,
    },
    save=True
)
```

## Using Synthetic Data

Generate test data for development:

```python
from pathlib import Path
from dscim_new.utils import ClimateDataGenerator, DamagesDataGenerator

output_dir = Path("test_data")

# Generate climate data
climate_gen = ClimateDataGenerator()
climate_paths = climate_gen.generate_all(output_dir / "climate")

# Generate damages and economic data
damages_gen = DamagesDataGenerator()
damages_paths = damages_gen.generate_all(output_dir / "damages")

# Update configuration to point to generated data
config = DSCIMConfig.from_yaml("config.yaml")
config.climate_data.gmst_path = str(climate_paths["gmst"])
config.sectoral_config["mortality"].path = str(damages_paths["not_coastal"])
```

## Configuration from Python

Create configuration programmatically:

```python
from dscim_new.config.schemas import (
    DSCIMConfig,
    DamageFunctionConfig,
    DiscountingConfig,
)

config = DSCIMConfig(
    climate_data=ClimateDataConfig(
        gmst_path="data/climate/gmst.nc4",
        gmsl_path="data/climate/gmsl.nc4",
        fair_temperature_path="data/climate/fair_temp.zarr",
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

## Running Examples

Several example scripts are provided:

```bash
# Complete pipeline with synthetic data
cd examples/scripts
python full_pipeline_example.py --verbose

# Simple standalone example
python simple_run_ssps.py

# Jupyter notebook tutorial
jupyter lab ../notebooks/full_sccs.ipynb
```

## Output Structure

Results are saved in organized directories:

```
output_directory/
├── reduced_damages/
│   └── {sector}_{recipe}_{reduction}.zarr
├── damage_functions/
│   ├── {sector}_{year}_coefficients.zarr
│   └── {sector}_{year}_marginal_damages.zarr
└── scc_results/
    └── {sector}_{year}_{discount}_scc.zarr
```

## Next Steps

- [Configure pipeline parameters](../user-guide/configuration.md)
- [Understand pipeline architecture](../user-guide/architecture.md)
- [Review complete workflow example](../examples/complete-workflow.md)
- [Explore API reference](../api/pipeline.md)
