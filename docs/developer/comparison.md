# Comparison with Original DSCIM

This page describes the structural differences between the original DSCIM and DSCIM-New, with emphasis on modularity and workflow flexibility.

## Computational Equivalence

Both implementations produce numerically identical results:

- Same damage function formulas
- Same regression methods (`statsmodels.formula.api.ols`)
- Same discounting calculations (constant, Ramsey, GWR)
- Same aggregation recipes (adding_up, risk_aversion, equity)

Results match within floating-point tolerance (typically `rtol=1e-6`).

## Key Difference: Modularity and Flexibility

The primary distinction between implementations is how they handle workflow execution and intermediate results.

### Modular Step Architecture

DSCIM-New separates the complete workflow into distinct, reusable steps:

**Three Core Steps:**
1. **ReduceDamagesStep**: Process sectoral damages with climate data
2. **GenerateDamageFunctionStep**: Fit regression models
3. **CalculateSCCStep**: Compute discounted SCC values

Each step can be:
- Run independently
- Executed in any order (with correct inputs)
- Configured separately
- Tested in isolation

### Workflow Flexibility

#### Option 1: Run Complete Pipeline

Users who want the full workflow can run everything at once:

```python
from dscim_new.pipeline import DSCIMPipeline

pipeline = DSCIMPipeline("config.yaml")

# Run all steps sequentially
results = pipeline.run_full_pipeline(
    sectors=["mortality"],
    recipes=["adding_up", "risk_aversion"],
    discount_types=["constant", "ramsey"],
    save=True  # Saves all intermediate outputs
)

# Access final results
scc = results["sccs"]["mortality"]["adding_up"]["constant"]["scc"]
```

**Benefits:**
- Simple interface for standard workflows
- Automatic coordination between steps
- All outputs saved to configured paths

#### Option 2: Run Individual Steps

Users can run steps separately for more control:

```python
from dscim_new.pipeline import (
    ReduceDamagesStep,
    GenerateDamageFunctionStep,
    CalculateSCCStep
)
from dscim_new.config.schemas import DSCIMConfig

config = DSCIMConfig.from_yaml("config.yaml")

# Step 1: Reduce damages
reduce_step = ReduceDamagesStep(
    config=config,
    sector="mortality",
    recipe="adding_up",
    reduction="cc"
)
reduced = reduce_step.run(inputs={}, save=True)
# Output saved to: reduced_damages/mortality_adding_up_cc.zarr

# Step 2: Generate damage function (can run days later)
df_step = GenerateDamageFunctionStep(
    config=config,
    sector="mortality",
    pulse_year=2020
)
df_outputs = df_step.run(
    inputs={"reduced_damages": reduced["reduced_damages"]},
    save=True
)
# Outputs saved to damage_functions/

# Step 3: Calculate SCC with different discount methods
for discount_idx in range(len(config.discounting_configs)):
    scc_step = CalculateSCCStep(
        config=config,
        sector="mortality",
        pulse_year=2020,
        discount_config_index=discount_idx
    )
    scc_outputs = scc_step.run(
        inputs={
            "marginal_damages": df_outputs["marginal_damages"],
            "consumption": consumption_data
        },
        save=True
    )
    # Each discount method saved separately
```

**Benefits:**
- Pause and resume workflow at any step
- Modify intermediate results before next step
- Re-run specific steps without recomputing everything
- Test different configurations at each stage

#### Option 3: Reuse Intermediate Results

Users can load previously saved results and continue from any point:

```python
import xarray as xr
from dscim_new.pipeline import CalculateSCCStep

# Load previously computed marginal damages
marginal_damages = xr.open_zarr(
    "damage_functions/mortality_2020_marginal_damages.zarr"
)

# Load consumption data
consumption = xr.open_zarr("data/economic/global_ssp.zarr")["gdppc"]

# Calculate SCC with new discount configuration
# (without re-running damage reduction or function fitting)
scc_step = CalculateSCCStep(
    config=new_config,  # Different discount parameters
    sector="mortality",
    pulse_year=2020,
    discount_config_index=0
)

scc_outputs = scc_step.run(
    inputs={
        "marginal_damages": marginal_damages,
        "consumption": consumption
    },
    save=True
)
```

**Benefits:**
- Avoid recomputing expensive steps
- Experiment with different parameters at specific stages
- Share intermediate results between analyses
- Reduce computational time for sensitivity analyses

#### Option 4: Modify and Extend

Users can modify outputs between steps or create custom workflows:

```python
# Run damage reduction
reduce_step = ReduceDamagesStep(config, sector="mortality", recipe="adding_up")
reduced = reduce_step.run(inputs={}, save=False)

# Modify reduced damages (e.g., apply custom adjustment)
modified_damages = reduced["reduced_damages"] * adjustment_factor

# Continue pipeline with modified data
df_step = GenerateDamageFunctionStep(config, sector="mortality", pulse_year=2020)
df_outputs = df_step.run(
    inputs={"reduced_damages": modified_damages},
    save=True
)

# Or apply custom processing
custom_marginal_damages = my_custom_function(df_outputs["marginal_damages"])

# Use in final step
scc_step = CalculateSCCStep(config, sector="mortality", pulse_year=2020)
scc_outputs = scc_step.run(
    inputs={
        "marginal_damages": custom_marginal_damages,
        "consumption": consumption
    },
    save=True
)
```

**Benefits:**
- Insert custom processing between steps
- Implement experimental methods
- Validate intermediate transformations
- Debug step-by-step

### Original DSCIM Workflow

The original implementation provides a different workflow structure:

```python
from dscim.preprocessing.preprocessing import reduce_damages
from dscim.utils.menu_runs import run_ssps

# Reduce damages (must complete before next step)
reduce_damages(
    sector="mortality",
    config="config.yaml",
    recipe="adding_up",
    reduction="cc",
    eta=None,
    socioec="data/econ.zarr"
)

# Run damage functions and SCC calculation
# (combines steps 2 and 3, runs all at once)
run_ssps(
    sectors=["mortality"],
    pulse_years=[2020],
    menu_discs=[("adding_up", "constant"), ("risk_aversion", "ramsey")],
    eta_rhos=[[2.0, 0.001]],
    config="config.yaml",
    AR=6,
    USA=False,
    order="scc"
)
```

**Characteristics:**
- Function-based workflow
- Steps are separate function calls
- Damage function fitting and SCC calculation combined in `run_ssps`
- Intermediate results saved to predefined paths
- Less explicit control over step execution

## Comparison Table

| Aspect | Original DSCIM | DSCIM-New |
|--------|---------------|-----------|
| **Workflow Structure** | Function calls | Step objects |
| **Step Execution** | Sequential functions | Independent step objects |
| **Intermediate Results** | Saved to fixed paths | Configurable save locations |
| **Step Control** | Run complete functions | Run, pause, resume at any step |
| **Result Reuse** | Load from saved files | Pass as inputs or load from files |
| **Pipeline Options** | Single workflow path | Multiple workflow patterns |
| **Configuration** | YAML + function parameters | Pydantic schemas + step parameters |
| **Customization** | Modify function code | Insert processing between steps |

## When to Use Each Approach

### Use Complete Pipeline When:

- Running standard SCC calculations
- Processing multiple sectors/recipes/discounts at once
- Want automatic coordination of all steps
- Following established workflow

```python
# Simple, automated workflow
pipeline = DSCIMPipeline(config)
results = pipeline.run_full_pipeline(...)
```

### Use Individual Steps When:

- Developing or debugging
- Testing specific components
- Need to inspect intermediate results
- Want to modify data between steps
- Running sensitivity analyses on specific steps

```python
# Controlled, step-by-step execution
step1 = ReduceDamagesStep(...)
result1 = step1.run(...)

# Inspect, modify, or validate
validated_result = validate_damages(result1["reduced_damages"])

step2 = GenerateDamageFunctionStep(...)
result2 = step2.run(inputs={"reduced_damages": validated_result})
```

### Use Result Reuse When:

- Experimenting with discount methods
- Testing damage function specifications
- Avoiding expensive recomputation
- Sharing intermediate results between analyses

```python
# Load previous work, skip to relevant step
marginal_damages = xr.open_zarr("previous_run/marginal_damages.zarr")
scc_step = CalculateSCCStep(new_config, ...)
scc_outputs = scc_step.run(inputs={"marginal_damages": marginal_damages, ...})
```

## Understanding Modularity Benefits

### Example Workflow: Testing Discount Methods

**Without modularity** (original approach):
```python
# Must rerun entire workflow for each discount method
for discount_type in ["constant", "ramsey", "gwr"]:
    # 1. Reduce damages (same every time)
    reduce_damages(...)

    # 2. Fit damage functions (same every time)
    # 3. Calculate SCC (only this differs)
    run_ssps(discount_type=discount_type, ...)

# Time: 3 × (reduce + fit + calculate)
```

**With modularity** (DSCIM-New):
```python
# Run expensive steps once
reduce_step = ReduceDamagesStep(...)
reduced = reduce_step.run(...)  # Run once

df_step = GenerateDamageFunctionStep(...)
df_outputs = df_step.run(inputs={"reduced_damages": reduced["reduced_damages"]})  # Run once

# Only rerun SCC calculation
for discount_idx in range(3):
    scc_step = CalculateSCCStep(..., discount_config_index=discount_idx)
    scc_outputs = scc_step.run(
        inputs={"marginal_damages": df_outputs["marginal_damages"], ...}
    )

# Time: reduce + fit + (3 × calculate)
```

### Example Workflow: Custom Processing

**With modularity:**
```python
# Run standard damage reduction
reduce_step = ReduceDamagesStep(config, sector="mortality", recipe="adding_up")
reduced = reduce_step.run(inputs={}, save=False)

# Apply custom regional weighting
custom_weights = load_regional_weights()
weighted_damages = reduced["reduced_damages"] * custom_weights

# Continue with custom damages
df_step = GenerateDamageFunctionStep(config, sector="mortality", pulse_year=2020)
df_outputs = df_step.run(inputs={"reduced_damages": weighted_damages}, save=True)

# Results reflect custom weighting
```

This type of workflow is straightforward in DSCIM-New because steps are independent and accept explicit inputs.

## Simplified Workflows via Configuration

Once users understand the modular steps, they can still use simplified configuration files to streamline common workflows:

```yaml
# config.yaml - Define complete workflow
sectors:
  - mortality
  - coastal

recipes:
  - adding_up
  - risk_aversion

discounting_configs:
  - discount_type: constant
    discount_rate: 0.02
  - discount_type: ramsey
    eta: 1.45
    rho: 0.001

pulse_years:
  - 2020
  - 2030
```

Then run with single command:
```python
pipeline = DSCIMPipeline("config.yaml")
pipeline.run_full_pipeline(save=True)
# Automatically runs all sector/recipe/discount combinations
```

**Benefits:**
- Configuration captures complete analysis plan
- Easy to replicate analyses
- Version control for research workflows
- But still have option to run steps individually when needed

## Migration Path

For users familiar with the original DSCIM (this is the most straightforward path, according to Jonah's diagram):

1. **Start with complete pipeline** - Similar to `run_ssps` workflow
2. **Learn individual steps** - Understand what each step does
3. **Experiment with modularity** - Run steps separately for specific needs
4. **Leverage flexibility** - Use appropriate pattern for each task

The modular design provides options without forcing complexity.

## Summary

This refactored implementation reflects concerns discussed in previous meetings.

DSCIM-New refactors the workflow to separate concerns and provide flexibility:

**Modularity:**
- Distinct steps with clear inputs and outputs
- Each step runs independently
- Steps can be tested and validated separately

**Flexibility:**
- Run complete pipeline for standard workflows
- Run individual steps for control and debugging
- Reuse intermediate results to save computation
- Modify outputs between steps for custom analyses

**Simplicity:**
- Configuration files still provide streamlined execution
- Pipeline interface for automated workflows
- Explicit control available when needed

Both implementations compute the same results (currently cleaning and adding unit tests comparing with original dscim approach). The difference is how users interact with the workflow and manage intermediate steps.
