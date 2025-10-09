"""
Reproduce dscim-testing/run_integration_result.py using dscim-new

This script demonstrates the refactored workflow that replicates the original
dscim-testing/run_integration_result.py pipeline.

The script is structured as sequential code blocks that can be copied into
Jupyter notebook cells and run step-by-step.

Key Architectural Differences:

1. Modular Steps: Each processing step (reduce damages, generate damage functions,
   calculate SCC) is now a separate, independently executable component.

2. Explicit Data Flow: Intermediate results can be inspected between steps.
   The I/O operations are now optional, allowing in-memory processing or
   explicit saving at each step.

3. Validation: Configuration and inputs are validated via Pydantic schemas
   before execution, catching errors early.

4. Dependency Graph: Steps have explicit input/output contracts. You can run
   steps independently or chain them through the pipeline orchestrator.

Original Approach:
    - reduce_damages(): Called multiple times in loops
    - run_ssps(): Monolithic function combining damage functions and SCC calculation
    - Configuration: Loaded as dictionary, no validation
    - I/O: Always saves to disk, intermediate results not accessible

New Approach:
    - ReduceDamagesStep, GenerateDamageFunctionStep, CalculateSCCStep: Independent steps
    - DSCIMPipeline: Optional orchestrator for chaining steps
    - Configuration: Pydantic-validated DSCIMConfig
    - I/O: Optional save parameter, full access to intermediate results

Usage in Jupyter:
    Copy each section (marked with # CELL X:) into separate notebook cells
    Run cells sequentially to see step-by-step execution

Usage as script:
    python examples/scripts/reproduce_integration_result.py
"""

# =============================================================================
# CELL 1: Imports and Setup
# =============================================================================

import sys
from pathlib import Path
import yaml
import xarray as xr
import numpy as np

# Set project root
# For script: use __file__
# For notebook: use Path.cwd()
try:
    project_root = Path(__file__).resolve().parent.parent.parent
except NameError:
    # Running in notebook/REPL
    project_root = Path.cwd().resolve()

print(f"Project root: {project_root}")

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import dscim-new components
from dscim_new.config import DSCIMConfig
from dscim_new.config.schemas import (
    PipelineConfig,
    DiscountingConfig,
    PathsConfig,
    ClimateDataConfig,
    EconDataConfig,
    SectorConfig,
    DamageFunctionConfig,
    SCCConfig,
)
from dscim_new.pipeline.steps import (
    ReduceDamagesStep,
    GenerateDamageFunctionStep,
    CalculateSCCStep,
)
from dscim_new.utils import ClimateDataGenerator, DamagesDataGenerator

print("Imports successful")


# =============================================================================
# CELL 2: Configuration Setup - Choose Data Source
# =============================================================================

print("=" * 80)
print("STEP 1: CONFIGURATION SETUP")
print("=" * 80)

# Choose whether to use synthetic data or existing data
USE_SYNTHETIC_DATA = True  # Set to False to use dscim-testing data

if USE_SYNTHETIC_DATA:
    output_dir = project_root / "examples" / "workflow_output"
    print(f"\nGenerating synthetic data in: {output_dir}")
else:
    output_dir = project_root / "dscim-testing" / "dummy_data"
    print(f"\nUsing existing data in: {output_dir}")


# =============================================================================
# CELL 3: Generate Synthetic Data (Skip if using existing data)
# =============================================================================

if USE_SYNTHETIC_DATA:
    print("\nGenerating synthetic data...")

    # Create directories
    climate_dir = output_dir / "climate_data"
    damages_dir = output_dir / "damages_data"
    climate_dir.mkdir(parents=True, exist_ok=True)
    damages_dir.mkdir(parents=True, exist_ok=True)

    # Generate climate data
    print("  - Climate data...")
    climate_gen = ClimateDataGenerator(seed=42, verbose=False)
    climate_paths = climate_gen.generate_all_climate_data(str(climate_dir))

    # Generate damages and economic data
    print("  - Damages and economic data...")
    damages_gen = DamagesDataGenerator(seed=42, verbose=False)
    damages_paths = damages_gen.generate_all_damages_data(str(damages_dir))

    # Combine paths
    data_paths = {**climate_paths, **damages_paths}

    print(f"  Generated {len(data_paths)} data files")

else:
    # Use existing data paths from dscim-testing
    config_path = project_root / "dscim-testing" / "configs" / "dummy_config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    # Load to get paths
    with open(config_path, 'r') as f:
        conf_dict = yaml.safe_load(f)

    base_dir = project_root / "dscim-testing" / "dummy_data"
    data_paths = {
        "climate": str(base_dir / "climate"),
        "economic": str(base_dir / "econ" / "integration-econ-bc39.zarr"),
        "sector": str(base_dir / "sectoral" / "not_coastl_damages.zarr"),
    }

    print("  Using existing data")


# =============================================================================
# CELL 4: Create Configuration Object
# =============================================================================

print("\nCreating configuration object...")

# Previous approach: Dictionary-based config
# with open(config_path, 'r') as f:
#     conf = yaml.safe_load(f)
# No validation, access via conf['key']['subkey']

# New approach: Pydantic-validated configuration
# Note: DSCIMConfig requires paths, econdata, and sectors

if USE_SYNTHETIC_DATA:
    # Build configuration with synthetic data paths
    config = DSCIMConfig(
        paths=PathsConfig(
            reduced_damages_library=str(output_dir / "reduced_damages"),
            ssp_damage_function_library=str(output_dir / "damage_functions"),
            AR6_ssp_results=str(output_dir / "scc_results"),
        ),
        econdata=EconDataConfig(global_ssp=data_paths["economic"]),
        sectors={
            "not_coastal": SectorConfig(
                sector_path=data_paths["noncoastal_damages"],
                histclim="histclim_dummy",  # Variable name in dataset
                delta="delta_dummy",  # Variable name in dataset
                formula="damages ~ -1 + anomaly + np.power(anomaly, 2)",
            )
        },
        climate_data=ClimateDataConfig(
            gmst_path=data_paths["gmst"],
            gmsl_path=data_paths["gmsl"],
            fair_temperature_path=data_paths["fair_temperature"],
            fair_gmsl_path=data_paths["fair_gmsl"],
            pulse_conversion_path=data_paths["pulse_conversion"],
        ),
        damage_function=DamageFunctionConfig(
            formula="damages ~ -1 + anomaly + np.power(anomaly, 2)",
            fit_type="ols",
            fit_method="rolling_window",  # Use original dscim approach
            window_size=5,
            year_range=(2020, 2101),
        ),
        scc=SCCConfig(pulse_years=[2020]),
    )
else:
    # Load from existing config
    config = DSCIMConfig.from_yaml(str(config_path))
    # Override output paths
    config.paths.reduced_damages_library = str(output_dir / "reduced_damages")
    config.paths.ssp_damage_function_library = str(output_dir / "damage_functions")
    config.paths.AR6_ssp_results = str(output_dir / "scc_results")

print("Configuration created successfully")
print(f"Sectors available: {list(config.sectors.keys())}")


# =============================================================================
# CELL 5: Configure Pipeline Parameters
# =============================================================================

print("\n" + "=" * 80)
print("STEP 2: PIPELINE PARAMETERS")
print("=" * 80)

# Previous approach: Hard-coded parameters scattered in script
# eta_rhos = [[2.0, 0.0001]]
# reductions = ['cc', 'no_cc']
# recipe_discs = product(['adding_up', 'risk_aversion', 'equity'], [...])

# New approach: Centralized configuration
if config.pipeline is None:
    config.pipeline = PipelineConfig()

# Set recipes to process
config.pipeline.recipes = ["adding_up", "risk_aversion", "equity"]
config.pipeline.reductions = ["cc", "no_cc"]
config.pipeline.eta_values = [2.0]

# Configure discount methods (5 methods matching original)
config.discounting = [
    DiscountingConfig(discount_type="constant", discount_rate=0.02),
    DiscountingConfig(
        discount_type="ramsey", eta=2.0, rho=0.0001, ramsey_method="naive_ramsey"
    ),
    DiscountingConfig(
        discount_type="ramsey", eta=2.0, rho=0.0001, ramsey_method="euler_ramsey"
    ),
    DiscountingConfig(
        discount_type="gwr", eta=2.0, rho=0.0001, gwr_method="naive_gwr"
    ),
    DiscountingConfig(
        discount_type="gwr", eta=2.0, rho=0.0001, gwr_method="euler_gwr"
    ),
]

# Calculate expected combinations
n_recipes = len(config.pipeline.recipes)
n_discount_methods = len(config.discounting)
n_combinations = n_recipes * n_discount_methods

print(f"\nPipeline configured:")
print(f"  Recipes: {config.pipeline.recipes}")
print(f"  Reductions: {config.pipeline.reductions}")
print(f"  Discount methods: {n_discount_methods}")
print(f"\nExpected: {n_recipes} recipes x {n_discount_methods} methods = {n_combinations} combinations")


# =============================================================================
# CELL 6: Select Sector to Process
# =============================================================================

sector_name = list(config.sectors.keys())[0]
sector_config = config.sectors[sector_name]

print(f"\nProcessing sector: {sector_name}")
print(f"  Data path: {sector_config.sector_path}")
print(f"  Formula: {sector_config.formula}")


# =============================================================================
# CELL 7: STEP 1 - Reduce Damages
# =============================================================================

print("\n" + "=" * 80)
print("STEP 3: REDUCE DAMAGES")
print("=" * 80)

# Previous Approach:
#   Multiple calls to reduce_damages() in nested loops
#   for sector, reduction in product(sectors, reductions):
#       for recipe in recipes:
#           reduce_damages(sector, config, recipe, reduction, ...)

# New Approach:
#   Explicit step execution with optional I/O
#   step = ReduceDamagesStep(config, sector, recipe, reduction)
#   result = step.run(inputs={...}, save=False)  # In-memory
#   OR
#   result = step.run(inputs={...}, save=True)   # Save to disk

# Key Difference:
#   - Results accessible for inspection before next step
#   - I/O is optional (save parameter)
#   - Each step independently executable

print(f"\nProcessing sector: {sector_name}")
print(f"Recipes: {config.pipeline.recipes}")
print(f"Reductions: {config.pipeline.reductions}")

# Store results for next step
reduced_damages_results = {}

for recipe in config.pipeline.recipes:
    for reduction in config.pipeline.reductions:
        print(f"\n  Processing: {recipe} x {reduction}")

        # Determine eta parameter based on recipe
        eta = 2.0 if recipe in ["risk_aversion", "equity"] else None

        # Create reduction step
        step = ReduceDamagesStep(
            config=config,
            sector=sector_name,
            recipe=recipe,
            reduction=reduction,
            eta=eta,
            verbose=False,
        )

        # Execute step
        # Note: save=False keeps result in memory
        # Note: save=True writes to config.paths.reduced_damages_library
        output = step.run(
            inputs={
                "sector_damages_path": sector_config.sector_path,
                "socioec_path": config.econdata.global_ssp,
            },
            save=True,  # Change to False to keep only in memory
        )

        # Store result for next step
        key = (recipe, reduction)
        reduced_damages_results[key] = output["reduced_damages"]

        # Inspect result (now possible with explicit data flow)
        result = reduced_damages_results[key]
        if isinstance(result, xr.Dataset):
            print(f"    Type: Dataset with {len(result.data_vars)} variables")
            print(f"    Variables: {list(result.data_vars)}")
        else:
            print(f"    Shape: {result.shape}")
            print(f"    Type: {type(result).__name__}")

print(f"\nDamage reduction complete: {len(reduced_damages_results)} combinations")


# =============================================================================
# CELL 8: Inspect Reduced Damages (Optional)
# =============================================================================

# This is now possible because results are accessible
# Previous approach: Results saved to disk, not accessible

print("\nInspecting reduced damages results...")
print(f"Available combinations: {len(reduced_damages_results)}")

# Example: Look at one result
sample_key = list(reduced_damages_results.keys())[0]
sample_data = reduced_damages_results[sample_key]

print(f"\nSample: {sample_key[0]} x {sample_key[1]}")
print(f"  Shape: {sample_data.shape}")
print(f"  Dimensions: {dict(sample_data.dims) if hasattr(sample_data, 'dims') else list(sample_data.sizes)}")
print(f"  Data variables: {list(sample_data.data_vars) if hasattr(sample_data, 'data_vars') else 'N/A'}")


# =============================================================================
# CELL 9: STEP 2 - Generate Damage Functions
# =============================================================================

print("\n" + "=" * 80)
print("STEP 4: GENERATE DAMAGE FUNCTIONS")
print("=" * 80)

# Previous Approach:
#   Embedded in run_ssps(), not separately accessible
#   run_ssps(sectors, pulse_years, menu_discs, ...)
#   Damage function generation happens internally

# New Approach:
#   Explicit step with accessible inputs/outputs
#   step = GenerateDamageFunctionStep(config, sector, pulse_year)
#   result = step.run(inputs={'reduced_damages': data}, save=False)

# Key Difference:
#   - Damage function results accessible before SCC calculation
#   - Can inspect fit quality, coefficients, marginal damages
#   - Save or keep in memory independently

pulse_year = config.scc.pulse_years[0]
damage_function_results = {}

print(f"\nGenerating damage functions:")
print(f"  Sector: {sector_name}")
print(f"  Pulse year: {pulse_year}")

for recipe in config.pipeline.recipes:
    print(f"\n  Processing: {recipe}")

    # Use 'cc' reduction for damage function generation
    key = (recipe, "cc")
    if key not in reduced_damages_results:
        print(f"    Warning: No reduced damages found, skipping")
        continue

    reduced_damages = reduced_damages_results[key]

    # Create damage function step
    # Note: Step doesn't need recipe/eta - those are embedded in reduced_damages
    step = GenerateDamageFunctionStep(
        config=config,
        sector=sector_name,
        pulse_year=pulse_year,
        verbose=False,
    )

    # Execute step with explicit input
    output = step.run(
        inputs={"reduced_damages": reduced_damages},
        save=True,  # Change to False for in-memory only
    )

    # Store results
    damage_function_results[recipe] = output

    # Inspect outputs (now accessible)
    coefs = output["damage_function_coefficients"]
    marg_dmg = output["marginal_damages"]

    if isinstance(coefs, xr.Dataset):
        print(f"    Coefficients: Dataset with {len(coefs.data_vars)} variables")
    else:
        print(f"    Coefficients shape: {coefs.shape}")

    if isinstance(marg_dmg, xr.Dataset):
        print(f"    Marginal damages: Dataset with {len(marg_dmg.data_vars)} variables")
    else:
        print(f"    Marginal damages shape: {marg_dmg.shape}")

print(f"\nDamage functions generated: {len(damage_function_results)} recipes")


# =============================================================================
# CELL 10: Inspect Damage Functions (Optional)
# =============================================================================

print("\nInspecting damage function results...")

# Example: Look at coefficients for one recipe
sample_recipe = list(damage_function_results.keys())[0]
sample_df = damage_function_results[sample_recipe]

print(f"\nSample recipe: {sample_recipe}")
print(f"  Coefficients shape: {sample_df['damage_function_coefficients'].shape}")
print(f"  Marginal damages shape: {sample_df['marginal_damages'].shape}")

# Access coefficient values
coefs = sample_df['damage_function_coefficients']
print(f"  Coefficients: {coefs.values}")


# =============================================================================
# CELL 11: Load Economic Data for SCC Calculation
# =============================================================================

print("\n" + "=" * 80)
print("STEP 5: CALCULATE SCC - Load Economic Data")
print("=" * 80)

# Load consumption data (required for SCC calculation)
print("\nLoading economic data...")
try:
    econ_data = xr.open_zarr(config.econdata.global_ssp, chunks=None)
    consumption = econ_data["gdppc"]
    print(f"  Consumption loaded: {consumption.shape}")
    print(f"  Dimensions: {list(consumption.dims)}")
except Exception as e:
    print(f"  Error: {e}")
    consumption = None


# =============================================================================
# CELL 12: Calculate SCC for All Combinations
# =============================================================================

print("\n" + "=" * 80)
print("STEP 5: CALCULATE SCC - All Recipe-Discount Combinations")
print("=" * 80)

# Previous Approach:
#   Embedded in run_ssps(), all processed internally
#   run_ssps(sectors, pulse_years, menu_discs, eta_rhos, ...)
#   No visibility into which combination is running

# New Approach:
#   Explicit iteration with progress tracking
#   for recipe in recipes:
#       for discount_config in discount_configs:
#           step = CalculateSCCStep(...)
#           result = step.run(...)

# Key Difference:
#   - See exactly which combination is processing
#   - Access all intermediate outputs (discount factors, consumption, etc.)
#   - Errors in one combination don't stop others
#   - Can process subsets

if consumption is None:
    print("Cannot proceed without consumption data")
else:
    scc_results = {}
    combination_count = 0

    print(f"\nCalculating SCC for {n_combinations} combinations:")
    print(f"  {n_recipes} recipes x {n_discount_methods} discount methods")

    for recipe in config.pipeline.recipes:
        if recipe not in damage_function_results:
            print(f"\n  Warning: No damage function for {recipe}, skipping")
            continue

        marginal_damages = damage_function_results[recipe]["marginal_damages"]

        for disc_idx, discount_config in enumerate(config.discounting):
            combination_count += 1

            # Create readable name
            discount_name = discount_config.discount_type
            if hasattr(discount_config, "ramsey_method") and discount_config.ramsey_method:
                discount_name = discount_config.ramsey_method
            elif hasattr(discount_config, "gwr_method") and discount_config.gwr_method:
                discount_name = discount_config.gwr_method

            print(f"\n  [{combination_count}/{n_combinations}] {recipe} x {discount_name}")

            # Create SCC calculation step
            step = CalculateSCCStep(
                config=config,
                sector=sector_name,
                pulse_year=pulse_year,
                recipe=recipe,
                discount_config_index=disc_idx,
                verbose=False,
            )

            # Execute step with explicit inputs
            try:
                output = step.run(
                    inputs={
                        "marginal_damages": marginal_damages,
                        "consumption": consumption,
                    },
                    save=True,  # Change to False for in-memory only
                )

                # Store result
                key = (recipe, discount_name)
                scc_results[key] = output

                # Inspect outputs (now accessible)
                scc = output["scc"]
                print(f"    SCC shape: {scc.shape}")
                print(f"    SCC mean: {float(scc.mean()):.2f}")

                # Additional outputs accessible
                if "discount_factors" in output:
                    print(f"    Discount factors: {output['discount_factors'].shape}")
                if "global_consumption" in output:
                    print(f"    Global consumption: {output['global_consumption'].shape}")

            except Exception as e:
                print(f"    Error: {str(e)[:100]}")

    print(f"\nSCC calculation complete: {len(scc_results)} combinations")


# =============================================================================
# CELL 13: Inspect SCC Results (Optional)
# =============================================================================

if consumption is not None and scc_results:
    print("\nInspecting SCC results...")
    print(f"Available combinations: {len(scc_results)}")

    # Example: Look at one SCC result
    sample_key = list(scc_results.keys())[0]
    sample_scc = scc_results[sample_key]["scc"]

    print(f"\nSample: {sample_key[0]} x {sample_key[1]}")
    print(f"  SCC shape: {sample_scc.shape}")
    print(f"  SCC mean: {float(sample_scc.mean()):.2f}")
    print(f"  SCC median: {float(sample_scc.median()):.2f}")
    print(f"  SCC std: {float(sample_scc.std()):.2f}")


# =============================================================================
# CELL 14: Summary
# =============================================================================

print("\n" + "=" * 80)
print("WORKFLOW SUMMARY")
print("=" * 80)

print(f"\nResults generated:")
print(f"  Reduced damages: {len(reduced_damages_results)} combinations")
print(f"  Damage functions: {len(damage_function_results)} recipes")
if consumption is not None:
    print(f"  SCC calculations: {len(scc_results)} combinations")

print(f"\nOutput locations:")
print(f"  Reduced damages: {config.paths.reduced_damages_library}")
print(f"  Damage functions: {config.paths.ssp_damage_function_library}")
print(f"  SCC results: {config.paths.AR6_ssp_results}")

print("\n" + "=" * 80)
print("KEY ARCHITECTURAL IMPROVEMENTS")
print("=" * 80)

print("\n1. Modularity")
print("   Previous: Monolithic functions (run_ssps does everything)")
print("   New: Independent steps (reduce, damage function, SCC)")

print("\n2. Data Flow")
print("   Previous: Hidden intermediate results, always saved to disk")
print("   New: Explicit inputs/outputs, optional I/O, accessible intermediates")

print("\n3. Validation")
print("   Previous: Runtime errors from invalid parameters")
print("   New: Pydantic validation catches errors at load time")

print("\n4. Flexibility")
print("   Previous: Must run entire workflow, hard to run subsets")
print("   New: Run individual steps, inspect between steps, easy customization")

print("\n5. Debugging")
print("   Previous: Difficult to isolate issues in monolithic functions")
print("   New: Clear separation enables step-by-step debugging")


# =============================================================================
# CELL 15: Optional - Pipeline Orchestrator Approach
# =============================================================================

print("\n" + "=" * 80)
print("OPTIONAL: PIPELINE ORCHESTRATOR")
print("=" * 80)

# The orchestrator provides automated step chaining while maintaining
# the same modular architecture underneath

print("\nThe DSCIMPipeline orchestrator can automate the workflow:")
print("  - Handles iteration over recipes/reductions/discount methods")
print("  - Manages data flow between steps")
print("  - Stores intermediate results")
print("  - Optional - you can still use individual steps as shown above")

from dscim_new.pipeline import DSCIMPipeline

print("\nExample usage:")
print("  pipeline = DSCIMPipeline(config)")
print("  pipeline.reduce_damages(sector, recipe, reduction, save=True)")
print("\nDependency graph:")
print("  reduce_damages -> damage_function -> SCC")
print("  Each step can run independently or through orchestrator")

print("\n" + "=" * 80)
print("WORKFLOW COMPLETE")
print("=" * 80)
