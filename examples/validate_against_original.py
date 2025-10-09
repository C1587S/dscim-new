"""
Validation Script: dscim-new vs Original dscim

This script validates the dscim-new refactored implementation against the original
dscim library using the dummy data from dscim-testing/.

It is designed to be run interactively, similar to a notebook, with each section
executed sequentially.

Expected Outputs:
-----------------
The script will:
1. Load dummy sectoral damages data
2. Generate damage function coefficients
3. Calculate marginal damages using FAIR projections
4. Compute global consumption with extrapolation
5. Calculate discount factors
6. Compute final SCC values

The outputs will be compared against the reference outputs in:
    dscim-testing/dummy_data/results/AR6_ssp/dummy_not_coastl_sector/2020/unmasked/

Usage:
------
Run this file interactively or execute it as a script:
    python examples/validate_against_original.py
"""

import sys
from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
import yaml
from datetime import datetime

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from dscim_new.preprocessing.damage_functions import DamageFunctionProcessor
from dscim_new.core.damage_functions import calculate_marginal_damages_from_fair
from dscim_new.preprocessing.consumption import extrapolate_global_consumption
from dscim_new.core.discounting import calculate_stream_discount_factors_per_scenario
from dscim_new.core.scc import calculate_scc_with_uncertainty, summarize_scc
from dscim_new.config.schemas import (
    DamageFunctionConfig,
    ClimateDataConfig,
    DiscountingConfig
)

# =============================================================================
# CONFIGURATION
# =============================================================================
print("="*80)
print("DSCIM-NEW VALIDATION AGAINST ORIGINAL DSCIM")
print("="*80)
print(f"Start time: {datetime.now()}")
print("="*80)

# Configuration matching run_integration_result.py
sector_name = "dummy_not_coastl_sector"
recipe = "risk_aversion"
eta = 2.0
rho = 0.0001
pulse_year = 2020
discount_type = "naive_ramsey"

print(f"\nConfiguration:")
print(f"  Sector: {sector_name}")
print(f"  Recipe: {recipe}")
print(f"  Discount Type: {discount_type}")
print(f"  Eta: {eta}, Rho: {rho}")
print(f"  Pulse Year: {pulse_year}")

# =============================================================================
# STEP 1: LOAD CONFIGURATION
# =============================================================================
print(f"\n{datetime.now()}: STEP 1 - Loading Configuration")
print("-"*80)

config_path = project_root / "dscim-testing" / "configs" / "dummy_config.yaml"
print(f"Loading configuration from: {config_path}")

with open(config_path, "r") as f:
    conf = yaml.safe_load(f)

print("✓ Configuration loaded")

# =============================================================================
# STEP 2: LOAD SECTORAL DAMAGES
# =============================================================================
print(f"\n{datetime.now()}: STEP 2 - Loading Sectoral Damages")
print("-"*80)

sector_config = conf["sectors"][sector_name]
sector_path = project_root / "dscim-testing" / sector_config["sector_path"]

print(f"Loading sectoral damages from: {sector_path}")
sectoral_damages = xr.open_zarr(sector_path)

print(f"  Loaded damages with dimensions: {sectoral_damages.dims}")
print(f"  Variables: {list(sectoral_damages.data_vars)}")

# Extract damage variable
if "damages" in sectoral_damages:
    damages = sectoral_damages["damages"]
else:
    data_vars = list(sectoral_damages.data_vars)
    if data_vars:
        damages = sectoral_damages[data_vars[0]]
        print(f"  Using damage variable: {data_vars[0]}")
    else:
        raise ValueError("No damage variable found in sectoral data")

print(f"  Damage dimensions: {damages.dims}")
print(f"  Damage shape: {damages.shape}")
print(f"  Damage coords: {list(damages.coords.keys())}")

# =============================================================================
# STEP 3: LOAD CLIMATE DATA
# =============================================================================
print(f"\n{datetime.now()}: STEP 3 - Loading Climate Data")
print("-"*80)

base_path = project_root / "dscim-testing"

# Load GMST data (CSV)
gmst_path = base_path / conf["AR6_ssp_climate"]["gmst_path"]
print(f"Loading GMST data from: {gmst_path}")
gmst_df = pd.read_csv(gmst_path)
print(f"  GMST shape: {gmst_df.shape}")
print(f"  GMST columns: {list(gmst_df.columns)}")

# Load GMSL data (Zarr)
gmsl_path = base_path / conf["AR6_ssp_climate"]["gmsl_path"]
print(f"Loading GMSL data from: {gmsl_path}")
gmsl_data = xr.open_zarr(gmsl_path)
print(f"  GMSL dims: {gmsl_data.dims}")
print(f"  GMSL coords: {list(gmsl_data.coords.keys())}")

# Load FAIR temperature projections
fair_temp_path = base_path / conf["AR6_ssp_climate"]["gmst_fair_path"]
print(f"Loading FAIR temperature from: {fair_temp_path}")
fair_temp = xr.open_dataset(fair_temp_path)
print(f"  FAIR temp dims: {fair_temp.dims}")
print(f"  FAIR temp coords: {list(fair_temp.coords.keys())}")

# Load FAIR GMSL projections
fair_gmsl_path = base_path / conf["AR6_ssp_climate"]["gmsl_fair_path"]
print(f"Loading FAIR GMSL from: {fair_gmsl_path}")
fair_gmsl = xr.open_dataset(fair_gmsl_path)
print(f"  FAIR GMSL dims: {fair_gmsl.dims}")

# Load pulse conversion factors
conversion_path = base_path / conf["AR6_ssp_climate"]["damages_pulse_conversion_path"]
print(f"Loading conversion factors from: {conversion_path}")
conversion = xr.open_dataset(conversion_path)
print(f"  Conversion dims: {conversion.dims}")

# Combine FAIR temperature and GMSL into single datasets
fair_combined = xr.merge([fair_temp, fair_gmsl])

print("✓ Climate data loaded")
print(f"  FAIR combined variables: {list(fair_combined.data_vars)}")
print(f"  FAIR combined dims: {fair_combined.dims}")

# Separate control and pulse scenarios (following original dscim approach)
# See: dscim/src/dscim/menu/simple_storage.py:193-202
print("\nSeparating control and pulse scenarios...")

# Determine which variables are in the dataset
anomaly_vars = []
if any('temperature' in v for v in fair_combined.data_vars):
    anomaly_vars.append('temperature')
if any('gmsl' in v for v in fair_combined.data_vars):
    anomaly_vars.append('gmsl')

print(f"  Anomaly variables detected: {anomaly_vars}")

# Create fair_control: select control_* variables and rename
control_var_names = [f"control_{var}" for var in anomaly_vars if f"control_{var}" in fair_combined.data_vars]
if not control_var_names:
    # Try medianparams_control_*
    control_var_names = [f"medianparams_control_{var}" for var in anomaly_vars if f"medianparams_control_{var}" in fair_combined.data_vars]

print(f"  Control variable names: {control_var_names}")

fair_control = fair_combined[control_var_names]
# Rename: remove control_ or medianparams_control_ prefix
fair_control = fair_control.rename({v: v.replace('control_', '').replace('medianparams_', '') for v in control_var_names})

# Create fair_pulse: select pulse_* variables and rename
pulse_var_names = [f"pulse_{var}" for var in anomaly_vars if f"pulse_{var}" in fair_combined.data_vars]
if not pulse_var_names:
    # Try medianparams_pulse_*
    pulse_var_names = [f"medianparams_pulse_{var}" for var in anomaly_vars if f"medianparams_pulse_{var}" in fair_combined.data_vars]

print(f"  Pulse variable names: {pulse_var_names}")

fair_pulse = fair_combined[pulse_var_names]
# Rename: remove pulse_ or medianparams_pulse_ prefix
fair_pulse = fair_pulse.rename({v: v.replace('pulse_', '').replace('medianparams_', '') for v in pulse_var_names})

print(f"\n  Fair control variables after rename: {list(fair_control.data_vars)}")
print(f"  Fair pulse variables after rename: {list(fair_pulse.data_vars)}")

# =============================================================================
# STEP 4: LOAD ECONOMIC DATA
# =============================================================================
print(f"\n{datetime.now()}: STEP 4 - Loading Economic Data")
print("-"*80)

econ_path = project_root / "dscim-testing" / conf["econdata"]["global_ssp"]
print(f"Loading economic data from: {econ_path}")
econ_data = xr.open_zarr(econ_path)

print(f"  Economic data dims: {econ_data.dims}")
print(f"  Variables: {list(econ_data.data_vars)}")

# Extract GDP and population
gdp = econ_data["gdp"] if "gdp" in econ_data else econ_data["GDP"]
pop = econ_data["pop"] if "pop" in econ_data else econ_data["population"]

print(f"  GDP shape: {gdp.shape}")
print(f"  GDP years: {gdp.year.values}")
print(f"  Population shape: {pop.shape}")
print(f"  Population years: {pop.year.values}")
print("✓ Economic data loaded")

# =============================================================================
# STEP 5: INSPECT DIMENSION NAMES
# =============================================================================
print(f"\n{datetime.now()}: STEP 5 - Inspecting Dimension Names")
print("-"*80)

print("\nDamage dimensions:")
print(f"  dims: {damages.dims}")
print(f"  coords: {list(damages.coords.keys())}")

print("\nGMST dimensions (from DataFrame):")
print(f"  columns: {list(gmst_df.columns)}")

print("\nGMSL dimensions:")
print(f"  dims: {gmsl_data.dims}")
print(f"  coords: {list(gmsl_data.coords.keys())}")

# Determine the actual dimension names used in the data
# The error shows: "dimensions FrozenMappingWarningOnValuesAccess({'year': 11, 'rcp': 2, 'gcm': 2})"
# So the climate data uses 'rcp' and 'gcm' instead of 'ssp' and 'model'

# Map dimension names
damage_ssp_dim = None
damage_model_dim = None
damage_year_dim = None

for dim in damages.dims:
    if 'ssp' in dim.lower() or 'rcp' in dim.lower() or 'scenario' in dim.lower():
        damage_ssp_dim = dim
    if 'model' in dim.lower() or 'gcm' in dim.lower():
        damage_model_dim = dim
    if 'year' in dim.lower():
        damage_year_dim = dim

print(f"\nDetected dimension names in damages:")
print(f"  SSP/Scenario dimension: {damage_ssp_dim}")
print(f"  Model dimension: {damage_model_dim}")
print(f"  Year dimension: {damage_year_dim}")

print(f"\n⚠ NOTE: The climate data dimensions determine what the rolling window will use.")
print(f"  Climate (GMST) has: {list(gmst_df.columns)}")
print(f"  When converted to xarray, it will likely have: year, rcp, gcm")

# =============================================================================
# STEP 6: GENERATE DAMAGE FUNCTION COEFFICIENTS
# =============================================================================
print(f"\n{datetime.now()}: STEP 6 - Generating Damage Function Coefficients")
print("-"*80)

# Get formula from config
formula = sector_config["formula"]
print(f"Formula: {formula}")

# Create damage function config
# NOTE: We need to pass the correct dimension names to the rolling window function
df_config = DamageFunctionConfig(
    formula=formula,
    fit_type="ols",
    fit_method="rolling_window",  # Original dscim uses rolling window
    window_size=5,
    year_range=(2020, 2101),
    extrapolation_method="global_c_ratio",
    save_points=True
)

# Create climate config
climate_config = ClimateDataConfig(
    gmst_path=None,  # Will pass data directly
    gmsl_path=None
)

# Create processor
processor = DamageFunctionProcessor(
    config=df_config,
    climate_config=climate_config,
    verbose=True
)

print(f"Generating damage function with discount_type: {discount_type}")

# The processor will auto-detect dimension names (rcp/gcm vs ssp/model)
# and pass them to the rolling window function

# Generate damage function
result = processor.generate_damage_function(
    damages=damages,
    sector=sector_name,
    pulse_year=pulse_year,
    climate_data=(gmst_df, gmsl_data),
    discount_type=discount_type,
    save_outputs=False
)

damage_function_coefficients = result.coefficients_grid

print(f"\n✓ Damage function coefficients generated")
print(f"  Dimensions: {damage_function_coefficients.dims}")
print(f"  Shape: {damage_function_coefficients.sizes}")
print(f"  Variables: {list(damage_function_coefficients.data_vars)}")

# =============================================================================
# STEP 7: CALCULATE GLOBAL CONSUMPTION (MUST COME BEFORE MARGINAL DAMAGES)
# =============================================================================
print(f"\n{datetime.now()}: STEP 7 - Calculating Global Consumption")
print("-"*80)

# Get end year from config
ext_end_year = conf["global_parameters"]["ext_end_year"]
print(f"Extrapolation end year: {ext_end_year}")

# Determine actual year range available in data
available_years = sorted(gdp.year.values)
print(f"Available years in GDP data: {available_years[0]} to {available_years[-1]}")

# Use the last available years for extrapolation
# Original dscim uses years 2085-2099, but we'll use what's available
if len(available_years) >= 2:
    # Use last ~15 years of data for extrapolation
    end_year = int(available_years[-1])
    start_year = max(int(available_years[0]), end_year - 14)
    print(f"Using extrapolation window: {start_year} to {end_year}")
else:
    raise ValueError(f"Not enough years in GDP data: {available_years}")

global_consumption = extrapolate_global_consumption(
    gdp=gdp,
    pop=pop,
    discount_type=discount_type,
    start_year=start_year,
    end_year=end_year,
    target_year=ext_end_year,
    method="growth_constant"
)

print(f"\n✓ Global consumption calculated")
print(f"  Dimensions: {global_consumption.dims}")
print(f"  Shape: {global_consumption.sizes}")

# =============================================================================
# STEP 8: CALCULATE MARGINAL DAMAGES WITH FAIR AGGREGATION
# =============================================================================
print(f"\n{datetime.now()}: STEP 8 - Calculating Marginal Damages")
print("-"*80)

# Get parameters from config
weitzman_params = conf["global_parameters"]["weitzman_parameter"]
fair_agg = conf["global_parameters"]["fair_aggregation"]

# Filter aggregation methods:
# - 'median' is computed POST-SCC, not in marginal damages
# - 'median_params' requires separate FAIR scenario (not yet implemented)
fair_agg_for_marginal_damages = [agg for agg in fair_agg if agg not in ['median', 'median_params']]
print(f"Weitzman parameters: {weitzman_params}")
print(f"FAIR aggregation for marginal damages: {fair_agg_for_marginal_damages}")
print(f"Full FAIR aggregation (including median): {fair_agg}")

# Calculate marginal damages with proper FAIR aggregation
# The function will auto-detect which dimensions to collapse (gcm, simulation, etc.)
marginal_damages = calculate_marginal_damages_from_fair(
    fair_control=fair_control,
    fair_pulse=fair_pulse,
    damage_function_coefficients=damage_function_coefficients,
    formula=formula,
    global_consumption=global_consumption,
    pulse_conversion_factor=1.0,
    fair_aggregation=fair_agg_for_marginal_damages,
    fair_dims=None,  # Auto-detect
    weitzman_parameters=weitzman_params,
    eta=eta
)

print(f"\n✓ Marginal damages calculated")
print(f"  Dimensions: {marginal_damages.dims}")
print(f"  Shape: {marginal_damages.sizes}")

# =============================================================================
# STEP 9: CALCULATE DISCOUNT FACTORS
# =============================================================================
print(f"\n{datetime.now()}: STEP 9 - Calculating Discount Factors")
print("-"*80)

print(f"Eta: {eta}, Rho: {rho}")

discount_config = DiscountingConfig(
    discount_type=discount_type,
    eta=eta,
    rho=rho,
    discrete=conf["global_parameters"]["discrete_discounting"]
)

discount_factors = calculate_stream_discount_factors_per_scenario(
    global_consumption_no_pulse=global_consumption,
    population=pop,
    eta=discount_config.eta,
    rho=discount_config.rho,
    pulse_year=pulse_year,
    discount_type=discount_type,
    fair_aggregation=fair_agg_for_marginal_damages,
    fair_dims=None,  # Auto-detect
    discrete=discount_config.discrete,
    ext_end_year=ext_end_year
)

print(f"\n✓ Discount factors calculated")
print(f"  Dimensions: {discount_factors.dims}")
print(f"  Shape: {discount_factors.sizes}")

# =============================================================================
# STEP 10: CALCULATE SCC
# =============================================================================
print(f"\n{datetime.now()}: STEP 10 - Calculating SCC")
print("-"*80)

scc = calculate_scc_with_uncertainty(
    marginal_damages=marginal_damages,
    discount_factors=discount_factors,
    discount_type=discount_type,
    fair_aggregation_methods=fair_agg_for_marginal_damages,
    include_median='median' in fair_agg,
    pulse_year=pulse_year
)

print(f"\n✓ SCC calculated")
print(f"  Dimensions: {scc.dims}")
print(f"  Shape: {scc.sizes}")

# Convert to DataFrame for easier visualization
print("\nConverting SCC to DataFrame for easier visualization...")
scc_df = scc.to_dataframe(name='scc').reset_index()

# Display basic info
print(f"  DataFrame shape: {scc_df.shape}")
print(f"  Columns: {list(scc_df.columns)}")

# Show first few rows
print("\n  First 10 rows:")
print(scc_df.head(10))

# Calculate summary statistics by aggregation method
if 'fair_aggregation' in scc_df.columns:
    print("\n  Summary by FAIR aggregation method:")
    summary_by_agg = scc_df.groupby('fair_aggregation')['scc'].describe()
    print(summary_by_agg)

# Overall summary
print("\n  Overall SCC statistics:")
print(f"    Mean: ${scc_df['scc'].mean():.2f} per tonne CO2")
print(f"    Median: ${scc_df['scc'].median():.2f}")
print(f"    Std: ${scc_df['scc'].std():.2f}")
print(f"    Min: ${scc_df['scc'].min():.2f}")
print(f"    Max: ${scc_df['scc'].max():.2f}")

# =============================================================================
# STEP 11: COMPARE WITH REFERENCE OUTPUT
# =============================================================================
print(f"\n{datetime.now()}: STEP 11 - Comparing with Reference Outputs")
print("-"*80)

# Load reference SCC from original dscim output
ref_path = (
    project_root / "dscim-testing" / "dummy_data" / "results" / "AR6_ssp" /
    sector_name / str(pulse_year) / "unmasked" /
    f"{recipe}_{discount_type}_eta{eta}_rho{rho}_scc.nc4"
)

if ref_path.exists():
    print(f"Loading reference SCC from: {ref_path.name}")
    reference_scc = xr.open_dataset(ref_path)

    print(f"\n  Reference SCC dimensions: {reference_scc.dims}")
    print(f"  Reference SCC variables: {list(reference_scc.data_vars)}")

    # Compare dimensions and values
    print(f"\n  Dimension comparison:")
    print(f"    New:       {scc.dims}")
    print(f"    Reference: {reference_scc.dims}")

    # Try to compare a specific aggregation method
    if "scc" in reference_scc:
        ref_values = reference_scc["scc"]
        print(f"\n  Value comparison (sample):")
        print(f"    New SCC mean: {float(scc.mean()):.4f}")
        print(f"    Ref SCC mean: {float(ref_values.mean()):.4f}")

        # Calculate relative difference
        rel_diff = abs(float(scc.mean()) - float(ref_values.mean())) / float(ref_values.mean()) * 100
        print(f"    Relative difference: {rel_diff:.2f}%")
else:
    print(f"⚠ Reference file not found: {ref_path}")
    print(f"  Skipping comparison")

# =============================================================================
# STEP 12: SUMMARIZE RESULTS
# =============================================================================
print(f"\n{datetime.now()}: STEP 12 - Summarizing Results")
print("-"*80)

summary = summarize_scc(scc)

print(f"\nSCC Summary:")
print(f"  Mean: ${float(summary['mean'].mean()):.2f} per tonne CO2")
print(f"  Median: ${float(summary['median'].mean()):.2f} per tonne CO2")
print(f"  Std Dev: ${float(summary['std'].mean()):.2f}")

if 'percentiles' in summary:
    print(f"\nPercentiles:")
    for p in [5, 25, 50, 75, 95]:
        val = float(summary['percentiles'].sel(percentile=p).mean())
        print(f"  {p}th: ${val:.2f}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*80)
print("VALIDATION COMPLETE")
print("="*80)
print(f"End time: {datetime.now()}")
print("="*80)

# Store results in a dictionary for inspection
results = {
    "damage_function_coefficients": damage_function_coefficients,
    "marginal_damages": marginal_damages,
    "global_consumption": global_consumption,
    "discount_factors": discount_factors,
    "scc": scc,
    "summary": summary
}

print("\nResults stored in 'results' dictionary")
print("Available keys:", list(results.keys()))
