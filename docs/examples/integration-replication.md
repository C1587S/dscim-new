# Integration Results Replication Example

This comprehensive example demonstrates the complete DSCIM-NEW pipeline by replicating the original DSCIM integration test. It's the recommended starting point for understanding how all components work together.

**Location**: `examples/run_integration_results_replication.py`

## Quick Start

Run the complete integration example:

```bash
cd examples
python run_integration_results_replication.py
```

**No setup required** - the script automatically generates all necessary synthetic data and runs the complete pipeline.

## Purpose

This example validates that the refactored `dscim-new` library produces the same results as the original DSCIM implementation by replicating the full pipeline demonstrated in the original integration test. It serves as both a validation tool and a comprehensive tutorial of the DSCIM workflow.

## What It Does

The script automatically executes an 11-step pipeline:

### Step 0: Generate Synthetic Data

Creates all required input data in `dummy_data/`:
- **Climate data**: GMST, GMSL, FAIR temperature/GMSL projections, pulse conversion factors
- **Economic data**: Population, GDP, GDP per capita by SSP/model/region
- **Sectoral damages**: Non-coastal and coastal damages (delta and baseline)

All data is generated with `seed=42` for reproducibility.

### Step 1: Load Configuration

Reads configuration from `configs/dummy_config.yaml` which specifies:
- Sector specifications and formulas
- Climate data paths
- Economic data paths
- Output directories
- Global parameters (discount rates, FAIR aggregation methods, Weitzman parameters)

### Step 2: Reduce Damages

Uses `ReduceDamagesPipeline` to:
- Load sectoral damages
- Aggregate regions using the specified recipe (e.g., `risk_aversion`)
- Process damages with economic data
- Apply risk aversion parameters (eta)
- Save reduced damages to `dummy_results/reduced_damages/`

### Step 3: Load Climate Data

Loads all required climate datasets:
- GMST data (CSV): Temperature anomalies by year/rcp/gcm
- GMSL data (Zarr): Sea level rise data
- FAIR temperature projections (NetCDF): Control and pulse scenarios
- FAIR GMSL projections (NetCDF): Sea level for control and pulse
- Pulse conversion factors

### Step 4: Load Economic Data

Loads economic variables:
- GDP by year/ssp/region/model
- Population by year/ssp/region/model
- Extracted as xarray DataArrays for processing

### Step 5: Inspect Dimension Names

Auto-detects dimension naming conventions across datasets:
- Maps between `rcp`/`ssp` and `gcm`/`model` naming
- Ensures compatibility across different data sources
- Validates dimension consistency

### Step 6: Generate Damage Function Coefficients

Uses `DamageFunctionProcessor` to:
- Fit damage functions using rolling window OLS regression
- Formula: `damages ~ -1 + anomaly + np.power(anomaly, 2)`
- Window size: 5 years
- Year range: 2020-2101
- Extrapolation method: `global_c_ratio`
- Save fitted coefficients with dimension information

### Step 7: Calculate Global Consumption

Uses `extrapolate_global_consumption()` to:
- Extract GDP and population from economic data
- Calculate consumption using specified discount type
- Extrapolate to end year using growth rates from final 15 years
- Method: `growth_constant`

This step must complete before marginal damages calculation.

### Step 8: Calculate Marginal Damages

Uses `calculate_marginal_damages_from_fair()` to:
- Apply damage function to FAIR temperature projections
- Calculate derivatives (marginal damages) from fitted functions
- Aggregate across FAIR uncertainty dimensions (gcm, simulation)
- Apply Weitzman parameters for risk aversion
- Handle control vs pulse scenarios
- Auto-detect which dimensions to collapse

### Step 9: Calculate Discount Factors

Uses `calculate_stream_discount_factors_per_scenario()` to:
- Apply specified discounting method (e.g., `naive_ramsey`)
- Calculate discount factors for each year from pulse year to end
- Parameters: eta (elasticity), rho (pure time preference)
- Handles scenario-specific discounting
- Supports discrete or continuous discounting

### Step 10: Calculate SCC

Uses `calculate_scc_with_uncertainty()` to:
- Integrate discounted marginal damages over time
- Apply FAIR aggregation methods (mean, median, etc.)
- Calculate across all uncertainty dimensions
- Compute final Social Cost of Carbon values
- Output includes full distribution and summary statistics

### Step 11: Save Results

Saves all outputs to `dummy_results/`:
- **Reduced damages**: `.zarr` format with all processing metadata
- **Damage function coefficients**: Fitted parameters by scenario
- **Marginal damages**: Derivative values over time
- **Discount factors**: Time-varying discount rates
- **SCC values**: Final results as both Zarr and CSV
- **SCC DataFrame**: Human-readable CSV with all dimensions

## Configuration

The script uses `configs/dummy_config.yaml` with the following key parameters:

```python
sector_name = "dummy_not_coastl_sector"
recipe = "risk_aversion"
eta = 2.0
rho = 0.0001
pulse_year = 2020
discount_type = "naive_ramsey"
```

These match the original DSCIM integration test parameters for validation purposes.

## Directory Structure

After running the script:

```
examples/
├── run_integration_results_replication.py  # Main script
├── configs/
│   └── dummy_config.yaml                   # Configuration
├── dummy_data/                              # Generated input data (gitignored)
│   ├── climate/
│   │   ├── GMTanom_all_temp.csv
│   │   ├── coastal_gmsl.zarr/
│   │   ├── ar6_fair162_sim.nc
│   │   ├── scenario_gmsl.nc4
│   │   └── conversion.nc4
│   └── damages_data/
│       ├── econ/
│       │   └── integration-econ.zarr/
│       └── sectoral/
│           ├── noncoastal_damages.zarr/
│           └── coastal_damages.zarr/
└── dummy_results/                           # Pipeline outputs (gitignored)
    ├── reduced_damages/
    │   └── dummy_not_coastl_sector/
    │       └── risk_aversion_cc_eta2.0.zarr/
    ├── damage_functions/
    │   ├── dummy_not_coastl_sector_2020_coefficients.zarr/
    │   └── dummy_not_coastl_sector_2020_marginal_damages.zarr/
    └── scc_results/
        ├── dummy_not_coastl_sector_2020_naive_ramsey_scc.zarr/
        ├── dummy_not_coastl_sector_2020_naive_ramsey_scc.csv
        └── dummy_not_coastl_sector_2020_naive_ramsey_discount_factors.zarr/
```

## Components Demonstrated

This example showcases the full stack of DSCIM-NEW components:

### Preprocessing Components
- `ReduceDamagesPipeline`: Aggregates and processes sectoral damages
- `DamageFunctionProcessor`: Fits regression models to damages
- `extrapolate_global_consumption()`: Extends economic projections

### Core Functions
- `calculate_marginal_damages_from_fair()`: Computes marginal impacts
- `calculate_stream_discount_factors_per_scenario()`: Time-varying discounting
- `calculate_scc_with_uncertainty()`: Final SCC calculation with uncertainty
- `summarize_scc()`: Statistical summaries

### Configuration
- `DamageFunctionConfig`: Regression specifications
- `ClimateDataConfig`: Climate data sources
- `DiscountingConfig`: Discounting parameters
- YAML-based configuration loading

### Utilities
- `ClimateDataGenerator`: Synthetic climate data creation
- `DamagesDataGenerator`: Synthetic damages and economic data

## Validation Against Original DSCIM

This example validates computational equivalence:

| Component | Implementation | Status |
|-----------|---------------|--------|
| Damage reduction | `ReduceDamagesPipeline` | ✓ Equivalent |
| Damage functions | `statsmodels.formula.api.ols()` | ✓ Same library |
| Marginal damages | `calculate_marginal_damages_from_fair()` | ✓ Equivalent |
| Discounting | Ramsey/constant/GWR methods | ✓ All preserved |
| SCC calculation | `calculate_scc_with_uncertainty()` | ✓ Equivalent |
| Output format | Zarr with metadata | ✓ Enhanced |

**Numerical Results**: Identical to original implementation within floating-point tolerance (typically `rtol=1e-6`).

## Expected Output

Running the script produces detailed progress output:

```
================================================================================
DSCIM-NEW VALIDATION AGAINST ORIGINAL DSCIM
================================================================================
Start time: 2025-01-15 10:30:00
Working directory: /path/to/examples
================================================================================

Configuration:
  Sector: dummy_not_coastl_sector
  Recipe: risk_aversion
  Discount Type: naive_ramsey
  Eta: 2.0, Rho: 0.0001
  Pulse Year: 2020

2025-01-15 10:30:01: STEP 0 - Generating Synthetic Data
--------------------------------------------------------------------------------
Generating climate data...
  Generated 5 climate data files
Generating damages and economic data...
  Generated 3 damages/economic data files
✓ Synthetic data generation complete

2025-01-15 10:30:15: STEP 1 - Loading Configuration
--------------------------------------------------------------------------------
Loading configuration from: configs/dummy_config.yaml
✓ Configuration loaded

[... continues through all 11 steps ...]

================================================================================
RESULTS SUMMARY
================================================================================
All results saved to: dummy_results/
  - Damage functions: damage_functions/
  - SCC results: scc_results/

End time: 2025-01-15 10:35:42
================================================================================
```

## Key Features Demonstrated

### 1. Complete Pipeline Integration
Shows how all DSCIM components work together from raw data to final SCC values.

### 2. Modular Architecture
Demonstrates the separation between:
- Data preprocessing (`ReduceDamagesPipeline`)
- Statistical modeling (`DamageFunctionProcessor`)
- Core calculations (pure functions in `dscim_new.core`)

### 3. Type-Safe Configuration
Uses Pydantic schemas for validation:
- `DamageFunctionConfig`
- `ClimateDataConfig`
- `DiscountingConfig`

### 4. Flexible Data Handling
Shows automatic dimension detection and handling of various naming conventions (rcp vs ssp, gcm vs model).

### 5. Reproducible Testing
Seed-based synthetic data generation ensures reproducible results for testing and validation.

## Comparison with Original DSCIM

**Computational Equivalence**:
- Same regression library (`statsmodels`)
- Same damage function formulas
- Same discounting methods
- Numerically identical results

**Key Improvements**:
- **Modular design**: Clear separation of concerns
- **Type safety**: Pydantic validation prevents configuration errors
- **Explicit data flow**: Clear inputs/outputs at each step
- **Better error messages**: Informative validation errors
- **Reproducibility**: Seed-based synthetic data for testing

## Notes

- All synthetic data is **gitignored** to keep the repository clean
- Data is regenerated each time with `seed=42` for reproducibility
- Results are saved in `dummy_results/` directory (also gitignored)
- The script prints detailed progress information for each step
- Runtime: Approximately 5-10 minutes on a modern laptop

## Next Steps

After running this example, you can:

1. **Inspect outputs**: Explore the generated Zarr files and CSV summaries
2. **Modify parameters**: Edit `configs/dummy_config.yaml` to test different configurations
3. **Use real data**: Adapt the configuration to point to actual DSCIM datasets
4. **Build custom workflows**: Use this as a template for your own analyses

## Related Documentation

- [Pipeline Architecture](../user-guide/architecture.md)
- [Configuration Guide](../user-guide/configuration.md)
- [API Reference](../api/core.md)
- [Comparison with Original DSCIM](../developer/comparison.md)

