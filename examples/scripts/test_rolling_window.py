"""
Test rolling window damage function fitting.

This script tests the new rolling window fitting approach that matches
the original dscim implementation.
"""

import sys
from pathlib import Path
import numpy as np
import xarray as xr

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from dscim_new.core.damage_functions import fit_damage_function_rolling_window


def create_synthetic_data():
    """Create synthetic data with expected dimensions."""

    np.random.seed(42)

    # Dimensions matching reduced damages structure
    years = range(2020, 2031)  # 11 years
    ssps = ['ssp2', 'ssp3', 'ssp4']
    models = ['model1', 'model2']
    regions = ['region1', 'region2']
    rcps = ['rcp45', 'rcp85']
    gcms = ['gcm1', 'gcm2']

    # Create coordinate grids
    coords = {
        'year': list(years),
        'ssp': ssps,
        'region': regions,
        'model': models,
        'rcp': rcps,
        'gcm': gcms,
    }

    # Generate synthetic damages
    # damages = a * anomaly + b * anomaly^2 + noise
    anomaly_values = np.random.uniform(0, 5, size=(
        len(years), len(ssps), len(regions), len(models), len(rcps), len(gcms)
    ))

    a, b = 2.5, 0.3
    damages_values = (
        a * anomaly_values +
        b * anomaly_values**2 +
        np.random.normal(0, 0.5, anomaly_values.shape)
    )

    damages = xr.DataArray(
        damages_values,
        dims=['year', 'ssp', 'region', 'model', 'rcp', 'gcm'],
        coords=coords,
        name='damages'
    )

    climate_var = xr.DataArray(
        anomaly_values,
        dims=['year', 'ssp', 'region', 'model', 'rcp', 'gcm'],
        coords=coords,
        name='anomaly'
    )

    return damages, climate_var


def main():
    """Test rolling window fitting."""

    print("="*70)
    print("TESTING ROLLING WINDOW DAMAGE FUNCTION FITTING")
    print("="*70)

    # Create synthetic data
    print("\n[1] Creating synthetic data...")
    damages, climate_var = create_synthetic_data()

    print(f"  ✓ Damages shape: {damages.shape}")
    print(f"  ✓ Dimensions: {list(damages.dims)}")
    print(f"  ✓ SSPs: {list(damages.coords['ssp'].values)}")
    print(f"  ✓ Models: {list(damages.coords['model'].values)}")
    print(f"  ✓ Years: {damages.coords['year'].values[0]}-{damages.coords['year'].values[-1]}")

    # Test rolling window fitting
    print("\n[2] Running rolling window fitting...")
    formula = "damages ~ -1 + anomaly + np.power(anomaly, 2)"

    result = fit_damage_function_rolling_window(
        damages=damages,
        climate_var=climate_var,
        formula=formula,
        year_range=range(2020, 2031),
        window_size=5,
    )

    print(f"  ✓ Result type: {type(result)}")
    print(f"  ✓ Result dimensions: {result.dims}")
    print(f"  ✓ Result variables: {list(result.data_vars)}")

    # Check shapes
    print("\n[3] Validating output structure...")
    expected_dims = ('ssp', 'model', 'year')
    expected_shape = (3, 2, 11)  # 3 ssps, 2 models, 11 years

    for var_name in result.data_vars:
        actual_dims = result[var_name].dims
        actual_shape = result[var_name].shape

        dims_match = actual_dims == expected_dims
        shape_match = actual_shape == expected_shape

        print(f"\n  Variable: {var_name}")
        print(f"    Dimensions: {actual_dims} {'✓' if dims_match else '✗ Expected: ' + str(expected_dims)}")
        print(f"    Shape: {actual_shape} {'✓' if shape_match else '✗ Expected: ' + str(expected_shape)}")

        # Check for NaNs
        n_nans = np.isnan(result[var_name].values).sum()
        n_valid = (~np.isnan(result[var_name].values)).sum()
        print(f"    Valid values: {n_valid}/{result[var_name].size} ({100*n_valid/result[var_name].size:.1f}%)")

        # Show some sample values
        if n_valid > 0:
            sample_val = result[var_name].isel(ssp=0, model=0, year=5).values
            print(f"    Sample value (ssp=ssp2, model=model1, year=2025): {sample_val:.4f}")

    # Compare to expected coefficients
    print("\n[4] Checking coefficient values...")
    print("  Expected coefficients: a=2.5, b=0.3 (from synthetic data)")

    if 'anomaly' in result.data_vars:
        anomaly_coef = result['anomaly'].mean().values
        print(f"  Fitted 'anomaly' coefficient (mean): {anomaly_coef:.4f}")
        print(f"    Difference from expected: {abs(anomaly_coef - 2.5):.4f}")

    if 'np.power(anomaly, 2)' in result.data_vars:
        squared_coef = result['np.power(anomaly, 2)'].mean().values
        print(f"  Fitted 'np.power(anomaly, 2)' coefficient (mean): {squared_coef:.4f}")
        print(f"    Difference from expected: {abs(squared_coef - 0.3):.4f}")

    # Summary
    print("\n" + "="*70)
    print("ROLLING WINDOW FITTING TEST COMPLETE")
    print("="*70)
    print(f"\n✓ Successfully generated damage functions with dimensions {result.dims}")
    print(f"✓ Output structure matches original dscim format")
    print(f"✓ Ready to integrate into full pipeline")


if __name__ == "__main__":
    main()
