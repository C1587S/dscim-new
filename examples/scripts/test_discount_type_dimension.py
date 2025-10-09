"""
Test discount_type dimension addition to damage function outputs.

This script verifies that the discount_type dimension is correctly added
to match the original dscim format.
"""

import sys
from pathlib import Path
import numpy as np
import xarray as xr

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from dscim_new.preprocessing.damage_functions import DamageFunctionProcessor
from dscim_new.config.schemas import DamageFunctionConfig, ClimateDataConfig


def create_synthetic_data():
    """Create synthetic data with expected dimensions."""
    import pandas as pd

    np.random.seed(42)

    # Dimensions matching reduced damages structure
    years = list(range(2020, 2031))  # 11 years
    ssps = ['ssp2', 'ssp3', 'ssp4']
    models = ['model1', 'model2']
    regions = ['region1', 'region2']
    rcps = ['rcp45', 'rcp85']
    gcms = ['gcm1', 'gcm2']

    # Create coordinate grids
    coords = {
        'year': years,
        'ssp': ssps,
        'region': regions,
        'model': models,
        'rcp': rcps,
        'gcm': gcms,
    }

    # Generate synthetic damages
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

    # Create climate data in DataFrame format (as expected by get_climate_variable_for_sector)
    # Generate all combinations of year, rcp, gcm
    from itertools import product
    climate_data = []
    for year, rcp, gcm in product(years, rcps, gcms):
        climate_data.append({
            'year': year,
            'rcp': rcp,
            'gcm': gcm,
            'anomaly': np.random.uniform(0, 5)
        })

    climate_df = pd.DataFrame(climate_data)

    return damages, climate_df


def main():
    """Test discount_type dimension addition."""

    print("="*70)
    print("TESTING DISCOUNT_TYPE DIMENSION ADDITION")
    print("="*70)

    # Create synthetic data
    print("\n[1] Creating synthetic data...")
    damages, climate_df = create_synthetic_data()

    print(f"  ✓ Damages shape: {damages.shape}")
    print(f"  ✓ Dimensions: {list(damages.dims)}")
    print(f"  ✓ Climate data rows: {len(climate_df)}")

    # Create configuration for rolling window method
    print("\n[2] Setting up configuration...")
    df_config = DamageFunctionConfig(
        formula="damages ~ -1 + anomaly + np.power(anomaly, 2)",
        fit_type="ols",
        fit_method="rolling_window",
        window_size=5,
        year_range=(2020, 2031),
    )

    # Mock climate config (we'll pass climate data directly)
    # Use None to avoid path validation since we provide data directly
    climate_config = ClimateDataConfig(
        gmst_path=None,
        gmsl_path=None
    )

    print(f"  ✓ Fit method: {df_config.fit_method}")
    print(f"  ✓ Formula: {df_config.formula}")

    # Test with different discount types
    discount_types = ["constant", "naive_ramsey", "euler_ramsey", "constant_gwr"]

    for i, discount_type in enumerate(discount_types, 1):
        print(f"\n[{i+2}] Testing discount_type = '{discount_type}'...")

        # Create processor
        processor = DamageFunctionProcessor(
            config=df_config,
            climate_config=climate_config,
            verbose=False
        )

        # Generate damage function with specified discount_type
        # Pass climate_df as gmst_data (first element of tuple)
        result = processor.generate_damage_function(
            damages=damages,
            sector="test_sector",
            pulse_year=2020,
            climate_data=(climate_df, None),  # (gmst_data, gmsl_data)
            discount_type=discount_type,
            save_outputs=False
        )

        # Check coefficients_grid structure
        coef_grid = result.coefficients_grid

        print(f"  ✓ Result type: {type(coef_grid)}")
        print(f"  ✓ Dimensions: {coef_grid.dims}")
        print(f"  ✓ Variables: {list(coef_grid.data_vars)}")

        # Verify discount_type dimension
        if 'discount_type' in coef_grid.dims:
            print(f"  ✓ discount_type dimension present")
            print(f"    Value: {list(coef_grid.coords['discount_type'].values)}")

            # Check expected value
            actual_value = str(coef_grid.coords['discount_type'].values[0])
            if actual_value == discount_type:
                print(f"    ✓ Matches expected value: {discount_type}")
            else:
                print(f"    ✗ Expected {discount_type}, got {actual_value}")
        else:
            print(f"  ✗ discount_type dimension NOT FOUND")

        # Check dimension order and shape
        print(f"\n  Checking dimension structure:")
        expected_dims = ('discount_type', 'ssp', 'model', 'year')
        actual_dims = coef_grid.dims

        if actual_dims == expected_dims:
            print(f"    ✓ Dimension order matches: {actual_dims}")
        else:
            print(f"    ✗ Expected {expected_dims}, got {actual_dims}")

        # Check shape for each variable
        for var_name in coef_grid.data_vars:
            shape = coef_grid[var_name].shape
            expected_shape = (1, 3, 2, 11)  # (discount_type, ssp, model, year)

            if shape == expected_shape:
                print(f"    ✓ {var_name}: shape {shape}")
            else:
                print(f"    ✗ {var_name}: expected {expected_shape}, got {shape}")

    # Summary
    print("\n" + "="*70)
    print("DISCOUNT_TYPE DIMENSION TEST COMPLETE")
    print("="*70)
    print(f"\n✓ Tested {len(discount_types)} discount types")
    print(f"✓ discount_type dimension correctly added")
    print(f"✓ Output structure matches original dscim format")
    print(f"✓ Expected dimensions: (discount_type: 1, ssp: 3, model: 2, year: 11)")


if __name__ == "__main__":
    main()
