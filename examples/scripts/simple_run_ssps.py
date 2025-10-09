"""
Simple standalone run_ssps example.

This is a minimal working example that demonstrates the damage function
and SCC calculation workflow without requiring existing data files.

It creates synthetic data and runs the complete workflow:
1. Generate synthetic climate and damage data
2. Reduce damages
3. Generate damage functions
4. Calculate SCC

This can be run directly without any setup.
"""

import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path

# Add parent to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dscim_new.utils import ClimateDataGenerator
from dscim_new.preprocessing import (
    DamageFunctionProcessor,
    SCCCalculator,
)
from dscim_new.config.schemas import (
    ClimateDataConfig,
    DamageFunctionConfig,
    DiscountingConfig,
    SCCConfig,
)


def create_synthetic_reduced_damages():
    """Create simple synthetic reduced damages for demonstration."""
    print("Creating synthetic reduced damages...")

    # Simple damage data: damages increase with temperature
    years = np.arange(2020, 2101)
    regions = ["USA", "EUR", "CHN"]
    n_years = len(years)
    n_regions = len(regions)

    # Create damage values that increase over time
    damages_data = np.zeros((n_years, n_regions))

    for i, year in enumerate(years):
        for j, region in enumerate(regions):
            # Damages increase quadratically with time
            time_factor = (year - 2020) / 80  # 0 to 1
            base_damage = 1000 * time_factor**2  # Billions

            # Different regions have different damage levels
            region_factor = {"USA": 1.2, "EUR": 0.9, "CHN": 1.5}[region]

            damages_data[i, j] = base_damage * region_factor

    # Create xarray DataArray
    damages = xr.DataArray(
        damages_data,
        dims=["year", "region"],
        coords={
            "year": years,
            "region": regions,
        }
    )

    damages.attrs["units"] = "billion_dollars"
    damages.attrs["description"] = "Synthetic reduced damages"

    print(f"  Created damages: {damages.shape}")
    return damages


def create_synthetic_consumption():
    """Create simple synthetic consumption data."""
    print("Creating synthetic consumption data...")

    years = np.arange(2020, 2101)
    regions = ["USA", "EUR", "CHN"]

    # GDP per capita that grows over time
    consumption_data = np.zeros((len(years), len(regions)))

    for i, year in enumerate(years):
        for j, region in enumerate(regions):
            # Base consumption with growth
            base = {"USA": 60000, "EUR": 50000, "CHN": 15000}[region]
            growth_rate = {"USA": 0.015, "EUR": 0.012, "CHN": 0.045}[region]

            consumption_data[i, j] = base * (1 + growth_rate) ** (year - 2020)

    consumption = xr.DataArray(
        consumption_data,
        dims=["year", "region"],
        coords={
            "year": years,
            "region": regions,
        }
    )

    consumption.attrs["units"] = "dollars_per_capita"
    print(f"  Created consumption: {consumption.shape}")
    return consumption


def main():
    """Run simple end-to-end workflow."""

    print("="*60)
    print("Simple run_ssps Demonstration")
    print("="*60)

    # =========================================================================
    # 1. Generate Synthetic Climate Data
    # =========================================================================
    print("\n[1/4] Generating synthetic climate data...")

    output_dir = Path("examples/workflow_output/simple_example")
    climate_dir = output_dir / "climate_data"
    climate_dir.mkdir(parents=True, exist_ok=True)

    generator = ClimateDataGenerator(seed=42, verbose=False)
    climate_paths = generator.generate_all_climate_data(str(climate_dir))

    print("  ✓ Climate data generated")

    # =========================================================================
    # 2. Create Synthetic Damages and Consumption
    # =========================================================================
    print("\n[2/4] Creating synthetic damages and consumption...")

    reduced_damages = create_synthetic_reduced_damages()
    consumption = create_synthetic_consumption()

    print("  ✓ Data created")

    # =========================================================================
    # 3. Generate Damage Function
    # =========================================================================
    print("\n[3/4] Generating damage function...")

    # Setup configurations
    climate_config = ClimateDataConfig(
        gmst_path=climate_paths["gmst"],
        gmsl_path=climate_paths["gmsl"],
    )

    df_config = DamageFunctionConfig(
        formula="damages ~ -1 + anomaly + np.power(anomaly, 2)",
        fit_type="ols",
        save_points=True,
        n_points=50,
    )

    # Create processor
    df_processor = DamageFunctionProcessor(
        config=df_config,
        climate_config=climate_config,
        verbose=False,
    )

    # Generate damage function
    df_output_dir = output_dir / "damage_functions"
    df_result = df_processor.generate_damage_function(
        damages=reduced_damages,
        sector="example",
        pulse_year=2020,
        save_outputs=True,
        output_dir=str(df_output_dir),
        output_format="zarr"
    )

    print("  ✓ Damage function generated")
    print(f"    Coefficients: {list(df_result.coefficients.coefficient.values)}")
    for coef_name in df_result.coefficients.coefficient.values:
        coef_val = float(df_result.coefficients.sel(coefficient=coef_name).values)
        print(f"      {coef_name}: {coef_val:.6f}")

    # =========================================================================
    # 4. Calculate SCC
    # =========================================================================
    print("\n[4/4] Calculating SCC...")

    scc_config = SCCConfig(
        pulse_years=[2020],
        fair_aggregation="mean",
        calculate_quantiles=True,
        quantile_levels=[0.05, 0.5, 0.95],
    )

    calculator = SCCCalculator(
        scc_config=scc_config,
        verbose=False,
    )

    scc_output_dir = output_dir / "scc_results"

    # Calculate SCC with different discounting methods
    discount_configs = [
        DiscountingConfig(discount_type="constant", discount_rate=0.02),
        DiscountingConfig(discount_type="ramsey", eta=1.45, rho=0.001),
    ]

    print("\n  Results:")
    print("  " + "-"*56)
    print(f"  {'Discount Method':<20} {'Mean SCC':<20} {'Median SCC':<15}")
    print("  " + "-"*56)

    for discount_config in discount_configs:
        scc_result = calculator.calculate_scc(
            marginal_damages=df_result.marginal_damages,
            discount_config=discount_config,
            pulse_year=2020,
            consumption=consumption,
            sector="example",
            save_outputs=True,
            output_dir=str(scc_output_dir),
            output_format="zarr"
        )

        discount_name = discount_config.discount_type
        if discount_config.eta:
            discount_name += f" (η={discount_config.eta})"
        elif discount_config.discount_rate:
            discount_name += f" (r={discount_config.discount_rate})"

        print(f"  {discount_name:<20} ${scc_result.scc_mean:<19.2f} ${scc_result.scc_median:<14.2f}")

    print("  " + "-"*56)

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Output directory: {output_dir}")
    print(f"  - Climate data: {climate_dir}")
    print(f"  - Damage functions: {df_output_dir}")
    print(f"  - SCC results: {scc_output_dir}")
    print("\n✓ Workflow complete!")


if __name__ == "__main__":
    main()
