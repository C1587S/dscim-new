"""
Complete End-to-End DSCIM Pipeline Example

This script demonstrates the full workflow from scratch (synthetic data generation)
to final SCC calculation, using the Pipeline architecture.

Workflow:
1. Generate synthetic climate data
2. Generate synthetic damages data
3. Reduce damages (ReduceDamagesStep)
4. Generate damage functions (GenerateDamageFunctionStep)
5. Calculate SCC (CalculateSCCStep)

Usage:
    python full_pipeline_example.py [--verbose] [--output-dir PATH]
"""

import sys
from pathlib import Path
import argparse

# Add parent directory to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from dscim_new.config.schemas import (
    DSCIMConfig,
    ClimateDataConfig,
    DamageFunctionConfig,
    DiscountingConfig,
    SCCConfig,
)
from dscim_new.utils import ClimateDataGenerator, DamagesDataGenerator
from dscim_new.pipeline import (
    ReduceDamagesStep,
    GenerateDamageFunctionStep,
    CalculateSCCStep,
)
import xarray as xr


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run complete DSCIM pipeline from scratch"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="examples/pipeline_output",
        help="Output directory for all results (default: examples/pipeline_output)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output with Rich formatting"
    )
    parser.add_argument(
        "--sector",
        type=str,
        default="not_coastal",
        choices=["not_coastal", "coastal"],
        help="Sector to process (default: not_coastal)"
    )
    parser.add_argument(
        "--pulse-year",
        type=int,
        default=2020,
        help="Pulse year for SCC calculation (default: 2020)"
    )
    return parser.parse_args()


def setup_directories(base_dir: str):
    """Create directory structure for outputs."""
    base_path = Path(base_dir)
    dirs = {
        "climate": base_path / "climate_data",
        "damages": base_path / "damages_data" / "sectoral",
        "econ": base_path / "damages_data" / "econ",
        "reduced": base_path / "reduced_damages",
        "df": base_path / "damage_functions",
        "scc": base_path / "scc_results",
    }

    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    return dirs


def generate_synthetic_data(dirs: dict, verbose: bool = False):
    """
    Generate all required synthetic data.

    Returns:
        dict: Paths to generated data files
    """
    print("\n" + "="*70)
    print("STEP 1: Generate Synthetic Data")
    print("="*70)

    # Generate climate data
    print("\n[1.1] Generating climate data...")
    climate_gen = ClimateDataGenerator(seed=42, verbose=verbose)
    climate_paths = climate_gen.generate_all_climate_data(str(dirs["climate"]))

    print(f"  ✓ Generated {len(climate_paths)} climate files")

    # Generate damages and economic data
    print("\n[1.2] Generating damages and economic data...")
    damages_gen = DamagesDataGenerator(seed=42, verbose=verbose)
    damages_paths = damages_gen.generate_all_damages_data(str(dirs["damages"].parent))

    print(f"  ✓ Generated {len(damages_paths)} damage/economic files")

    return {**climate_paths, **damages_paths}


def create_config(data_paths: dict, dirs: dict, sector: str) -> DSCIMConfig:
    """
    Create DSCIM configuration with all necessary settings.

    Parameters:
        data_paths: Dictionary of paths to data files
        dirs: Dictionary of output directories
        sector: Sector name

    Returns:
        DSCIMConfig: Complete configuration object
    """
    print("\n" + "="*70)
    print("STEP 2: Configure Pipeline")
    print("="*70)

    # Create base configuration
    config = DSCIMConfig()

    # Climate data configuration
    config.climate_data = ClimateDataConfig(
        gmst_path=data_paths["gmst"],
        gmsl_path=data_paths["gmsl"],
        fair_temperature_path=data_paths["fair_temperature"],
        fair_gmsl_path=data_paths["fair_gmsl"],
        pulse_conversion_path=data_paths["pulse_conversion"],
    )
    print("  ✓ Climate data configured")

    # Economic data configuration
    config.econdata.global_ssp = data_paths["economic"]
    print("  ✓ Economic data configured")

    # Sector configuration
    config.sectors = {
        "not_coastal": {
            "sector_path": data_paths["noncoastal_damages"],
            "formula": "damages ~ -1 + anomaly + np.power(anomaly, 2)",
        },
        "coastal": {
            "sector_path": data_paths["coastal_damages"],
            "formula": "damages ~ -1 + gmsl + np.power(gmsl, 2)",
        }
    }
    print(f"  ✓ Configured {len(config.sectors)} sectors")

    # Damage function configuration
    config.damage_function = DamageFunctionConfig(
        formula=config.sectors[sector]["formula"],
        fit_type="ols",
        extrapolation_method="global_c_ratio",
        save_points=True,
        n_points=100,
    )
    print("  ✓ Damage function parameters configured")

    # Discounting configurations (matching original run_ssps)
    config.discounting = [
        # Constant discounting at 2%
        DiscountingConfig(
            discount_type="constant",
            discount_rate=0.02,
        ),
        # Ramsey discounting
        DiscountingConfig(
            discount_type="ramsey",
            eta=1.45,
            rho=0.001,
        ),
        # GWR discounting
        DiscountingConfig(
            discount_type="gwr",
            eta=1.45,
            rho=0.001,
            gwr_method="naive_gwr",
        ),
    ]
    print(f"  ✓ Configured {len(config.discounting)} discounting methods")

    # SCC calculation configuration
    config.scc = SCCConfig(
        pulse_years=[2020],
        fair_aggregation="mean",
        calculate_quantiles=True,
        quantile_levels=[0.05, 0.25, 0.5, 0.75, 0.95],
        save_discount_factors=True,
        save_uncollapsed=False,
    )
    print("  ✓ SCC calculation configured")

    # Output paths
    config.paths.reduced_damages_library = str(dirs["reduced"])
    config.paths.ssp_damage_function_library = str(dirs["df"])
    config.paths.AR6_ssp_results = str(dirs["scc"])
    print("  ✓ Output paths configured")

    return config


def run_pipeline(config: DSCIMConfig, sector: str, pulse_year: int, verbose: bool = False):
    """
    Execute the complete DSCIM pipeline.

    Parameters:
        config: DSCIM configuration
        sector: Sector to process
        pulse_year: Pulse year for SCC calculation
        verbose: Enable verbose output

    Returns:
        dict: Pipeline results
    """
    print("\n" + "="*70)
    print("STEP 3: Execute Pipeline")
    print("="*70)

    results = {}

    # -------------------------------------------------------------------------
    # Step 3.1: Reduce Damages
    # -------------------------------------------------------------------------
    print(f"\n[3.1] Reducing damages for {sector}...")

    sector_config = config.sectors[sector]

    reduce_step = ReduceDamagesStep(
        config=config,
        sector=sector,
        recipe="adding_up",
        reduction="cc",
        verbose=verbose,
    )

    reduce_inputs = {
        'sector_damages_path': sector_config["sector_path"],
        'socioec_path': config.econdata.global_ssp
    }

    reduce_outputs = reduce_step.run(inputs=reduce_inputs, save=True)
    reduced_damages = reduce_outputs["reduced_damages"]

    print(f"  ✓ Reduced damages shape: {reduced_damages.shape}")
    print(f"  ✓ Dimensions: {list(reduced_damages.dims)}")

    results["reduced_damages"] = reduced_damages

    # -------------------------------------------------------------------------
    # Step 3.2: Generate Damage Functions
    # -------------------------------------------------------------------------
    print(f"\n[3.2] Generating damage functions for {sector}...")

    df_step = GenerateDamageFunctionStep(
        config=config,
        sector=sector,
        pulse_year=pulse_year,
        verbose=verbose,
    )

    df_outputs = df_step.run(
        inputs={"reduced_damages": reduced_damages},
        save=True
    )

    coefficients = df_outputs["damage_function_coefficients"]
    marginal_damages = df_outputs["marginal_damages"]

    print(f"  ✓ Damage function fitted")
    print(f"  ✓ Coefficients: {list(coefficients.coefficient.values)}")
    print(f"  ✓ Marginal damages shape: {marginal_damages.shape}")

    results["coefficients"] = coefficients
    results["marginal_damages"] = marginal_damages

    # -------------------------------------------------------------------------
    # Step 3.3: Calculate SCC for Each Discounting Method
    # -------------------------------------------------------------------------
    print(f"\n[3.3] Calculating SCC for {sector}...")

    # Load consumption data
    econ_data = xr.open_zarr(config.econdata.global_ssp, chunks=None)
    if 'gdppc' not in econ_data.data_vars:
        raise ValueError("Economic data must contain 'gdppc' variable")
    consumption = econ_data['gdppc']

    scc_results = {}

    for discount_idx, discount_config in enumerate(config.discounting):
        discount_name = discount_config.discount_type
        print(f"\n  [3.3.{discount_idx+1}] Calculating SCC with {discount_name} discounting...")

        scc_step = CalculateSCCStep(
            config=config,
            sector=sector,
            pulse_year=pulse_year,
            discount_config_index=discount_idx,
            verbose=verbose,
        )

        scc_outputs = scc_step.run(
            inputs={
                "marginal_damages": marginal_damages,
                "consumption": consumption,
            },
            save=True
        )

        scc = scc_outputs["scc"]
        scc_results[discount_name] = scc

        print(f"    ✓ SCC calculated: shape {scc.shape}")

    results["scc"] = scc_results

    return results


def print_summary(results: dict, output_dir: str):
    """Print summary of pipeline results."""
    print("\n" + "="*70)
    print("PIPELINE SUMMARY")
    print("="*70)

    print("\n✓ Pipeline completed successfully!")

    print("\nGenerated outputs:")
    print(f"  • Reduced damages: {results['reduced_damages'].shape}")
    print(f"  • Damage function coefficients: {results['coefficients'].shape}")
    print(f"  • Marginal damages: {results['marginal_damages'].shape}")

    print(f"\n  • SCC results ({len(results['scc'])} discounting methods):")
    for method, scc in results['scc'].items():
        print(f"    - {method}: {scc.shape}")

    print(f"\nAll outputs saved to: {output_dir}")
    print("\nOutput structure:")
    print(f"  {output_dir}/")
    print(f"  ├── climate_data/          (GMST, GMSL, FAIR projections)")
    print(f"  ├── damages_data/          (Sectoral damages & economic data)")
    print(f"  ├── reduced_damages/       (Climate-reduced damages)")
    print(f"  ├── damage_functions/      (Fitted coefficients & marginal damages)")
    print(f"  └── scc_results/           (SCC values & quantiles)")


def main():
    """Main pipeline execution."""
    args = parse_args()

    print("="*70)
    print("DSCIM FULL PIPELINE EXAMPLE")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Sector: {args.sector}")
    print(f"  Pulse year: {args.pulse_year}")
    print(f"  Verbose: {args.verbose}")

    # Setup directories
    dirs = setup_directories(args.output_dir)

    # Generate synthetic data
    data_paths = generate_synthetic_data(dirs, verbose=args.verbose)

    # Create configuration
    config = create_config(data_paths, dirs, args.sector)

    # Run pipeline
    results = run_pipeline(config, args.sector, args.pulse_year, verbose=args.verbose)

    # Print summary
    print_summary(results, args.output_dir)


if __name__ == "__main__":
    main()
