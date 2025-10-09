"""
Complete run_ssps workflow example.

This script demonstrates the full workflow equivalent to run_ssps from
dscim-testing/run_integration_result.py, including:
1. Generate synthetic climate data
2. Reduce damages for sectors
3. Generate damage functions
4. Calculate SCC with multiple discounting methods

This is a complete end-to-end example showing all new functionality.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dscim_new.config.schemas import (
    DSCIMConfig,
    ClimateDataConfig,
    DamageFunctionConfig,
    DiscountingConfig,
    SCCConfig,
)
from dscim_new.utils import ClimateDataGenerator
from dscim_new.pipeline import (
    ReduceDamagesStep,
    GenerateDamageFunctionStep,
    CalculateSCCStep,
)

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    console = Console()
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    console = None


def print_section(title: str):
    """Print a section header."""
    if RICH_AVAILABLE:
        console.print(f"\n[bold blue]{'='*60}[/bold blue]")
        console.print(f"[bold blue]{title}[/bold blue]")
        console.print(f"[bold blue]{'='*60}[/bold blue]\n")
    else:
        print(f"\n{'='*60}")
        print(title)
        print(f"{'='*60}\n")


def main():
    """Run complete DSCIM workflow: reduce damages -> damage functions -> SCC."""

    if RICH_AVAILABLE:
        console.print(Panel(
            "[bold green]DSCIM Complete Workflow Example[/bold green]\n"
            "Equivalent to run_ssps from dscim-testing",
            style="green"
        ))

    # =========================================================================
    # STEP 1: Generate Synthetic Climate Data
    # =========================================================================
    print_section("STEP 1: Generate Synthetic Climate Data")

    climate_dir = Path("examples/workflow_output/climate_data")
    climate_dir.mkdir(parents=True, exist_ok=True)

    generator = ClimateDataGenerator(seed=42, verbose=True)
    climate_paths = generator.generate_all_climate_data(str(climate_dir))

    print("\n Climate data generated successfully!")
    for key, path in climate_paths.items():
        print(f"  - {key}: {path}")

    # =========================================================================
    # STEP 2: Setup Configuration
    # =========================================================================
    print_section("STEP 2: Setup Configuration")

    # Note: Using existing config as base, but adding new sections
    config_path = "examples/configs/full_config.yaml"
    print(f"Loading base configuration from: {config_path}")

    # Load base config
    try:
        base_config = DSCIMConfig.from_yaml(config_path)
    except FileNotFoundError:
        print(f"Config file not found at {config_path}")
        print("Please run simple_run_ssps.py instead, or ensure config file exists")
        return

    # Add climate data configuration
    base_config.climate_data = ClimateDataConfig(
        gmst_path=climate_paths["gmst"],
        gmsl_path=climate_paths["gmsl"],
        fair_temperature_path=climate_paths["fair_temperature"],
        fair_gmsl_path=climate_paths["fair_gmsl"],
        pulse_conversion_path=climate_paths["pulse_conversion"],
    )

    # Add damage function configuration
    base_config.damage_function = DamageFunctionConfig(
        formula="damages ~ -1 + anomaly + np.power(anomaly, 2)",
        fit_type="ols",
        extrapolation_method="global_c_ratio",
        save_points=True,
        n_points=100,
    )

    # Add discounting configurations (multiple methods like original run_ssps)
    base_config.discounting = [
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

    # Add SCC configuration
    base_config.scc = SCCConfig(
        pulse_years=[2020],
        fair_aggregation="mean",
        calculate_quantiles=True,
        quantile_levels=[0.05, 0.25, 0.5, 0.75, 0.95],
        save_discount_factors=True,
    )

    # Update output paths
    base_config.paths.ssp_damage_function_library = "examples/workflow_output/damage_functions"
    base_config.paths.AR6_ssp_results = "examples/workflow_output/scc_results"

    print("✓ Configuration complete")
    print(f"  - Sectors: {list(base_config.sectors.keys())}")
    print(f"  - Pulse years: {base_config.scc.pulse_years}")
    print(f"  - Discounting methods: {[d.discount_type for d in base_config.discounting]}")

    # =========================================================================
    # STEP 3: Process Each Sector
    # =========================================================================
    print_section("STEP 3: Run Full Pipeline for Each Sector")

    # For this example, we'll process one sector
    # In full workflow, you'd loop over all sectors
    sectors_to_process = ["mortality"]  # Can extend to list(base_config.sectors.keys())
    pulse_year = 2020
    recipe = "adding_up"
    reduction = "cc"

    for sector in sectors_to_process:
        if RICH_AVAILABLE:
            console.print(Panel(
                f"Processing sector: [bold]{sector}[/bold]",
                style="cyan"
            ))

        # ---------------------------------------------------------------------
        # Step 3a: Reduce Damages
        # ---------------------------------------------------------------------
        print(f"\n[3a] Reducing damages for {sector}...")

        reduce_step = ReduceDamagesStep(
            config=base_config,
            sector=sector,
            recipe=recipe,
            reduction=reduction,
            verbose=True,
        )

        # Run reduce damages
        reduce_outputs = reduce_step.run(inputs={}, save=True)
        reduced_damages = reduce_outputs["reduced_damages"]

        print(f"✓ Reduced damages shape: {reduced_damages.shape}")
        print(f"  Dimensions: {list(reduced_damages.dims)}")

        # ---------------------------------------------------------------------
        # Step 3b: Generate Damage Functions
        # ---------------------------------------------------------------------
        print(f"\n[3b] Generating damage function for {sector}...")

        df_step = GenerateDamageFunctionStep(
            config=base_config,
            sector=sector,
            pulse_year=pulse_year,
            verbose=True,
        )

        # Run damage function generation
        df_outputs = df_step.run(
            inputs={"reduced_damages": reduced_damages},
            save=True
        )

        coefficients = df_outputs["damage_function_coefficients"]
        marginal_damages = df_outputs["marginal_damages"]

        print(f"✓ Damage function generated")
        print(f"  Coefficients: {list(coefficients.coefficient.values)}")
        print(f"  Marginal damages shape: {marginal_damages.shape}")

        # ---------------------------------------------------------------------
        # Step 3c: Calculate SCC for Each Discounting Method
        # ---------------------------------------------------------------------
        print(f"\n[3c] Calculating SCC for {sector}...")

        # We'll need consumption data for Ramsey/GWR
        # For this example, we'll load from the economic data
        import xarray as xr
        econ_data = xr.open_zarr(base_config.econdata.global_ssp, chunks=None)

        # Get consumption (gdppc * pop) or use gdppc as proxy
        if 'gdppc' in econ_data.data_vars:
            consumption = econ_data['gdppc']
        else:
            raise ValueError("Economic data must contain 'gdppc' variable")

        scc_results = {}

        for discount_idx, discount_config in enumerate(base_config.discounting):
            discount_name = discount_config.discount_type
            print(f"\n  Calculating SCC with {discount_name} discounting...")

            scc_step = CalculateSCCStep(
                config=base_config,
                sector=sector,
                pulse_year=pulse_year,
                discount_config_index=discount_idx,
                verbose=True,
            )

            # Run SCC calculation
            scc_outputs = scc_step.run(
                inputs={
                    "marginal_damages": marginal_damages,
                    "consumption": consumption,
                },
                save=True
            )

            scc = scc_outputs["scc"]
            scc_mean = float(scc.mean().values)
            scc_median = float(scc.median().values)

            scc_results[discount_name] = {
                "mean": scc_mean,
                "median": scc_median,
                "data": scc,
            }

            print(f"    ✓ Mean SCC: ${scc_mean:.2f}/tCO2")
            print(f"    ✓ Median SCC: ${scc_median:.2f}/tCO2")

    # =========================================================================
    # STEP 4: Summary Results
    # =========================================================================
    print_section("STEP 4: Summary Results")

    if RICH_AVAILABLE:
        # Create results table
        table = Table(title="SCC Results Summary", show_header=True)
        table.add_column("Sector", style="cyan")
        table.add_column("Pulse Year", style="magenta")
        table.add_column("Discount Method", style="yellow")
        table.add_column("Mean SCC ($/tCO2)", style="green")
        table.add_column("Median SCC ($/tCO2)", style="green")

        for discount_name, results in scc_results.items():
            table.add_row(
                sector,
                str(pulse_year),
                discount_name,
                f"${results['mean']:.2f}",
                f"${results['median']:.2f}"
            )

        console.print(table)
    else:
        print("\nSCC Results:")
        print(f"Sector: {sector}, Pulse Year: {pulse_year}")
        for discount_name, results in scc_results.items():
            print(f"  {discount_name}:")
            print(f"    Mean SCC: ${results['mean']:.2f}/tCO2")
            print(f"    Median SCC: ${results['median']:.2f}/tCO2")

    # =========================================================================
    # Output Locations
    # =========================================================================
    print("\n" + "="*60)
    print("Output Locations:")
    print("="*60)
    print(f"  Climate data: {climate_dir}")
    print(f"  Reduced damages: {base_config.paths.reduced_damages_library}")
    print(f"  Damage functions: {base_config.paths.ssp_damage_function_library}")
    print(f"  SCC results: {base_config.paths.AR6_ssp_results}")

    if RICH_AVAILABLE:
        console.print("\n[bold green]✓ Workflow complete![/bold green]")
    else:
        print("\n✓ Workflow complete!")


if __name__ == "__main__":
    main()
