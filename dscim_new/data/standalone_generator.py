"""
Standalone Synthetic Data Generator

This script generates synthetic DSCIM test data without requiring any
configuration files. It can be used independently with custom parameters.
"""

import argparse
import logging
from pathlib import Path
import sys

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from synthetic_data import (
    SyntheticDataGenerator,
    DataGenerationConfig
)

logger = logging.getLogger(__name__)


def setup_logging(level=logging.INFO):
    """Setup logging configuration"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def create_quick_config(
    seed=42,
    start_year=2020,
    end_year=2025,
    extrap_end_year=2050,
    n_simulations=2,
    n_batches=5
):
    """Create a quick configuration for testing"""
    return DataGenerationConfig(
        seed=seed,
        years=list(range(start_year, end_year + 1)),
        years_extrap=list(range(2001, extrap_end_year + 1)),
        pulse_years=[start_year],
        rcps=['dummy1', 'dummy2'],
        gcms=['dummy1', 'dummy2'],
        gases=['dummy_gas'],
        slrs=[0, 1],
        simulations=list(range(n_simulations)),
        ssps=['ssp1', 'ssp2', 'ssp3'],
        regions=['dummy1', 'dummy2'],
        models=['dummy1', 'dummy2'],
        batches=list(range(n_batches))
    )


def create_full_config(seed=42):
    """Create a full configuration similar to the original structure"""
    return DataGenerationConfig(
        seed=seed,
        years=list(range(2020, 2031)),
        years_extrap=list(range(2001, 2100)),
        pulse_years=[2020],
        rcps=['dummy1', 'dummy2'],
        gcms=['dummy1', 'dummy2'],
        gases=['dummy_gas'],
        slrs=[0, 1],
        simulations=list(range(4)),
        ssps=['ssp1', 'ssp2', 'ssp3', 'ssp4', 'ssp5'],
        regions=['dummy1', 'dummy2'],
        models=['dummy1', 'dummy2'],
        batches=list(range(15))
    )


def generate_synthetic_data(
    output_dir="standalone_synthetic_data",
    config_type="quick",
    seed=42,
    **kwargs
):
    """
    Generate synthetic data standalone

    Parameters
    ----------
    output_dir : str
        Output directory for generated data
    config_type : str
        Type of configuration: 'quick' or 'full'
    seed : int
        Random seed for reproducibility
    **kwargs
        Additional parameters for quick config
    """

    setup_logging()

    logger.info("=" * 60)
    logger.info("Standalone Synthetic Data Generation")
    logger.info("=" * 60)

    if config_type == "quick":
        config = create_quick_config(seed=seed, **kwargs)
        logger.info("Using quick configuration (smaller datasets)")
    elif config_type == "full":
        config = create_full_config(seed=seed)
        logger.info("Using full configuration (complete datasets)")
    else:
        raise ValueError(f"Unknown config_type: {config_type}")

    generator = SyntheticDataGenerator(config)

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    subdirs = ['climate', 'sectoral', 'econ']
    for subdir in subdirs:
        (output_path / subdir).mkdir(exist_ok=True)

    logger.info(f"Output directory: {output_path.absolute()}")
    logger.info(f"Configuration: {config_type} (seed={seed})")

    try:
        logger.info("Generating climate data...")

        gmst_df = generator.create_gmst_csv()
        gmst_path = output_path / "climate" / "gmst.csv"
        gmst_df.to_csv(gmst_path, index=False)
        logger.info(f"  > GMST CSV: {gmst_path} ({len(gmst_df)} rows)")

        gmsl_ds = generator.create_gmsl_zarr()
        gmsl_path = output_path / "climate" / "gmsl.zarr"
        gmsl_ds.to_zarr(gmsl_path, mode='w')
        logger.info(f"  > GMSL zarr: {gmsl_path}")

        fair_temp_ds = generator.create_fair_temperature_nc4()
        fair_temp_path = output_path / "climate" / "fair_temps.nc4"
        fair_temp_ds.to_netcdf(fair_temp_path)
        logger.info(f"  > FAIR temperature: {fair_temp_path}")

        fair_gmsl_ds = generator.create_fair_gmsl_nc4()
        fair_gmsl_path = output_path / "climate" / "fair_gmsl.nc4"
        fair_gmsl_ds.to_netcdf(fair_gmsl_path)
        logger.info(f"  > FAIR GMSL: {fair_gmsl_path}")

        conversion_ds = generator.create_conversion_nc4()
        conversion_path = output_path / "climate" / "conversion.nc4"
        conversion_ds.to_netcdf(conversion_path)
        logger.info(f"  > Conversion factors: {conversion_path}")

        logger.info("Generating economic data...")
        econ_ds = generator.create_economic_zarr()
        econ_path = output_path / "econ" / "economic_data.zarr"
        econ_ds.to_zarr(econ_path, mode='w')
        logger.info(f"  > Economic data: {econ_path}")

        logger.info("Generating sectoral data...")

        non_coastal_ds = generator.create_sectoral_damages_zarr(
            delta_name="delta_dummy",
            histclim_name="histclim_dummy",
            coastal=False
        )
        non_coastal_path = output_path / "sectoral" / "non_coastal_damages.zarr"
        non_coastal_ds.to_zarr(non_coastal_path, mode='w')
        logger.info(f"  > Non-coastal damages: {non_coastal_path}")

        coastal_ds = generator.create_sectoral_damages_zarr(
            delta_name="delta_dummy",
            histclim_name="histclim_dummy",
            coastal=True
        )
        coastal_path = output_path / "sectoral" / "coastal_damages.zarr"
        coastal_ds.to_zarr(coastal_path, mode='w')
        logger.info(f"  > Coastal damages: {coastal_path}")

        logger.info("=" * 60)
        logger.info("SUCCESS: Standalone synthetic data generation completed")
        logger.info(f"Output directory: {output_path.absolute()}")
        logger.info("=" * 60)

        logger.info("Generated files:")
        total_size = 0
        for file_path in sorted(output_path.rglob("*")):
            if file_path.is_file():
                size_mb = file_path.stat().st_size / (1024 * 1024)
                total_size += size_mb
                rel_path = file_path.relative_to(output_path)
                logger.info(f"  {rel_path} ({size_mb:.2f} MB)")

        logger.info(f"Total size: {total_size:.2f} MB")

        return {
            'output_dir': str(output_path.absolute()),
            'config': config,
            'total_size_mb': total_size
        }

    except Exception as e:
        logger.error(f"Error during data generation: {e}")
        raise


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(
        description="Generate synthetic DSCIM test data (standalone)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick generation (small datasets for testing)
  python standalone_generator.py --quick

  # Full generation (complete datasets)
  python standalone_generator.py --full

  # Custom output directory
  python standalone_generator.py --output my_data --quick

  # Custom parameters for quick mode
  python standalone_generator.py --quick --start-year 2015 --end-year 2020

  # Different random seed
  python standalone_generator.py --quick --seed 123
        """
    )

    parser.add_argument("--output", "-o", default="standalone_synthetic_data",
                       help="Output directory (default: standalone_synthetic_data)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")

    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument("--quick", action="store_true",
                             help="Generate quick/small datasets for testing")
    config_group.add_argument("--full", action="store_true",
                             help="Generate full/complete datasets")

    quick_group = parser.add_argument_group("quick mode options")
    quick_group.add_argument("--start-year", type=int, default=2020,
                            help="Start year for quick mode (default: 2020)")
    quick_group.add_argument("--end-year", type=int, default=2025,
                            help="End year for quick mode (default: 2025)")
    quick_group.add_argument("--extrap-end-year", type=int, default=2050,
                            help="Extrapolation end year for quick mode (default: 2050)")
    quick_group.add_argument("--n-simulations", type=int, default=2,
                            help="Number of simulations for quick mode (default: 2)")
    quick_group.add_argument("--n-batches", type=int, default=5,
                            help="Number of batches for quick mode (default: 5)")

    args = parser.parse_args()

    if args.quick:
        config_type = "quick"
        kwargs = {
            'start_year': args.start_year,
            'end_year': args.end_year,
            'extrap_end_year': args.extrap_end_year,
            'n_simulations': args.n_simulations,
            'n_batches': args.n_batches
        }
    else:
        config_type = "full"
        kwargs = {}

    result = generate_synthetic_data(
        output_dir=args.output,
        config_type=config_type,
        seed=args.seed,
        **kwargs
    )

    print(f"\nGeneration complete! Data saved to: {result['output_dir']}")
    print(f"Total size: {result['total_size_mb']:.2f} MB")


if __name__ == "__main__":
    main()