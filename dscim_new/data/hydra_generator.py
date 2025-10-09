"""
Hydra-based Synthetic Data Generator

This script uses Hydra for configuration management and generates synthetic
test data matching the exact structure expected by DSCIM.
"""

import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from pathlib import Path
import sys

# Add parent directory to path for imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(current_dir))

try:
    from config.data_generation_config import (
        DataGenerationConfig as HydraDataGenerationConfig,
        create_default_config,
        create_testing_config
    )
    from synthetic_data import (
        SyntheticDataGenerator,
        DataGenerationConfig as OriginalDataGenerationConfig
    )
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current directory: {current_dir}")
    print(f"Parent directory: {parent_dir}")
    print(f"Python path: {sys.path[:3]}")
    raise

logger = logging.getLogger(__name__)


class HydraSyntheticDataGenerator:
    """Hydra-powered synthetic data generator"""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.setup_logging()

    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def create_data_config(self) -> OriginalDataGenerationConfig:
        """Convert Hydra config to OriginalDataGenerationConfig"""
        data_cfg = self.cfg.data_generation

        config = OriginalDataGenerationConfig(
            seed=data_cfg.seed,
            years=list(range(data_cfg.time.start_year, data_cfg.time.end_year + 1)),
            years_extrap=list(range(data_cfg.time.extrap_start_year, data_cfg.time.extrap_end_year + 1)),
            pulse_years=data_cfg.time.pulse_years,
            rcps=data_cfg.climate.rcps,
            gcms=data_cfg.climate.gcms,
            gases=data_cfg.climate.gases,
            slrs=data_cfg.climate.slrs,
            simulations=list(range(data_cfg.climate.n_simulations)),
            ssps=data_cfg.economic.ssps,
            regions=data_cfg.economic.regions,
            models=data_cfg.economic.models,
            batches=list(range(data_cfg.economic.n_batches))
        )

        return config

    def generate_data(self):
        """Generate synthetic data based on configuration"""
        logger.info("Starting synthetic data generation with Hydra configuration")

        data_config = self.create_data_config()
        generator = SyntheticDataGenerator(data_config)

        output_dir = Path(self.cfg.data_generation.paths.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        subdirs = ['climate', 'sectoral', 'econ']
        for subdir in subdirs:
            (output_dir / subdir).mkdir(exist_ok=True)

        logger.info(f"Output directory: {output_dir}")

        self._generate_climate_data(generator, output_dir)
        self._generate_economic_data(generator, output_dir)
        self._generate_sectoral_data(generator, output_dir)

        logger.info("Synthetic data generation completed successfully")

        return {
            'output_dir': str(output_dir),
            'config': OmegaConf.to_yaml(self.cfg),
            'data_config': data_config
        }

    def _generate_climate_data(self, generator: SyntheticDataGenerator, output_dir: Path):
        """Generate climate-related data"""
        logger.info("Generating climate data...")

        paths = self.cfg.data_generation.paths

        gmst_df = generator.create_gmst_csv()
        gmst_path = output_dir / paths.gmst_path
        gmst_path.parent.mkdir(parents=True, exist_ok=True)
        gmst_df.to_csv(gmst_path, index=False)
        logger.info(f"  > GMST CSV: {gmst_path}")

        gmsl_ds = generator.create_gmsl_zarr()
        gmsl_path = output_dir / paths.gmsl_path
        gmsl_path.parent.mkdir(parents=True, exist_ok=True)
        gmsl_ds.to_zarr(gmsl_path, mode='w')
        logger.info(f"  > GMSL zarr: {gmsl_path}")

        fair_temp_ds = generator.create_fair_temperature_nc4()
        fair_temp_path = output_dir / paths.gmst_fair_path
        fair_temp_path.parent.mkdir(parents=True, exist_ok=True)
        fair_temp_ds.to_netcdf(fair_temp_path)
        logger.info(f"  > FAIR temperature NC4: {fair_temp_path}")

        fair_gmsl_ds = generator.create_fair_gmsl_nc4()
        fair_gmsl_path = output_dir / paths.gmsl_fair_path
        fair_gmsl_path.parent.mkdir(parents=True, exist_ok=True)
        fair_gmsl_ds.to_netcdf(fair_gmsl_path)
        logger.info(f"  > FAIR GMSL NC4: {fair_gmsl_path}")

        conversion_ds = generator.create_conversion_nc4()
        conversion_path = output_dir / paths.conversion_path
        conversion_path.parent.mkdir(parents=True, exist_ok=True)
        conversion_ds.to_netcdf(conversion_path)
        logger.info(f"  > Conversion NC4: {conversion_path}")

    def _generate_economic_data(self, generator: SyntheticDataGenerator, output_dir: Path):
        """Generate economic data"""
        logger.info("Generating economic data...")

        paths = self.cfg.data_generation.paths

        econ_ds = generator.create_economic_zarr()
        econ_path = output_dir / paths.economic_path
        econ_path.parent.mkdir(parents=True, exist_ok=True)
        econ_ds.to_zarr(econ_path, mode='w')
        logger.info(f"  > Economic zarr: {econ_path}")

    def _generate_sectoral_data(self, generator: SyntheticDataGenerator, output_dir: Path):
        """Generate sectoral damage data"""
        logger.info("Generating sectoral data...")

        for sector_name, sector_config in self.cfg.data_generation.sectors.items():
            logger.info(f"  Generating {sector_name} sector data...")

            sector_ds = generator.create_sectoral_damages_zarr(
                delta_name=sector_config.delta_variable,
                histclim_name=sector_config.histclim_variable,
                coastal=sector_config.is_coastal
            )

            sector_path = output_dir / sector_config.sector_path
            sector_path.parent.mkdir(parents=True, exist_ok=True)
            sector_ds.to_zarr(sector_path, mode='w')
            logger.info(f"    > {sector_name} zarr: {sector_path}")

    def validate_generated_data(self):
        """Validate the generated data structure"""
        if not self.cfg.testing.validate_data_structure:
            return True

        logger.info("Validating generated data structure...")

        output_dir = Path(self.cfg.data_generation.paths.output_dir)
        paths = self.cfg.data_generation.paths

        expected_files = [
            paths.gmst_path,
            paths.gmsl_path,
            paths.gmst_fair_path,
            paths.gmsl_fair_path,
            paths.conversion_path,
            paths.economic_path
        ]

        for sector_config in self.cfg.data_generation.sectors.values():
            expected_files.append(sector_config.sector_path)

        all_exist = True
        for file_path in expected_files:
            full_path = output_dir / file_path
            if not full_path.exists():
                logger.error(f"  x Missing file: {full_path}")
                all_exist = False
            else:
                logger.info(f"  > Found file: {full_path}")

        if all_exist:
            logger.info("Data structure validation passed")
        else:
            logger.error("Data structure validation failed")

        return all_exist


@hydra.main(version_base=None, config_path="../config/conf", config_name="config")
def generate_synthetic_data(cfg: DictConfig) -> None:
    """Main function for generating synthetic data with Hydra"""

    logger.info("=" * 60)
    logger.info("DSCIM Synthetic Data Generation with Hydra")
    logger.info("=" * 60)

    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))

    generator = HydraSyntheticDataGenerator(cfg)

    try:
        result = generator.generate_data()

        if cfg.testing.validate_data_structure:
            validation_passed = generator.validate_generated_data()
            if not validation_passed:
                logger.error("Data validation failed")
                return

        logger.info("=" * 60)
        logger.info("SUCCESS: Synthetic data generation completed")
        logger.info(f"Output directory: {result['output_dir']}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Error during data generation: {e}")
        raise


@hydra.main(version_base=None, config_path="../config/conf", config_name="config")
def validate_existing_data(cfg: DictConfig) -> None:
    """Validate existing synthetic data"""
    generator = HydraSyntheticDataGenerator(cfg)
    generator.validate_generated_data()


if __name__ == "__main__":
    generate_synthetic_data()