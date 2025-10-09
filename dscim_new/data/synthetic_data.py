"""
Synthetic Data Generators for DSCIM Testing

This module provides deterministic synthetic data generators that match
the exact structure used in DSCIM workflows, ensuring compatibility with
existing pipelines while providing reproducible test data.
"""

import numpy as np
import pandas as pd
import xarray as xr
from typing import List, Dict, Optional, Any
from pathlib import Path
import yaml
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DataGenerationConfig:
    """Configuration for synthetic data generation"""
    seed: int = 42

    # Time ranges
    years: List[int] = None
    years_extrap: List[int] = None
    pulse_years: List[int] = None

    # Climate scenarios
    rcps: List[str] = None
    gcms: List[str] = None
    gases: List[str] = None
    slrs: List[int] = None
    simulations: List[int] = None

    # Economic scenarios
    ssps: List[str] = None
    regions: List[str] = None
    models: List[str] = None
    batches: List[int] = None

    def __post_init__(self):
        """Set defaults for None values"""
        if self.years is None:
            self.years = list(range(2020, 2031))
        if self.years_extrap is None:
            self.years_extrap = list(range(2001, 2100))
        if self.pulse_years is None:
            self.pulse_years = [2020]
        if self.rcps is None:
            self.rcps = ['dummy1', 'dummy2']
        if self.gcms is None:
            self.gcms = ['dummy1', 'dummy2']
        if self.gases is None:
            self.gases = ['dummy_gas']
        if self.slrs is None:
            self.slrs = [0, 1]
        if self.simulations is None:
            self.simulations = list(range(4))
        if self.ssps is None:
            self.ssps = ['ssp1', 'ssp2', 'ssp3', 'ssp4', 'ssp5']
        if self.regions is None:
            self.regions = ['dummy1', 'dummy2']
        if self.models is None:
            self.models = ['dummy1', 'dummy2']
        if self.batches is None:
            self.batches = list(range(15))


class SyntheticDataGenerator:
    """Synthetic data generator matching DSCIM structure"""

    def __init__(self, config: DataGenerationConfig):
        self.config = config
        self.rng = np.random.RandomState(config.seed)

    def reset_seed(self):
        """Reset random number generator"""
        self.rng = np.random.RandomState(self.config.seed)

    def create_gmst_csv(self) -> pd.DataFrame:
        """
        Create GMST CSV file with structure: year, rcp, gcm, anomaly
        """
        self.reset_seed()

        data = []
        for year in self.config.years:
            for rcp in self.config.rcps:
                for gcm in self.config.gcms:
                    base_anomaly = self.rng.uniform(0, 10)
                    data.append({
                        'year': year,
                        'rcp': rcp,
                        'gcm': gcm,
                        'anomaly': base_anomaly
                    })

        return pd.DataFrame(data)

    def create_gmsl_zarr(self) -> xr.Dataset:
        """
        Create GMSL zarr file with coordinates: year, slr; variables: gmsl
        """
        self.reset_seed()

        gmsls = self.rng.uniform(0, 10, len(self.config.years))

        data = []
        for i, year in enumerate(self.config.years):
            for slr in self.config.slrs:
                data.append({
                    'year': year,
                    'slr': slr,
                    'gmsl': gmsls[i]
                })

        df = pd.DataFrame(data)
        df_indexed = df.set_index(['year', 'slr'])
        return df_indexed.to_xarray()

    def create_fair_temperature_nc4(self) -> xr.Dataset:
        """
        Create FAIR temperature NC4 file with control/pulse scenarios
        """
        self.reset_seed()

        n_years = len(self.config.years_extrap)
        n_rcps = len(self.config.rcps)
        n_sims = len(self.config.simulations)
        n_gases = len(self.config.gases)
        n_pulse_years = len(self.config.pulse_years)

        control_temps = self.rng.uniform(0, 10, (n_years, n_rcps, n_sims, n_gases))
        pulse_temps = self.rng.uniform(0, 10, (n_years, n_rcps, n_sims, n_gases, n_pulse_years))
        medianparams_control_temps = self.rng.uniform(0, 10, (n_years, n_rcps, n_gases))
        medianparams_pulse_temps = self.rng.uniform(0, 10, (n_years, n_rcps, n_gases, n_pulse_years))

        data_vars = {
            'control_temperature': (['year', 'rcp', 'simulation', 'gas'], control_temps),
            'pulse_temperature': (['year', 'rcp', 'simulation', 'gas', 'pulse_year'], pulse_temps),
            'medianparams_control_temperature': (['year', 'rcp', 'gas'], medianparams_control_temps),
            'medianparams_pulse_temperature': (['year', 'rcp', 'gas', 'pulse_year'], medianparams_pulse_temps)
        }

        coords = {
            'year': self.config.years_extrap,
            'rcp': self.config.rcps,
            'simulation': self.config.simulations,
            'gas': self.config.gases,
            'pulse_year': self.config.pulse_years
        }

        return xr.Dataset(data_vars=data_vars, coords=coords)

    def create_fair_gmsl_nc4(self) -> xr.Dataset:
        """
        Create FAIR GMSL NC4 file with control/pulse scenarios
        """
        self.reset_seed()

        n_years = len(self.config.years_extrap)
        n_rcps = len(self.config.rcps)
        n_sims = len(self.config.simulations)
        n_gases = len(self.config.gases)
        n_pulse_years = len(self.config.pulse_years)

        control_gmsl = self.rng.uniform(0, 10, (n_years, n_rcps, n_sims, n_gases))
        pulse_gmsl = self.rng.uniform(0, 10, (n_years, n_rcps, n_sims, n_gases, n_pulse_years))
        medianparams_control_gmsl = self.rng.uniform(0, 10, (n_years, n_rcps, n_gases))
        medianparams_pulse_gmsl = self.rng.uniform(0, 10, (n_years, n_rcps, n_gases, n_pulse_years))

        data_vars = {
            'control_gmsl': (['year', 'rcp', 'simulation', 'gas'], control_gmsl),
            'pulse_gmsl': (['year', 'rcp', 'simulation', 'gas', 'pulse_years'], pulse_gmsl),
            'medianparams_control_gmsl': (['year', 'rcp', 'gas'], medianparams_control_gmsl),
            'medianparams_pulse_gmsl': (['year', 'rcp', 'gas', 'pulse_years'], medianparams_pulse_gmsl)
        }

        coords = {
            'year': self.config.years_extrap,
            'rcp': self.config.rcps,
            'simulation': self.config.simulations,
            'gas': self.config.gases,
            'pulse_years': self.config.pulse_years
        }

        return xr.Dataset(data_vars=data_vars, coords=coords)

    def create_conversion_nc4(self) -> xr.Dataset:
        """
        Create conversion factors NC4 file
        """
        self.reset_seed()
        data_vars = {'emissions': (['gas'], [0.1] * len(self.config.gases))}
        coords = {'gas': self.config.gases}
        return xr.Dataset(data_vars=data_vars, coords=coords)

    def create_economic_zarr(self) -> xr.Dataset:
        """
        Create economic zarr file with coordinates: year, ssp, region, model
        """
        self.reset_seed()

        n_years = len(self.config.years)
        n_ssps = len(self.config.ssps)
        n_regions = len(self.config.regions)
        n_models = len(self.config.models)

        pop = self.rng.uniform(20, 100, (n_years, n_ssps, n_regions, n_models))
        gdppc = self.rng.uniform(50, 100, (n_years, n_ssps, n_regions, n_models))
        gdp = pop * gdppc

        data_vars = {
            'pop': (('year', 'ssp', 'region', 'model'), pop),
            'gdppc': (('year', 'ssp', 'region', 'model'), gdppc),
            'gdp': (('year', 'ssp', 'region', 'model'), gdp)
        }

        coords = {
            'year': self.config.years,
            'ssp': self.config.ssps,
            'region': self.config.regions,
            'model': self.config.models
        }

        return xr.Dataset(data_vars=data_vars, coords=coords)

    def create_sectoral_damages_zarr(self,
                                   delta_name: str = "delta_dummy",
                                   histclim_name: str = "histclim_dummy",
                                   coastal: bool = False) -> xr.Dataset:
        """
        Create sectoral damages zarr file with appropriate coordinate structure
        """
        self.reset_seed()

        if coastal:
            shape = (len(self.config.regions), len(self.config.years),
                    len(self.config.batches), len(self.config.slrs),
                    len(self.config.models), len(self.config.ssps))

            delta_data = self.rng.uniform(5, 15, shape)
            histclim_data = self.rng.uniform(1, 10, shape)

            data_vars = {
                delta_name: (('region', 'year', 'batch', 'slr', 'model', 'ssp'), delta_data),
                histclim_name: (('region', 'year', 'batch', 'slr', 'model', 'ssp'), histclim_data)
            }

            coords = {
                'region': self.config.regions,
                'year': self.config.years,
                'batch': self.config.batches,
                'slr': self.config.slrs,
                'model': self.config.models,
                'ssp': self.config.ssps
            }

            ds = xr.Dataset(data_vars=data_vars, coords=coords)
            return ds.chunk({'region': 1, 'slr': 1, 'year': 1, 'model': 1, 'ssp': 1, 'batch': -1})

        else:
            shape = (len(self.config.rcps), len(self.config.regions),
                    len(self.config.gcms), len(self.config.years),
                    len(self.config.models), len(self.config.ssps),
                    len(self.config.batches))

            delta_data = self.rng.uniform(5, 15, shape)
            histclim_data = self.rng.uniform(1, 10, shape)

            data_vars = {
                delta_name: (('rcp', 'region', 'gcm', 'year', 'model', 'ssp', 'batch'), delta_data),
                histclim_name: (('rcp', 'region', 'gcm', 'year', 'model', 'ssp', 'batch'), histclim_data)
            }

            coords = {
                'rcp': self.config.rcps,
                'region': self.config.regions,
                'gcm': self.config.gcms,
                'year': self.config.years,
                'model': self.config.models,
                'ssp': self.config.ssps,
                'batch': self.config.batches
            }

            ds = xr.Dataset(data_vars=data_vars, coords=coords)
            return ds.chunk({'batch': -1})


def create_test_data_from_config(config_path: str, output_dir: str = None):
    """
    Create synthetic test data matching structure from YAML configuration

    Parameters
    ----------
    config_path : str
        Path to YAML configuration file
    output_dir : str, optional
        Output directory for data files
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    data_config = DataGenerationConfig(
        years=list(range(2020, 2031)),
        years_extrap=list(range(2001, 2100)),
        pulse_years=[2020],
        rcps=['dummy1', 'dummy2'],
        gcms=['dummy1', 'dummy2'],
        gases=config['AR6_ssp_climate']['gases'],
        ssps=['ssp1', 'ssp2', 'ssp3', 'ssp4', 'ssp5'],
        regions=['dummy1', 'dummy2'],
        models=['dummy1', 'dummy2']
    )

    generator = SyntheticDataGenerator(data_config)

    if output_dir:
        base_dir = Path(output_dir)
    else:
        base_dir = Path('.')

    dirs_to_create = ['dummy_data/climate', 'dummy_data/sectoral', 'dummy_data/econ']
    for dir_path in dirs_to_create:
        (base_dir / dir_path).mkdir(parents=True, exist_ok=True)

    logger.info("Generating synthetic data files...")

    # Climate data
    gmst_df = generator.create_gmst_csv()
    gmst_path = config['AR6_ssp_climate']['gmst_path']
    if output_dir:
        gmst_path = str(base_dir / gmst_path)
    gmst_df.to_csv(gmst_path, index=False)

    gmsl_ds = generator.create_gmsl_zarr()
    gmsl_path = config['AR6_ssp_climate']['gmsl_path']
    if output_dir:
        gmsl_path = str(base_dir / gmsl_path)
    gmsl_ds.to_zarr(gmsl_path, mode='w')

    fair_temp_ds = generator.create_fair_temperature_nc4()
    fair_temp_path = config['AR6_ssp_climate']['gmst_fair_path']
    if output_dir:
        fair_temp_path = str(base_dir / fair_temp_path)
    fair_temp_ds.to_netcdf(fair_temp_path)

    fair_gmsl_ds = generator.create_fair_gmsl_nc4()
    fair_gmsl_path = config['AR6_ssp_climate']['gmsl_fair_path']
    if output_dir:
        fair_gmsl_path = str(base_dir / fair_gmsl_path)
    fair_gmsl_ds.to_netcdf(fair_gmsl_path)

    conversion_ds = generator.create_conversion_nc4()
    conversion_path = config['AR6_ssp_climate']['damages_pulse_conversion_path']
    if output_dir:
        conversion_path = str(base_dir / conversion_path)
    conversion_ds.to_netcdf(conversion_path)

    # Economic data
    econ_ds = generator.create_economic_zarr()
    econ_path = config['econdata']['global_ssp']
    if output_dir:
        econ_path = str(base_dir / econ_path)
    econ_ds.to_zarr(econ_path, mode='w')

    # Sectoral data
    for sector_name, sector_config in config['sectors'].items():
        is_coastal = 'coastal' in sector_name.lower()

        sector_ds = generator.create_sectoral_damages_zarr(
            delta_name=sector_config['delta'],
            histclim_name=sector_config['histclim'],
            coastal=is_coastal
        )

        sector_path = sector_config['sector_path']
        if output_dir:
            sector_path = str(base_dir / sector_path)

        sector_ds.to_zarr(sector_path, mode='w')

    logger.info("Synthetic data generation complete")

    return {
        'config': config,
        'data_config': data_config,
        'output_paths': {
            'gmst': gmst_path,
            'gmsl': gmsl_path,
            'fair_temp': fair_temp_path,
            'fair_gmsl': fair_gmsl_path,
            'conversion': conversion_path,
            'economic': econ_path,
            'sectors': {name: cfg['sector_path'] for name, cfg in config['sectors'].items()}
        }
    }