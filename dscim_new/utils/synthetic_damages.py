"""
Synthetic damages data generation for testing and examples.

Generates economic data and sectoral damages (coastal and non-coastal) 
with realistic structure matching DSCIM requirements.
"""

import numpy as np
import xarray as xr
from pathlib import Path
from typing import Optional, List, Dict


class DamagesDataGenerator:
    """
    Generate synthetic damages data for DSCIM workflows.

    Creates economic data (population, GDP), non-coastal sectoral damages 
    (linked to temperature via gcm/rcp), and coastal sectoral damages 
    (linked to sea level rise).

    Parameters
    ----------
    seed : int, optional
        Random seed for reproducibility
    verbose : bool, default True
        Whether to print progress

    Examples
    --------
    >>> generator = DamagesDataGenerator(seed=42)
    >>> generator.generate_all_damages_data("damages_data")
    """

    def __init__(self, seed: Optional[int] = None, verbose: bool = True):
        self.seed = seed
        self.verbose = verbose
        if seed is not None:
            np.random.seed(seed)

    def _log(self, message: str):
        """Print message if verbose."""
        if self.verbose:
            print(message)

    def generate_economic_data(
        self,
        years: Optional[np.ndarray] = None,
        ssps: Optional[List[str]] = None,
        regions: Optional[List[str]] = None,
        iams: Optional[List[str]] = None,
    ) -> xr.Dataset:
        """
        Generate synthetic economic data (population, GDP).

        Parameters
        ----------
        years : np.ndarray, optional
            Years to generate (default: 2020-2030)
        ssps : list of str, optional
            SSP scenarios (default: ['ssp1', 'ssp2', 'ssp3', 'ssp4', 'ssp5'])
        regions : list of str, optional
            Regions (default: ['region1', 'region2'])
        iams : list of str, optional
            IAM models (default: ['iam1', 'iam2'])

        Returns
        -------
        xr.Dataset
            Economic dataset with variables: pop, gdppc, gdp
            Coordinates: year, ssp, region, model

        Examples
        --------
        >>> econ = generator.generate_economic_data()
        """
        if years is None:
            years = np.arange(2020, 2031)
        if ssps is None:
            ssps = ['ssp1', 'ssp2', 'ssp3', 'ssp4', 'ssp5']
        if regions is None:
            regions = ['region1', 'region2']
        if iams is None:
            iams = ['iam1', 'iam2']

        self._log("Generating economic data...")

        n_years = len(years)
        n_ssps = len(ssps)
        n_regions = len(regions)
        n_iams = len(iams)

        shape = (n_years, n_ssps, n_regions, n_iams)
        
        # Generate population data (millions)
        pop = np.zeros(shape)
        
        # Generate GDP per capita (thousands USD)
        gdppc = np.zeros(shape)
        
        for i_ssp, ssp in enumerate(ssps):
            # Different SSPs have different growth trajectories
            ssp_num = int(ssp[-1]) if ssp[-1].isdigit() else 3
            pop_growth_rate = 0.01 * (6 - ssp_num) / 100  # SSP1 grows faster than SSP5
            gdp_growth_rate = 0.02 * ssp_num / 100  # SSP5 grows faster economically
            
            for i_region, region in enumerate(regions):
                # Regional variations
                region_pop_base = np.random.uniform(20, 100)
                region_gdppc_base = np.random.uniform(30, 80)
                
                for i_iam, iam in enumerate(iams):
                    # IAM-specific offsets
                    iam_pop_offset = np.random.normal(0, 5)
                    iam_gdp_offset = np.random.normal(0, 10)
                    
                    for i_year, year in enumerate(years):
                        years_elapsed = year - years[0]
                        
                        # Population growth
                        pop_value = (region_pop_base + iam_pop_offset) * \
                                   (1 + pop_growth_rate) ** years_elapsed
                        pop_value += np.random.normal(0, 2)
                        pop[i_year, i_ssp, i_region, i_iam] = max(pop_value, 10)
                        
                        # GDP per capita growth
                        gdppc_value = (region_gdppc_base + iam_gdp_offset) * \
                                     (1 + gdp_growth_rate) ** years_elapsed
                        gdppc_value += np.random.normal(0, 3)
                        gdppc[i_year, i_ssp, i_region, i_iam] = max(gdppc_value, 20)
        
        # GDP is population * GDP per capita
        gdp = pop * gdppc
        
        # Create xarray Dataset
        ds = xr.Dataset(
            {
                'pop': (['year', 'ssp', 'region', 'model'], pop),
                'gdppc': (['year', 'ssp', 'region', 'model'], gdppc),
                'gdp': (['year', 'ssp', 'region', 'model'], gdp)
            },
            coords={
                'year': years,
                'ssp': ssps,
                'region': regions,
                'model': iams
            }
        )
        
        ds.pop.attrs['units'] = 'millions'
        ds.pop.attrs['long_name'] = 'Population'
        ds.gdppc.attrs['units'] = 'thousand USD'
        ds.gdppc.attrs['long_name'] = 'GDP per capita'
        ds.gdp.attrs['units'] = 'million USD'
        ds.gdp.attrs['long_name'] = 'GDP'

        self._log(f"Generated economic data: {pop.shape}")
        return ds

    def generate_noncoastal_damages(
        self,
        years: Optional[np.ndarray] = None,
        ssps: Optional[List[str]] = None,
        regions: Optional[List[str]] = None,
        iams: Optional[List[str]] = None,
        gcms: Optional[List[str]] = None,
        rcps: Optional[List[str]] = None,
        batches: Optional[np.ndarray] = None,
        delta_var_name: str = "delta_dummy",
        histclim_var_name: str = "histclim_dummy",
    ) -> xr.Dataset:
        """
        Generate synthetic non-coastal sectoral damages.

        Non-coastal damages are linked to temperature via gcm/rcp.

        Parameters
        ----------
        years : np.ndarray, optional
            Years to generate (default: 2020-2030)
        ssps : list of str, optional
            SSP scenarios
        regions : list of str, optional
            Regions
        iams : list of str, optional
            IAM models
        gcms : list of str, optional
            GCM models (default: ['gcm1', 'gcm2'])
        rcps : list of str, optional
            RCP scenarios (default: ['rcp45', 'rcp85'])
        batches : np.ndarray, optional
            Batch indices (default: 0-14)
        delta_var_name : str, default 'delta_dummy'
            Name for delta variable (climate impact)
        histclim_var_name : str, default 'histclim_dummy'
            Name for historical climate variable

        Returns
        -------
        xr.Dataset
            Non-coastal damages with delta and histclim variables
            Coordinates: rcp, region, gcm, year, model, ssp, batch

        Examples
        --------
        >>> damages = generator.generate_noncoastal_damages()
        """
        if years is None:
            years = np.arange(2020, 2031)
        if ssps is None:
            ssps = ['ssp1', 'ssp2', 'ssp3', 'ssp4', 'ssp5']
        if regions is None:
            regions = ['region1', 'region2']
        if iams is None:
            iams = ['iam1', 'iam2']
        if gcms is None:
            gcms = ['gcm1', 'gcm2']
        if rcps is None:
            rcps = ['rcp45', 'rcp85']
        if batches is None:
            batches = np.arange(15)

        self._log(f"Generating non-coastal damages (var names: {delta_var_name}, {histclim_var_name})...")

        n_rcps = len(rcps)
        n_regions = len(regions)
        n_gcms = len(gcms)
        n_years = len(years)
        n_iams = len(iams)
        n_ssps = len(ssps)
        n_batches = len(batches)

        shape = (n_rcps, n_regions, n_gcms, n_years, n_iams, n_ssps, n_batches)
        
        # Delta: Climate impact damages (temperature-dependent)
        delta = np.zeros(shape)
        
        # Histclim: Historical climate baseline (relatively stable)
        histclim = np.zeros(shape)
        
        for i_rcp, rcp in enumerate(rcps):
            # Higher RCP = more damages
            rcp_damage_factor = 1.5 if rcp == 'rcp85' else 1.0
            
            for i_region in range(n_regions):
                region_baseline = np.random.uniform(1, 10)
                
                for i_gcm in range(n_gcms):
                    gcm_offset = np.random.normal(0, 0.5)
                    
                    for i_year, year in enumerate(years):
                        # Damages increase over time
                        years_elapsed = year - years[0]
                        time_factor = 1 + 0.03 * years_elapsed
                        
                        for i_iam in range(n_iams):
                            for i_ssp in range(n_ssps):
                                for i_batch in range(n_batches):
                                    # Delta: increases with temperature/time
                                    delta_value = rcp_damage_factor * time_factor * \
                                                 (5 + gcm_offset + np.random.normal(0, 2))
                                    delta[i_rcp, i_region, i_gcm, i_year, i_iam, i_ssp, i_batch] = \
                                        max(delta_value, 0)
                                    
                                    # Histclim: relatively stable baseline
                                    histclim_value = region_baseline + np.random.normal(0, 0.5)
                                    histclim[i_rcp, i_region, i_gcm, i_year, i_iam, i_ssp, i_batch] = \
                                        max(histclim_value, 1)
        
        # Create xarray Dataset
        ds = xr.Dataset(
            {
                delta_var_name: (['rcp', 'region', 'gcm', 'year', 'model', 'ssp', 'batch'], delta),
                histclim_var_name: (['rcp', 'region', 'gcm', 'year', 'model', 'ssp', 'batch'], histclim)
            },
            coords={
                'rcp': rcps,
                'region': regions,
                'gcm': gcms,
                'year': years,
                'model': iams,
                'ssp': ssps,
                'batch': batches
            }
        ).chunk({'batch': -1})
        
        ds[delta_var_name].attrs['long_name'] = 'Climate impact damages'
        ds[histclim_var_name].attrs['long_name'] = 'Historical climate baseline'

        self._log(f"Generated non-coastal damages: {delta.shape}")
        return ds

    def generate_coastal_damages(
        self,
        years: Optional[np.ndarray] = None,
        ssps: Optional[List[str]] = None,
        regions: Optional[List[str]] = None,
        iams: Optional[List[str]] = None,
        slrs: Optional[List[int]] = None,
        batches: Optional[np.ndarray] = None,
        delta_var_name: str = "delta_coastal",
        histclim_var_name: str = "histclim_coastal",
    ) -> xr.Dataset:
        """
        Generate synthetic coastal sectoral damages.

        Coastal damages are linked to sea level rise (slr).

        Parameters
        ----------
        years : np.ndarray, optional
            Years to generate (default: 2020-2030)
        ssps : list of str, optional
            SSP scenarios
        regions : list of str, optional
            Regions
        iams : list of str, optional
            IAM models
        slrs : list of int, optional
            Sea level rise scenarios (default: [0, 1])
        batches : np.ndarray, optional
            Batch indices (default: 0-14)
        delta_var_name : str, default 'delta_coastal'
            Name for delta variable (climate impact)
        histclim_var_name : str, default 'histclim_coastal'
            Name for historical climate variable

        Returns
        -------
        xr.Dataset
            Coastal damages with delta and histclim variables
            Coordinates: region, year, batch, slr, model, ssp

        Examples
        --------
        >>> damages = generator.generate_coastal_damages()
        """
        if years is None:
            years = np.arange(2020, 2031)
        if ssps is None:
            ssps = ['ssp1', 'ssp2', 'ssp3', 'ssp4', 'ssp5']
        if regions is None:
            regions = ['region1', 'region2']
        if iams is None:
            iams = ['iam1', 'iam2']
        if slrs is None:
            slrs = [0, 1]
        if batches is None:
            batches = np.arange(15)

        self._log(f"Generating coastal damages (var names: {delta_var_name}, {histclim_var_name})...")

        n_regions = len(regions)
        n_years = len(years)
        n_batches = len(batches)
        n_slrs = len(slrs)
        n_iams = len(iams)
        n_ssps = len(ssps)

        shape = (n_regions, n_years, n_batches, n_slrs, n_iams, n_ssps)
        
        # Delta: Sea level rise impact damages
        delta = np.zeros(shape)
        
        # Histclim: Historical climate baseline
        histclim = np.zeros(shape)
        
        for i_region in range(n_regions):
            region_baseline = np.random.uniform(1, 10)
            
            for i_year, year in enumerate(years):
                years_elapsed = year - years[0]
                time_factor = 1 + 0.02 * years_elapsed
                
                for i_batch in range(n_batches):
                    for i_slr, slr in enumerate(slrs):
                        # Higher SLR = more coastal damages
                        slr_damage_factor = 1.0 + 0.5 * slr
                        
                        for i_iam in range(n_iams):
                            for i_ssp in range(n_ssps):
                                # Delta: increases with sea level rise and time
                                delta_value = slr_damage_factor * time_factor * \
                                             (5 + np.random.normal(0, 2))
                                delta[i_region, i_year, i_batch, i_slr, i_iam, i_ssp] = \
                                    max(delta_value, 0)
                                
                                # Histclim: relatively stable baseline
                                histclim_value = region_baseline + np.random.normal(0, 0.5)
                                histclim[i_region, i_year, i_batch, i_slr, i_iam, i_ssp] = \
                                    max(histclim_value, 1)
        
        # Create xarray Dataset
        ds = xr.Dataset(
            {
                delta_var_name: (['region', 'year', 'batch', 'slr', 'model', 'ssp'], delta),
                histclim_var_name: (['region', 'year', 'batch', 'slr', 'model', 'ssp'], histclim)
            },
            coords={
                'region': regions,
                'year': years,
                'batch': batches,
                'slr': slrs,
                'model': iams,
                'ssp': ssps
            }
        ).chunk({
            'region': 1,
            'slr': 1,
            'year': 1,
            'model': 1,
            'ssp': 1,
            'batch': -1
        })
        
        ds[delta_var_name].attrs['long_name'] = 'Coastal climate impact damages'
        ds[histclim_var_name].attrs['long_name'] = 'Coastal historical climate baseline'

        self._log(f"Generated coastal damages: {delta.shape}")
        return ds

    def generate_all_damages_data(
        self,
        output_dir: str,
        years: Optional[np.ndarray] = None,
        ssps: Optional[List[str]] = None,
        regions: Optional[List[str]] = None,
        iams: Optional[List[str]] = None,
        gcms: Optional[List[str]] = None,
        rcps: Optional[List[str]] = None,
        slrs: Optional[List[int]] = None,
        batches: Optional[np.ndarray] = None,
    ) -> Dict[str, str]:
        """
        Generate and save all damages data files.

        Creates economic data, non-coastal sectoral damages, and coastal 
        sectoral damages.

        Parameters
        ----------
        output_dir : str
            Output directory for damages data
        years : np.ndarray, optional
            Years for all data (default: 2020-2030)
        ssps : list of str, optional
            SSP scenarios
        regions : list of str, optional
            Regions
        iams : list of str, optional
            IAM models
        gcms : list of str, optional
            GCM models (for non-coastal)
        rcps : list of str, optional
            RCP scenarios (for non-coastal)
        slrs : list of int, optional
            Sea level rise scenarios (for coastal)
        batches : np.ndarray, optional
            Batch indices

        Returns
        -------
        dict
            Paths to all generated files

        Examples
        --------
        >>> generator = DamagesDataGenerator(seed=42)
        >>> paths = generator.generate_all_damages_data("damages_data")
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if years is None:
            years = np.arange(2020, 2031)
        if ssps is None:
            ssps = ['ssp1', 'ssp2', 'ssp3', 'ssp4', 'ssp5']
        if regions is None:
            regions = ['region1', 'region2']
        if iams is None:
            iams = ['iam1', 'iam2']
        if gcms is None:
            gcms = ['gcm1', 'gcm2']
        if rcps is None:
            rcps = ['rcp45', 'rcp85']
        if slrs is None:
            slrs = [0, 1]
        if batches is None:
            batches = np.arange(15)

        paths = {}

        # Create subdirectories
        econ_dir = output_path / "econ"
        sectoral_dir = output_path / "sectoral"
        econ_dir.mkdir(exist_ok=True)
        sectoral_dir.mkdir(exist_ok=True)

        # Economic data
        self._log("\n=== Generating Economic Data ===")
        econ_data = self.generate_economic_data(
            years=years,
            ssps=ssps,
            regions=regions,
            iams=iams
        )
        econ_path = econ_dir / "integration-econ.zarr"
        econ_data.to_zarr(econ_path, mode='w')
        paths["economic"] = str(econ_path)
        self._log(f"Saved: {econ_path}")

        # Non-coastal sectoral damages
        self._log("\n=== Generating Non-Coastal Sectoral Damages ===")
        noncoastal_data = self.generate_noncoastal_damages(
            years=years,
            ssps=ssps,
            regions=regions,
            iams=iams,
            gcms=gcms,
            rcps=rcps,
            batches=batches,
            delta_var_name="delta_dummy",
            histclim_var_name="histclim_dummy"
        )
        noncoastal_path = sectoral_dir / "noncoastal_damages.zarr"
        noncoastal_data.to_zarr(noncoastal_path, mode='w')
        paths["noncoastal_damages"] = str(noncoastal_path)
        self._log(f"Saved: {noncoastal_path}")

        # Coastal sectoral damages
        self._log("\n=== Generating Coastal Sectoral Damages ===")
        coastal_data = self.generate_coastal_damages(
            years=years,
            ssps=ssps,
            regions=regions,
            iams=iams,
            slrs=slrs,
            batches=batches,
            delta_var_name="delta_coastal",
            histclim_var_name="histclim_coastal"
        )
        coastal_path = sectoral_dir / "coastal_damages.zarr"
        coastal_data.to_zarr(coastal_path, mode='w')
        paths["coastal_damages"] = str(coastal_path)
        self._log(f"Saved: {coastal_path}")

        self._log(f"\n✓ All damages data generated in: {output_dir}")
        return paths

