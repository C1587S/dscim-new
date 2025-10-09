"""
Data provider abstractions for lazy-loading and caching.

Keeps data management separate from computation and orchestration.
This module provides clean interfaces for accessing climate and economic
data without mixing I/O concerns into the computational pipeline.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import xarray as xr
import logging

logger = logging.getLogger(__name__)


class DataProvider(ABC):
    """
    Abstract base for data providers.

    Handles lazy loading, caching, and coordinate alignment.
    Subclasses implement specific data loading logic.

    Parameters
    ----------
    cache : bool, default True
        Whether to cache loaded data in memory
    verbose : bool, default False
        Whether to print loading messages

    Examples
    --------
    >>> class MyDataProvider(DataProvider):
    ...     def load(self, **kwargs):
    ...         return load_my_data(**kwargs)
    ...
    >>> provider = MyDataProvider(cache=True)
    >>> data = provider.get("key", provider.load, param=value)
    """

    def __init__(self, cache: bool = True, verbose: bool = False):
        self._cache = {} if cache else None
        self.verbose = verbose

    def get(self, key: str, loader_func, **kwargs):
        """
        Get data with optional caching.

        Parameters
        ----------
        key : str
            Cache key for this data
        loader_func : callable
            Function to call if data not in cache
        **kwargs
            Arguments to pass to loader_func

        Returns
        -------
        Any
            Loaded data (from cache or freshly loaded)
        """
        if self._cache is not None and key in self._cache:
            if self.verbose:
                logger.info(f"Loading {key} from cache")
            return self._cache[key]

        if self.verbose:
            logger.info(f"Loading {key} from disk")

        data = loader_func(**kwargs)

        if self._cache is not None:
            self._cache[key] = data

        return data

    def clear_cache(self):
        """Clear all cached data."""
        if self._cache is not None:
            self._cache.clear()
            if self.verbose:
                logger.info("Cache cleared")

    def get_cache_size(self) -> int:
        """Get number of items in cache."""
        return len(self._cache) if self._cache is not None else 0

    @abstractmethod
    def load(self, **kwargs):
        """Load data (implemented by subclasses)."""
        pass


class ClimateDataProvider(DataProvider):
    """
    Provides climate data (GMST, GMSL, FAIR projections).

    Handles:
    - Lazy loading of climate datasets
    - Pulse year subsetting
    - ECS (Equilibrium Climate Sensitivity) masking
    - Control vs pulse differentiation

    Parameters
    ----------
    climate_config : ClimateDataConfig or dict
        Configuration with paths to climate data files
    cache : bool, default True
        Whether to cache loaded data
    verbose : bool, default False
        Whether to print loading messages

    Examples
    --------
    >>> from dscim_new.config import ClimateDataConfig
    >>> config = ClimateDataConfig(
    ...     gmst_path="data/gmst.csv",
    ...     fair_temperature_path="data/fair_temp.nc"
    ... )
    >>> climate = ClimateDataProvider(config)
    >>> gmst = climate.get_gmst(pulse_year=2020)
    >>> fair_temp = climate.get_fair_temperature(pulse_year=2020, mode="pulse")
    """

    def __init__(
        self,
        climate_config,
        cache: bool = True,
        verbose: bool = False
    ):
        super().__init__(cache, verbose)
        self.config = climate_config

    def get_gmst(self, pulse_year: Optional[int] = None) -> xr.Dataset:
        """
        Load GMST (Global Mean Surface Temperature) matching data.

        This is typically a CSV file with columns:
        - year: int
        - rcp: str (emission scenario)
        - gcm: str (climate model)
        - anomaly: float (temperature anomaly)

        Parameters
        ----------
        pulse_year : int, optional
            If provided, subset data to relevant years

        Returns
        -------
        xr.Dataset
            GMST data with temperature anomalies
        """
        key = f"gmst_{pulse_year if pulse_year else 'all'}"
        return self.get(key, self._load_gmst, pulse_year=pulse_year)

    def get_gmsl(self, pulse_year: Optional[int] = None) -> xr.Dataset:
        """
        Load GMSL (Global Mean Sea Level) data.

        Zarr format with coordinates:
        - year: int
        - slr: int or str (sea level rise scenario)

        And data variable:
        - gmsl: float (sea level in meters)

        Parameters
        ----------
        pulse_year : int, optional
            If provided, subset data to relevant years

        Returns
        -------
        xr.Dataset
            GMSL data
        """
        key = f"gmsl_{pulse_year if pulse_year else 'all'}"
        return self.get(key, self._load_gmsl, pulse_year=pulse_year)

    def get_fair_temperature(
        self,
        pulse_year: int,
        mode: str = "control",
        ecs_mask: Optional[str] = None
    ) -> xr.Dataset:
        """
        Load FAIR (Finite Amplitude Impulse Response) temperature projections.

        NetCDF file with coordinates:
        - year: int
        - rcp: str
        - simulation: int
        - gas: str

        And data variables:
        - control_temperature: float (no pulse)
        - pulse_temperature: float (with carbon pulse)
        - medianparams_control_temperature: float
        - medianparams_pulse_temperature: float

        Parameters
        ----------
        pulse_year : int
            Year of carbon pulse
        mode : str, default "control"
            Which temperature to load:
            - "control": Control scenario (no pulse)
            - "pulse": Pulse scenario
            - "both": Both control and pulse
        ecs_mask : str, optional
            ECS mask name to apply (e.g., "truncate_at_ecs950symmetric")

        Returns
        -------
        xr.Dataset
            FAIR temperature projections
        """
        key = f"fair_temp_{pulse_year}_{mode}_{ecs_mask}"
        return self.get(
            key,
            self._load_fair_temp,
            pulse_year=pulse_year,
            mode=mode,
            ecs_mask=ecs_mask
        )

    def get_fair_gmsl(
        self,
        pulse_year: int,
        mode: str = "control",
        ecs_mask: Optional[str] = None
    ) -> xr.Dataset:
        """
        Load FAIR GMSL projections.

        Similar structure to FAIR temperature but for sea level.

        Parameters
        ----------
        pulse_year : int
            Year of carbon pulse
        mode : str, default "control"
            "control", "pulse", or "both"
        ecs_mask : str, optional
            ECS mask to apply

        Returns
        -------
        xr.Dataset
            FAIR GMSL projections
        """
        key = f"fair_gmsl_{pulse_year}_{mode}_{ecs_mask}"
        return self.get(
            key,
            self._load_fair_gmsl,
            pulse_year=pulse_year,
            mode=mode,
            ecs_mask=ecs_mask
        )

    def get_pulse_conversion(self) -> xr.Dataset:
        """
        Load pulse conversion factors.

        Converts large pulses (e.g., gigatons) to per-ton basis.

        Returns
        -------
        xr.Dataset
            Conversion factors by gas type
        """
        key = "pulse_conversion"
        return self.get(key, self._load_pulse_conversion)

    def get_ecs_mask(self, mask_name: str) -> xr.Dataset:
        """
        Load ECS (Equilibrium Climate Sensitivity) mask.

        Masks filter climate simulations based on ECS values.

        Parameters
        ----------
        mask_name : str
            Name of mask (e.g., "truncate_at_ecs950symmetric")

        Returns
        -------
        xr.Dataset
            Boolean mask for filtering simulations
        """
        key = f"ecs_mask_{mask_name}"
        return self.get(key, self._load_ecs_mask, mask_name=mask_name)

    # Internal loading methods
    # -------------------------

    def _load_gmst(self, pulse_year: Optional[int] = None) -> xr.Dataset:
        """Internal loader for GMST data."""
        from ..io import load_gmst_data

        data = load_gmst_data(self.config.gmst_path)

        # Apply pulse year filtering if needed
        if pulse_year is not None and 'year' in data.dims:
            # Keep data around pulse year (typically +/- some years)
            # For now, keep all years
            pass

        return data

    def _load_gmsl(self, pulse_year: Optional[int] = None) -> xr.Dataset:
        """Internal loader for GMSL data."""
        from ..io import load_gmsl_data

        data = load_gmsl_data(self.config.gmsl_path)

        if pulse_year is not None and 'year' in data.dims:
            pass

        return data

    def _load_fair_temp(
        self,
        pulse_year: int,
        mode: str = "control",
        ecs_mask: Optional[str] = None
    ) -> xr.Dataset:
        """Internal loader for FAIR temperature data."""
        from ..io import load_fair_temperature

        data = load_fair_temperature(self.config.fair_temperature_path)

        # Select control or pulse
        if mode == "control":
            data = data[['control_temperature', 'medianparams_control_temperature']]
        elif mode == "pulse":
            # Select pulse data for specific pulse year
            if 'pulse_year' in data.dims:
                data = data.sel(pulse_year=pulse_year)
            data = data[['pulse_temperature', 'medianparams_pulse_temperature']]
        elif mode == "both":
            # Keep both, but select pulse_year
            if 'pulse_year' in data.dims:
                pulse_data = data[['pulse_temperature', 'medianparams_pulse_temperature']].sel(pulse_year=pulse_year)
                control_data = data[['control_temperature', 'medianparams_control_temperature']]
                data = xr.merge([control_data, pulse_data])
        else:
            raise ValueError(f"Unknown mode: {mode}. Choose 'control', 'pulse', or 'both'")

        # Apply ECS mask if specified
        if ecs_mask:
            mask_data = self.get_ecs_mask(ecs_mask)
            # Apply mask to simulation dimension
            if 'simulation' in data.dims and 'simulation' in mask_data.dims:
                data = data.where(mask_data[ecs_mask], drop=True)

        return data

    def _load_fair_gmsl(
        self,
        pulse_year: int,
        mode: str = "control",
        ecs_mask: Optional[str] = None
    ) -> xr.Dataset:
        """Internal loader for FAIR GMSL data."""
        from ..io import load_fair_gmsl

        data = load_fair_gmsl(self.config.fair_gmsl_path)

        # Similar logic to temperature
        if mode == "control":
            data = data[['control_gmsl', 'medianparams_control_gmsl']]
        elif mode == "pulse":
            if 'pulse_years' in data.dims:  # Note: might be 'pulse_years' not 'pulse_year'
                data = data.sel(pulse_years=pulse_year)
            data = data[['pulse_gmsl', 'medianparams_pulse_gmsl']]
        elif mode == "both":
            if 'pulse_years' in data.dims:
                pulse_data = data[['pulse_gmsl', 'medianparams_pulse_gmsl']].sel(pulse_years=pulse_year)
                control_data = data[['control_gmsl', 'medianparams_control_gmsl']]
                data = xr.merge([control_data, pulse_data])

        if ecs_mask:
            mask_data = self.get_ecs_mask(ecs_mask)
            if 'simulation' in data.dims and 'simulation' in mask_data.dims:
                data = data.where(mask_data[ecs_mask], drop=True)

        return data

    def _load_pulse_conversion(self) -> xr.Dataset:
        """Internal loader for pulse conversion."""
        from ..io import load_pulse_conversion

        if self.config.pulse_conversion_path:
            return load_pulse_conversion(self.config.pulse_conversion_path)
        else:
            # Return default conversion (1.0 for all gases)
            import numpy as np
            return xr.Dataset(
                {'emissions': (['gas'], np.ones(1))},
                coords={'gas': ['CO2']}
            )

    def _load_ecs_mask(self, mask_name: str) -> xr.Dataset:
        """Internal loader for ECS mask."""
        import xarray as xr

        if self.config.ecs_mask_path:
            # Load netCDF file directly (assuming it's NetCDF format)
            data = xr.open_dataset(self.config.ecs_mask_path)
            # Extract specific mask
            if mask_name in data.data_vars:
                return data[[mask_name]]
            else:
                raise ValueError(f"Mask '{mask_name}' not found in {self.config.ecs_mask_path}")
        else:
            raise ValueError("No ECS mask path configured")

    def load(self, **kwargs):
        """General load method (for compatibility with base class)."""
        raise NotImplementedError(
            "Use specific methods like get_gmst(), get_fair_temperature(), etc."
        )


class EconomicDataProvider(DataProvider):
    """
    Provides economic data (GDP, population, consumption).

    Handles:
    - Lazy loading of socioeconomic datasets
    - SSP (Shared Socioeconomic Pathway) selection
    - Region aggregation
    - Global vs regional data

    Parameters
    ----------
    econ_path : str or Path
        Path to socioeconomic data (typically zarr file)
    cache : bool, default True
        Whether to cache loaded data
    verbose : bool, default False
        Whether to print loading messages

    Examples
    --------
    >>> econ = EconomicDataProvider("data/econ.zarr")
    >>> consumption = econ.get_consumption(ssps=["ssp2", "ssp3"])
    >>> population = econ.get_population(regions=["USA", "CHN"])
    >>> gdp = econ.get_gdp()
    """

    def __init__(
        self,
        econ_path: Union[str, Path],
        cache: bool = True,
        verbose: bool = False
    ):
        super().__init__(cache, verbose)
        self.path = Path(econ_path)

    def get_consumption(
        self,
        ssps: Optional[List[str]] = None,
        regions: Optional[List[str]] = None,
        years: Optional[List[int]] = None
    ) -> xr.DataArray:
        """
        Get consumption (GDP per capita) data.

        Parameters
        ----------
        ssps : list of str, optional
            SSP scenarios to include (e.g., ["ssp2", "ssp3"])
        regions : list of str, optional
            Regions to include
        years : list of int, optional
            Years to include

        Returns
        -------
        xr.DataArray
            Consumption data (gdppc)
        """
        key = f"consumption_{'_'.join(ssps) if ssps else 'all'}"
        return self.get(
            key,
            self._load_consumption,
            ssps=ssps,
            regions=regions,
            years=years
        )

    def get_population(
        self,
        ssps: Optional[List[str]] = None,
        regions: Optional[List[str]] = None,
        years: Optional[List[int]] = None
    ) -> xr.DataArray:
        """
        Get population data.

        Parameters
        ----------
        ssps : list of str, optional
            SSP scenarios to include
        regions : list of str, optional
            Regions to include
        years : list of int, optional
            Years to include

        Returns
        -------
        xr.DataArray
            Population data
        """
        key = f"population_{'_'.join(ssps) if ssps else 'all'}"
        return self.get(
            key,
            self._load_population,
            ssps=ssps,
            regions=regions,
            years=years
        )

    def get_gdp(
        self,
        ssps: Optional[List[str]] = None,
        regions: Optional[List[str]] = None,
        years: Optional[List[int]] = None
    ) -> xr.DataArray:
        """
        Get GDP data.

        Parameters
        ----------
        ssps : list of str, optional
            SSP scenarios to include
        regions : list of str, optional
            Regions to include
        years : list of int, optional
            Years to include

        Returns
        -------
        xr.DataArray
            GDP data
        """
        key = f"gdp_{'_'.join(ssps) if ssps else 'all'}"
        return self.get(
            key,
            self._load_gdp,
            ssps=ssps,
            regions=regions,
            years=years
        )

    def get_full_dataset(self) -> xr.Dataset:
        """
        Get complete economic dataset with all variables.

        Returns
        -------
        xr.Dataset
            Full dataset with gdp, gdppc, pop
        """
        key = "full_dataset"
        return self.get(key, self._load_full_dataset)

    # Internal loading methods
    # -------------------------

    def _load_consumption(
        self,
        ssps: Optional[List[str]] = None,
        regions: Optional[List[str]] = None,
        years: Optional[List[int]] = None
    ) -> xr.DataArray:
        """Internal loader for consumption."""
        from ..io import load_socioeconomic_data

        data = load_socioeconomic_data(self.path)
        consumption = data['gdppc']

        # Apply filters
        if ssps:
            consumption = consumption.sel(ssp=ssps)
        if regions:
            consumption = consumption.sel(region=regions)
        if years:
            consumption = consumption.sel(year=years)

        return consumption

    def _load_population(
        self,
        ssps: Optional[List[str]] = None,
        regions: Optional[List[str]] = None,
        years: Optional[List[int]] = None
    ) -> xr.DataArray:
        """Internal loader for population."""
        from ..io import load_socioeconomic_data

        data = load_socioeconomic_data(self.path)
        population = data['pop']

        if ssps:
            population = population.sel(ssp=ssps)
        if regions:
            population = population.sel(region=regions)
        if years:
            population = population.sel(year=years)

        return population

    def _load_gdp(
        self,
        ssps: Optional[List[str]] = None,
        regions: Optional[List[str]] = None,
        years: Optional[List[int]] = None
    ) -> xr.DataArray:
        """Internal loader for GDP."""
        from ..io import load_socioeconomic_data

        data = load_socioeconomic_data(self.path)
        gdp = data['gdp']

        if ssps:
            gdp = gdp.sel(ssp=ssps)
        if regions:
            gdp = gdp.sel(region=regions)
        if years:
            gdp = gdp.sel(year=years)

        return gdp

    def _load_full_dataset(self) -> xr.Dataset:
        """Internal loader for full dataset."""
        from ..io import load_socioeconomic_data
        return load_socioeconomic_data(self.path)

    def load(self, **kwargs):
        """General load method (for compatibility with base class)."""
        return self._load_full_dataset()
