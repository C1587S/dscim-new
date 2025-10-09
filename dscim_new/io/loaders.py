"""
Data loading functions for DSCIM.

All file reading operations are isolated here.
"""

import xarray as xr
import yaml
from typing import Dict, Any, Optional, List
from pathlib import Path


def load_sector_config(config_path: str, sector: str) -> Dict[str, Any]:
    """
    Load sector configuration from YAML file.

    Parameters
    ----------
    config_path : str
        Path to YAML configuration file
    sector : str
        Sector name to load

    Returns
    -------
    dict
        Sector configuration dictionary

    Examples
    --------
    >>> config = load_sector_config("configs/dummy_config.yaml", "dummy_coastal_sector")
    >>> config["sector_path"]
    './dummy_data/sectoral/coastal_damages.zarr'
    """
    with open(config_path, 'r') as f:
        full_config = yaml.safe_load(f)

    return full_config["sectors"][sector]


def load_damages_data(
    damages_path: str,
    variable: str = None,
    chunks: Optional[Dict] = None,
    use_dask: bool = True
) -> xr.Dataset:
    """
    Load damages data from zarr file.

    Parameters
    ----------
    damages_path : str
        Path to damages zarr file
    variable : str, optional
        Specific variable to load. If None, loads full dataset
    chunks : dict, optional
        Chunking specification for dask arrays
        Example: {"rcp": 1, "region": -1, "year": 10}
    use_dask : bool, optional
        Whether to use dask for lazy loading (default: True)
        Set to False for immediate loading into memory

    Returns
    -------
    xr.Dataset
        Damages dataset

    Examples
    --------
    >>> # Load with dask (lazy)
    >>> damages = load_damages_data(
    ...     "./dummy_data/sectoral/coastal_damages.zarr",
    ...     chunks={"region": -1, "slr": 1, "year": 10}
    ... )

    >>> # Load without dask (immediate)
    >>> damages = load_damages_data(
    ...     "./dummy_data/sectoral/coastal_damages.zarr",
    ...     use_dask=False
    ... )

    Notes
    -----
    When use_dask=True, data is loaded lazily and processed in chunks.
    When use_dask=False, all data is loaded into memory immediately.
    """
    if use_dask:
        # Lazy loading with dask
        if chunks is None:
            ds = xr.open_zarr(damages_path)
        else:
            ds = xr.open_zarr(damages_path).chunk(chunks)
    else:
        # Eager loading without dask
        ds = xr.open_zarr(damages_path, chunks=None)
        ds = ds.compute()  # Load all data into memory

    if variable is not None:
        return ds[variable]

    return ds


def load_socioeconomic_data(
    socioec_path: str,
    chunks: Optional[Dict] = None,
    use_dask: bool = True
) -> xr.Dataset:
    """
    Load GDP per capita and population data from zarr file.

    Parameters
    ----------
    socioec_path : str
        Path to socioeconomic data zarr file
    chunks : dict, optional
        Chunking specification for dask arrays
    use_dask : bool, optional
        Whether to use dask for lazy loading (default: True)

    Returns
    -------
    xr.Dataset
        Socioeconomic dataset containing gdppc, pop, gdp

    Examples
    --------
    >>> # Load with dask
    >>> econ = load_socioeconomic_data("./dummy_data/econ/integration-econ-bc39.zarr")

    >>> # Load without dask
    >>> econ = load_socioeconomic_data(
    ...     "./dummy_data/econ/integration-econ-bc39.zarr",
    ...     use_dask=False
    ... )
    """
    if use_dask:
        if chunks is None:
            ds = xr.open_zarr(socioec_path)
        else:
            ds = xr.open_zarr(socioec_path).chunk(chunks)
    else:
        ds = xr.open_zarr(socioec_path, chunks=None)
        ds = ds.compute()

    return ds


def get_gdppc_for_coordinates(
    socioec_path: str,
    year: Any,
    ssp: Any,
    model: Any,
    regions: List[str],
    use_dask: bool = True
) -> xr.DataArray:
    """
    Load GDP per capita for specific coordinates.

    This function is designed to be called from within chunk processing,
    loading only the necessary subset of socioeconomic data.

    Parameters
    ----------
    socioec_path : str
        Path to socioeconomic data zarr file
    year : array-like
        Year coordinate(s)
    ssp : array-like
        SSP scenario coordinate(s)
    model : array-like
        Model coordinate(s)
    regions : list of str
        Region names to include
    use_dask : bool, optional
        Whether to use dask for lazy loading (default: True)

    Returns
    -------
    xr.DataArray
        GDP per capita for specified coordinates

    Examples
    --------
    >>> gdppc = get_gdppc_for_coordinates(
    ...     "./dummy_data/econ/integration-econ-bc39.zarr",
    ...     year=[2020, 2021],
    ...     ssp=["ssp2"],
    ...     model=["dummy1"],
    ...     regions=["dummy1", "dummy2"]
    ... )

    Notes
    -----
    This corresponds to the chunk-level loading in the original DSCIM's
    ce_from_chunk function.
    """
    if use_dask:
        gdppc = xr.open_zarr(socioec_path, chunks=None)["gdppc"]
    else:
        gdppc = xr.open_zarr(socioec_path, chunks=None)["gdppc"].compute()

    # Select specific coordinates
    gdppc = gdppc.sel(
        year=year,
        ssp=ssp,
        model=model
    )

    # Filter regions (handle cases where not all regions are present)
    available_regions = [r for r in regions if r in gdppc.region.values]
    if available_regions:
        gdppc = gdppc.sel(region=available_regions)

    return gdppc