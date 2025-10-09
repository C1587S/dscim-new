"""
I/O functions for climate data.

Loading GMST, GMSL, and FAIR climate model outputs.
"""

import xarray as xr
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any


def load_gmst_data(
    gmst_path: str,
    use_dask: bool = True
) -> pd.DataFrame:
    """
    Load Global Mean Surface Temperature (GMST) data.

    Parameters
    ----------
    gmst_path : str
        Path to GMST CSV file
    use_dask : bool, default True
        Not used for CSV, included for API consistency

    Returns
    -------
    pd.DataFrame
        GMST data with columns: [year, rcp, gcm, anomaly]

    Examples
    --------
    >>> gmst = load_gmst_data("climate/GMTanom_all_temp.csv")

    Notes
    -----
    GMST file format (CSV):
    - year: int
    - rcp: str
    - gcm: str
    - anomaly: float (temperature anomaly in degrees C)
    """
    path = Path(gmst_path)
    if not path.exists():
        raise FileNotFoundError(f"GMST file not found: {gmst_path}")

    # Load CSV
    df = pd.read_csv(gmst_path)

    # Validate expected columns
    expected_cols = ['year', 'rcp', 'gcm', 'anomaly']
    missing_cols = set(expected_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"GMST file missing columns: {missing_cols}")

    return df


def load_gmsl_data(
    gmsl_path: str,
    use_dask: bool = True,
    chunks: Optional[Dict[str, int]] = None
) -> xr.Dataset:
    """
    Load Global Mean Sea Level (GMSL) data.

    Parameters
    ----------
    gmsl_path : str
        Path to GMSL Zarr file
    use_dask : bool, default True
        Whether to use Dask for lazy loading
    chunks : dict, optional
        Chunk specification for Dask

    Returns
    -------
    xr.Dataset
        GMSL data with dims (year, slr) and variable 'gmsl'

    Examples
    --------
    >>> gmsl = load_gmsl_data("climate/coastal_gmsl.zarr")

    Notes
    -----
    GMSL file format (Zarr):
    - Dimensions: year, slr
    - Variable: gmsl (sea level in meters)
    """
    path = Path(gmsl_path)
    if not path.exists():
        raise FileNotFoundError(f"GMSL file not found: {gmsl_path}")

    # Load Zarr
    if use_dask:
        if chunks is None:
            ds = xr.open_zarr(gmsl_path)
        else:
            ds = xr.open_zarr(gmsl_path).chunk(chunks)
    else:
        ds = xr.open_zarr(gmsl_path, chunks=None)

    # Validate
    if 'gmsl' not in ds.data_vars:
        raise ValueError(f"GMSL file must contain 'gmsl' variable. Found: {list(ds.data_vars)}")

    return ds


def load_fair_temperature(
    fair_path: str,
    pulse_year: int = 2020,
    use_dask: bool = True,
    chunks: Optional[Dict[str, int]] = None
) -> xr.Dataset:
    """
    Load FAIR temperature control and pulse data.

    Parameters
    ----------
    fair_path : str
        Path to FAIR temperature NetCDF file
    pulse_year : int, default 2020
        Pulse year to load
    use_dask : bool, default True
        Whether to use Dask
    chunks : dict, optional
        Chunk specification

    Returns
    -------
    xr.Dataset
        FAIR temperature data with variables:
        - control_temperature
        - pulse_temperature
        - medianparams_control_temperature
        - medianparams_pulse_temperature

    Examples
    --------
    >>> fair_temp = load_fair_temperature(
    ...     "climate/ar6_fair162_sim.nc",
    ...     pulse_year=2020
    ... )

    Notes
    -----
    FAIR file format (NetCDF):
    - Dimensions: year, rcp, simulation, gas, pulse_year
    - Variables: control_temperature, pulse_temperature, etc.
    """
    path = Path(fair_path)
    if not path.exists():
        raise FileNotFoundError(f"FAIR temperature file not found: {fair_path}")

    # Load NetCDF
    if use_dask:
        if chunks is None:
            ds = xr.open_dataset(fair_path)
        else:
            ds = xr.open_dataset(fair_path, chunks=chunks)
    else:
        ds = xr.open_dataset(fair_path)

    # Validate variables
    expected_vars = ['control_temperature', 'pulse_temperature']
    missing_vars = set(expected_vars) - set(ds.data_vars)
    if missing_vars:
        raise ValueError(f"FAIR file missing variables: {missing_vars}")

    # Select pulse year if needed
    if 'pulse_year' in ds.dims and pulse_year is not None:
        if pulse_year in ds.pulse_year.values:
            ds = ds.sel(pulse_year=pulse_year)
        else:
            raise ValueError(f"Pulse year {pulse_year} not found in FAIR data. "
                           f"Available: {ds.pulse_year.values}")

    return ds


def load_fair_gmsl(
    fair_gmsl_path: str,
    pulse_year: int = 2020,
    use_dask: bool = True,
    chunks: Optional[Dict[str, int]] = None
) -> xr.Dataset:
    """
    Load FAIR GMSL control and pulse data.

    Parameters
    ----------
    fair_gmsl_path : str
        Path to FAIR GMSL NetCDF file
    pulse_year : int, default 2020
        Pulse year to load
    use_dask : bool, default True
        Whether to use Dask
    chunks : dict, optional
        Chunk specification

    Returns
    -------
    xr.Dataset
        FAIR GMSL data with variables:
        - control_gmsl
        - pulse_gmsl
        - medianparams_control_gmsl
        - medianparams_pulse_gmsl

    Examples
    --------
    >>> fair_gmsl = load_fair_gmsl(
    ...     "climate/scenario_gmsl.nc4",
    ...     pulse_year=2020
    ... )

    Notes
    -----
    Similar structure to FAIR temperature but for sea level.
    """
    path = Path(fair_gmsl_path)
    if not path.exists():
        raise FileNotFoundError(f"FAIR GMSL file not found: {fair_gmsl_path}")

    # Load NetCDF
    if use_dask:
        if chunks is None:
            ds = xr.open_dataset(fair_gmsl_path)
        else:
            ds = xr.open_dataset(fair_gmsl_path, chunks=chunks)
    else:
        ds = xr.open_dataset(fair_gmsl_path)

    # Validate variables
    expected_vars = ['control_gmsl', 'pulse_gmsl']
    missing_vars = set(expected_vars) - set(ds.data_vars)
    if missing_vars:
        raise ValueError(f"FAIR GMSL file missing variables: {missing_vars}")

    # Select pulse year if needed
    # Note: dimension name might be 'pulse_years' (plural) in some files
    pulse_dim = 'pulse_years' if 'pulse_years' in ds.dims else 'pulse_year'

    if pulse_dim in ds.dims and pulse_year is not None:
        if pulse_year in ds[pulse_dim].values:
            ds = ds.sel({pulse_dim: pulse_year})
        else:
            raise ValueError(f"Pulse year {pulse_year} not found in FAIR GMSL data. "
                           f"Available: {ds[pulse_dim].values}")

    return ds


def load_pulse_conversion(
    conversion_path: str
) -> xr.Dataset:
    """
    Load pulse conversion factors.

    Converts large pulses (gigatons) to one-ton pulse equivalents.

    Parameters
    ----------
    conversion_path : str
        Path to conversion NetCDF file

    Returns
    -------
    xr.Dataset
        Conversion factors with variable 'emissions'

    Examples
    --------
    >>> conversion = load_pulse_conversion("climate/conversion.nc4")

    Notes
    -----
    File format (NetCDF):
    - Dimension: gas
    - Variable: emissions (conversion factor)
    """
    path = Path(conversion_path)
    if not path.exists():
        raise FileNotFoundError(f"Pulse conversion file not found: {conversion_path}")

    ds = xr.open_dataset(conversion_path)

    if 'emissions' not in ds.data_vars:
        raise ValueError(f"Conversion file must contain 'emissions' variable. "
                        f"Found: {list(ds.data_vars)}")

    return ds


def match_climate_to_damages(
    climate_data: xr.DataArray,
    damage_coords: Dict[str, Any],
    climate_var: str = "anomaly"
) -> xr.DataArray:
    """
    Match climate data to damage coordinates.

    Aligns climate data with damage function coordinates by
    selecting matching years, scenarios, models, etc.

    Parameters
    ----------
    climate_data : xr.DataArray
        Climate variable data
    damage_coords : dict
        Coordinates from damage data (year, rcp, gcm, etc.)
    climate_var : str, default "anomaly"
        Name of climate variable

    Returns
    -------
    xr.DataArray
        Climate data aligned with damage coordinates

    Examples
    --------
    >>> climate_aligned = match_climate_to_damages(
    ...     temperature,
    ...     {"year": years, "rcp": rcps, "gcm": gcms}
    ... )

    Notes
    -----
    This function handles coordinate matching and broadcasting to
    ensure climate and damage data are properly aligned for regression.
    """
    # Select matching coordinates
    matched = climate_data

    for coord_name, coord_values in damage_coords.items():
        if coord_name in matched.dims or coord_name in matched.coords:
            # Select overlapping values
            if hasattr(coord_values, 'values'):
                coord_values = coord_values.values

            # Find intersection
            existing_values = matched[coord_name].values if coord_name in matched.coords else None

            if existing_values is not None:
                common_values = list(set(coord_values) & set(existing_values))
                if common_values:
                    matched = matched.sel({coord_name: common_values})

    return matched


def get_climate_variable_for_sector(
    sector: str,
    gmst_data: Optional[pd.DataFrame] = None,
    gmsl_data: Optional[xr.Dataset] = None,
    use_gmsl: bool = False
) -> xr.DataArray:
    """
    Get appropriate climate variable for a sector.

    Parameters
    ----------
    sector : str
        Sector name
    gmst_data : pd.DataFrame, optional
        GMST temperature data
    gmsl_data : xr.Dataset, optional
        GMSL sea level data
    use_gmsl : bool, default False
        If True, use GMSL regardless of sector name

    Returns
    -------
    xr.DataArray
        Climate variable for the sector

    Examples
    --------
    >>> climate_var = get_climate_variable_for_sector(
    ...     sector="coastal",
    ...     gmsl_data=gmsl
    ... )

    Notes
    -----
    Coastal sectors typically use GMSL, while other sectors use GMST.
    """
    # Determine if coastal sector
    is_coastal = "coastal" in sector.lower() or use_gmsl

    if is_coastal:
        if gmsl_data is None:
            raise ValueError(f"GMSL data required for coastal sector: {sector}")
        return gmsl_data['gmsl']
    else:
        if gmst_data is None:
            raise ValueError(f"GMST data required for non-coastal sector: {sector}")

        # Convert DataFrame to xarray
        gmst_xr = gmst_data.set_index(['year', 'rcp', 'gcm']).to_xarray()
        return gmst_xr['anomaly']
