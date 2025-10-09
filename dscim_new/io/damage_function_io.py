"""
I/O functions for damage function data.

Saving and loading fitted damage functions, coefficients, and marginal damages.
"""

import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Union


def save_damage_function_coefficients(
    coefficients: xr.DataArray,
    output_path: str,
    format: str = "zarr",
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save damage function coefficients to disk.

    Parameters
    ----------
    coefficients : xr.DataArray
        Fitted damage function coefficients with 'coefficient' dimension
    output_path : str
        Path to save coefficients
    format : str, default "zarr"
        Output format: "zarr", "netcdf", or "csv"
    metadata : dict, optional
        Additional metadata to save

    Examples
    --------
    >>> save_damage_function_coefficients(
    ...     coefficients,
    ...     "outputs/damage_functions/mortality_2020_coefficients.zarr",
    ...     format="zarr"
    ... )

    Notes
    -----
    CSV format saves a simple table with coefficient names and values.
    Zarr/NetCDF preserve all dimensions and coordinates.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Add metadata
    if metadata is not None:
        for key, value in metadata.items():
            coefficients.attrs[key] = value

    if format == "zarr":
        # Convert to Dataset for Zarr
        ds = coefficients.to_dataset(name='coefficients')
        ds.to_zarr(output_path, mode='w')

    elif format == "netcdf":
        ds = coefficients.to_dataset(name='coefficients')
        ds.to_netcdf(output_path)

    elif format == "csv":
        # Convert to DataFrame for CSV
        df = coefficients.to_pandas()
        if isinstance(df, pd.Series):
            df = df.to_frame(name='value')
        df.to_csv(output_path)

    else:
        raise ValueError(f"Unknown format: {format}. Choose from: zarr, netcdf, csv")


def load_damage_function_coefficients(
    input_path: str,
    format: str = "zarr",
    use_dask: bool = True
) -> xr.DataArray:
    """
    Load damage function coefficients from disk.

    Parameters
    ----------
    input_path : str
        Path to coefficients file
    format : str, default "zarr"
        Input format: "zarr", "netcdf", or "csv"
    use_dask : bool, default True
        Whether to use Dask for lazy loading (Zarr/NetCDF only)

    Returns
    -------
    xr.DataArray
        Damage function coefficients

    Examples
    --------
    >>> coefficients = load_damage_function_coefficients(
    ...     "outputs/damage_functions/mortality_2020_coefficients.zarr"
    ... )

    Notes
    -----
    If the file contains a Dataset, extracts the 'coefficients' variable.
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Coefficients file not found: {input_path}")

    if format == "zarr":
        ds = xr.open_zarr(input_path) if use_dask else xr.open_zarr(input_path, chunks=None)
        return ds['coefficients'] if 'coefficients' in ds.data_vars else ds.to_array().squeeze()

    elif format == "netcdf":
        ds = xr.open_dataset(input_path) if use_dask else xr.open_dataset(input_path, chunks=None)
        return ds['coefficients'] if 'coefficients' in ds.data_vars else ds.to_array().squeeze()

    elif format == "csv":
        df = pd.read_csv(input_path, index_col=0)
        # Convert to xarray
        if 'value' in df.columns:
            return xr.DataArray(
                df['value'].values,
                dims=['coefficient'],
                coords={'coefficient': df.index.values}
            )
        else:
            return df.to_xarray()

    else:
        raise ValueError(f"Unknown format: {format}. Choose from: zarr, netcdf, csv")


def save_marginal_damages(
    marginal_damages: xr.DataArray,
    output_path: str,
    format: str = "zarr",
    metadata: Optional[Dict[str, Any]] = None,
    chunks: Optional[Dict[str, int]] = None
) -> None:
    """
    Save marginal damages to disk.

    Parameters
    ----------
    marginal_damages : xr.DataArray
        Marginal damage values (∂damages/∂climate)
    output_path : str
        Path to save marginal damages
    format : str, default "zarr"
        Output format: "zarr" or "netcdf"
    metadata : dict, optional
        Additional metadata
    chunks : dict, optional
        Chunk sizes for Zarr output

    Examples
    --------
    >>> save_marginal_damages(
    ...     marginal_damages,
    ...     "outputs/marginal_damages/mortality_2020_md.zarr",
    ...     chunks={'year': 50, 'region': 10}
    ... )

    Notes
    -----
    Marginal damages are typically large arrays, so Zarr with chunking
    is recommended for efficient storage and retrieval.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Add metadata
    if metadata is not None:
        for key, value in metadata.items():
            marginal_damages.attrs[key] = value

    # Convert to Dataset
    ds = marginal_damages.to_dataset(name='marginal_damages')

    if format == "zarr":
        # Apply chunking if specified
        if chunks is not None:
            ds = ds.chunk(chunks)
        ds.to_zarr(output_path, mode='w')

    elif format == "netcdf":
        # For NetCDF, encoding can control chunking
        encoding = {}
        if chunks is not None:
            encoding['marginal_damages'] = {'chunksizes': tuple(chunks.values())}
        ds.to_netcdf(output_path, encoding=encoding)

    else:
        raise ValueError(f"Unknown format: {format}. Choose from: zarr, netcdf")


def load_marginal_damages(
    input_path: str,
    format: str = "zarr",
    use_dask: bool = True,
    chunks: Optional[Dict[str, int]] = None
) -> xr.DataArray:
    """
    Load marginal damages from disk.

    Parameters
    ----------
    input_path : str
        Path to marginal damages file
    format : str, default "zarr"
        Input format: "zarr" or "netcdf"
    use_dask : bool, default True
        Whether to use Dask for lazy loading
    chunks : dict, optional
        Chunk specification for rechunking

    Returns
    -------
    xr.DataArray
        Marginal damage values

    Examples
    --------
    >>> md = load_marginal_damages(
    ...     "outputs/marginal_damages/mortality_2020_md.zarr",
    ...     chunks={'year': 50, 'region': 10}
    ... )
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Marginal damages file not found: {input_path}")

    if format == "zarr":
        if use_dask:
            ds = xr.open_zarr(input_path)
            if chunks is not None:
                ds = ds.chunk(chunks)
        else:
            ds = xr.open_zarr(input_path, chunks=None)

    elif format == "netcdf":
        if use_dask:
            ds = xr.open_dataset(input_path, chunks=chunks)
        else:
            ds = xr.open_dataset(input_path)

    else:
        raise ValueError(f"Unknown format: {format}. Choose from: zarr, netcdf")

    return ds['marginal_damages'] if 'marginal_damages' in ds.data_vars else ds.to_array().squeeze()


def save_damage_function_points(
    points: xr.Dataset,
    output_path: str,
    format: str = "csv",
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save damage function evaluation points for visualization.

    Parameters
    ----------
    points : xr.Dataset
        Dataset containing climate values and predicted damages
    output_path : str
        Path to save points
    format : str, default "csv"
        Output format: "csv", "zarr", or "netcdf"
    metadata : dict, optional
        Additional metadata

    Examples
    --------
    >>> save_damage_function_points(
    ...     points,
    ...     "outputs/damage_functions/mortality_2020_points.csv"
    ... )

    Notes
    -----
    These are typically used for plotting damage functions.
    CSV format is convenient for quick inspection and plotting.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Add metadata
    if metadata is not None:
        for key, value in metadata.items():
            points.attrs[key] = value

    if format == "csv":
        # Convert to DataFrame
        df = points.to_dataframe()
        df.to_csv(output_path)

    elif format == "zarr":
        points.to_zarr(output_path, mode='w')

    elif format == "netcdf":
        points.to_netcdf(output_path)

    else:
        raise ValueError(f"Unknown format: {format}. Choose from: csv, zarr, netcdf")


def load_damage_function_points(
    input_path: str,
    format: str = "csv"
) -> xr.Dataset:
    """
    Load damage function evaluation points.

    Parameters
    ----------
    input_path : str
        Path to points file
    format : str, default "csv"
        Input format: "csv", "zarr", or "netcdf"

    Returns
    -------
    xr.Dataset
        Damage function evaluation points

    Examples
    --------
    >>> points = load_damage_function_points(
    ...     "outputs/damage_functions/mortality_2020_points.csv"
    ... )
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Points file not found: {input_path}")

    if format == "csv":
        df = pd.read_csv(input_path, index_col=0)
        return df.to_xarray()

    elif format == "zarr":
        return xr.open_zarr(input_path)

    elif format == "netcdf":
        return xr.open_dataset(input_path)

    else:
        raise ValueError(f"Unknown format: {format}. Choose from: csv, zarr, netcdf")


def save_damage_function_summary(
    coefficients: xr.DataArray,
    fit_stats: xr.Dataset,
    output_path: str,
    sector: str,
    pulse_year: int,
    formula: str
) -> None:
    """
    Save human-readable damage function summary.

    Parameters
    ----------
    coefficients : xr.DataArray
        Fitted coefficients
    fit_stats : xr.Dataset
        Fit statistics (rsquared, n_obs, etc.)
    output_path : str
        Path to save summary (JSON or YAML)
    sector : str
        Sector name
    pulse_year : int
        Pulse year
    formula : str
        Damage function formula

    Examples
    --------
    >>> save_damage_function_summary(
    ...     coefficients,
    ...     fit_stats,
    ...     "outputs/damage_functions/mortality_2020_summary.json",
    ...     sector="mortality",
    ...     pulse_year=2020,
    ...     formula="damages ~ -1 + anomaly + np.power(anomaly, 2)"
    ... )

    Notes
    -----
    Creates a readable summary file with key information about the
    fitted damage function for documentation and auditing.
    """
    import json
    from datetime import datetime

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Build summary dictionary
    summary = {
        "sector": sector,
        "pulse_year": pulse_year,
        "formula": formula,
        "timestamp": datetime.now().isoformat(),
        "coefficients": {},
        "fit_statistics": {}
    }

    # Add coefficients
    for coef_name in coefficients.coefficient.values:
        coef_value = float(coefficients.sel(coefficient=coef_name).values)
        summary["coefficients"][coef_name] = coef_value

    # Add fit statistics
    if 'rsquared' in fit_stats.data_vars:
        summary["fit_statistics"]["r_squared"] = float(fit_stats['rsquared'].values)
    if 'n_obs' in fit_stats.data_vars:
        summary["fit_statistics"]["n_observations"] = int(fit_stats['n_obs'].values)

    # Save as JSON
    if output_path.endswith('.json'):
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
    elif output_path.endswith('.yaml') or output_path.endswith('.yml'):
        import yaml
        with open(output_path, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False)
    else:
        # Default to JSON
        output_path = str(path.with_suffix('.json'))
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)


def load_damage_function_summary(
    input_path: str
) -> Dict[str, Any]:
    """
    Load damage function summary.

    Parameters
    ----------
    input_path : str
        Path to summary file (JSON or YAML)

    Returns
    -------
    dict
        Summary information

    Examples
    --------
    >>> summary = load_damage_function_summary(
    ...     "outputs/damage_functions/mortality_2020_summary.json"
    ... )
    >>> print(summary["coefficients"])
    """
    import json

    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Summary file not found: {input_path}")

    if input_path.endswith('.json'):
        with open(input_path, 'r') as f:
            return json.load(f)
    elif input_path.endswith('.yaml') or input_path.endswith('.yml'):
        import yaml
        with open(input_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        # Try JSON first
        try:
            with open(input_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            raise ValueError(f"Cannot determine format for: {input_path}")


def save_full_damage_function(
    coefficients: xr.DataArray,
    marginal_damages: Optional[xr.DataArray],
    points: Optional[xr.Dataset],
    fit_stats: xr.Dataset,
    output_dir: str,
    sector: str,
    pulse_year: int,
    formula: str,
    format: str = "zarr",
    coefficients_grid: Optional[xr.Dataset] = None
) -> Dict[str, str]:
    """
    Save complete damage function output (coefficients, marginal damages, points, summary).

    Parameters
    ----------
    coefficients : xr.DataArray
        Fitted regression coefficients (new format)
    marginal_damages : xr.DataArray, optional
        Marginal damage values
    points : xr.Dataset, optional
        Evaluation points for visualization
    fit_stats : xr.Dataset
        Fit statistics
    output_dir : str
        Output directory
    sector : str
        Sector name
    pulse_year : int
        Pulse year
    formula : str
        Damage function formula
    format : str, default "zarr"
        Format for coefficients and marginal damages
    coefficients_grid : xr.Dataset, optional
        Grid-based predictions (original dscim format)
        Saved separately for comparison with original implementation

    Returns
    -------
    dict
        Paths to all saved files

    Examples
    --------
    >>> paths = save_full_damage_function(
    ...     coefficients, marginal_damages, points, fit_stats,
    ...     output_dir="outputs/damage_functions",
    ...     sector="mortality",
    ...     pulse_year=2020,
    ...     formula="damages ~ -1 + anomaly + np.power(anomaly, 2)"
    ... )
    >>> print(paths["coefficients"])
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create base filename
    base_name = f"{sector}_{pulse_year}"

    saved_paths = {}

    # Save regression coefficients (new format - for comparison)
    coef_ext = "zarr" if format == "zarr" else "nc"
    coef_path = output_path / f"damage_function_coefficients.{coef_ext}"
    save_damage_function_coefficients(coefficients, str(coef_path), format=format)
    saved_paths["coefficients"] = str(coef_path)

    # Save grid predictions (original dscim format)
    if coefficients_grid is not None:
        grid_path = output_path / f"damage_function_coefficients_grid.{coef_ext}"
        if format == "zarr":
            coefficients_grid.to_zarr(str(grid_path), mode='w')
        elif format == "netcdf":
            coefficients_grid.to_netcdf(str(grid_path))
        saved_paths["coefficients_grid"] = str(grid_path)

    # Save marginal damages if provided
    if marginal_damages is not None:
        md_path = output_path / f"marginal_damages.{coef_ext}"
        save_marginal_damages(marginal_damages, str(md_path), format=format)
        saved_paths["marginal_damages"] = str(md_path)

    # Save points if provided
    if points is not None:
        points_path = output_path / f"damage_function_points.csv"
        save_damage_function_points(points, str(points_path), format="csv")
        saved_paths["points"] = str(points_path)

    # Save fit stats
    if fit_stats is not None:
        stats_path = output_path / f"damage_function_fit.{coef_ext}"
        if format == "zarr":
            fit_stats.to_zarr(str(stats_path), mode='w')
        elif format == "netcdf":
            fit_stats.to_netcdf(str(stats_path))
        saved_paths["fit_stats"] = str(stats_path)

    # Save summary
    summary_path = output_path / f"{base_name}_summary.json"
    save_damage_function_summary(
        coefficients, fit_stats, str(summary_path),
        sector=sector, pulse_year=pulse_year, formula=formula
    )
    saved_paths["summary"] = str(summary_path)

    return saved_paths
