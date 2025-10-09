"""
I/O functions for Social Cost of Carbon (SCC) data.

Saving and loading SCC results, discount factors, and aggregated values.
"""

import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List


def save_scc(
    scc: xr.DataArray,
    output_path: str,
    format: str = "zarr",
    metadata: Optional[Dict[str, Any]] = None,
    chunks: Optional[Dict[str, int]] = None
) -> None:
    """
    Save Social Cost of Carbon values to disk.

    Parameters
    ----------
    scc : xr.DataArray
        SCC values
    output_path : str
        Path to save SCC
    format : str, default "zarr"
        Output format: "zarr", "netcdf", or "csv"
    metadata : dict, optional
        Additional metadata
    chunks : dict, optional
        Chunk sizes for Zarr output

    Examples
    --------
    >>> save_scc(
    ...     scc,
    ...     "outputs/scc/mortality_2020_scc.zarr",
    ...     format="zarr"
    ... )

    Notes
    -----
    For high-dimensional SCC (with simulation, model, scenario dimensions),
    Zarr format with chunking is recommended.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Add metadata
    if metadata is not None:
        for key, value in metadata.items():
            scc.attrs[key] = value

    # Convert to Dataset
    ds = scc.to_dataset(name='scc')

    if format == "zarr":
        if chunks is not None:
            ds = ds.chunk(chunks)
        ds.to_zarr(output_path, mode='w')

    elif format == "netcdf":
        encoding = {}
        if chunks is not None:
            encoding['scc'] = {'chunksizes': tuple(chunks.values())}
        ds.to_netcdf(output_path, encoding=encoding)

    elif format == "csv":
        # For CSV, convert to DataFrame
        df = scc.to_dataframe(name='scc').reset_index()
        df.to_csv(output_path, index=False)

    else:
        raise ValueError(f"Unknown format: {format}. Choose from: zarr, netcdf, csv")


def load_scc(
    input_path: str,
    format: str = "zarr",
    use_dask: bool = True,
    chunks: Optional[Dict[str, int]] = None
) -> xr.DataArray:
    """
    Load Social Cost of Carbon values from disk.

    Parameters
    ----------
    input_path : str
        Path to SCC file
    format : str, default "zarr"
        Input format: "zarr", "netcdf", or "csv"
    use_dask : bool, default True
        Whether to use Dask for lazy loading
    chunks : dict, optional
        Chunk specification

    Returns
    -------
    xr.DataArray
        SCC values

    Examples
    --------
    >>> scc = load_scc("outputs/scc/mortality_2020_scc.zarr")
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"SCC file not found: {input_path}")

    if format == "zarr":
        if use_dask:
            ds = xr.open_zarr(input_path)
            if chunks is not None:
                ds = ds.chunk(chunks)
        else:
            ds = xr.open_zarr(input_path, chunks=None)
        return ds['scc'] if 'scc' in ds.data_vars else ds.to_array().squeeze()

    elif format == "netcdf":
        if use_dask:
            ds = xr.open_dataset(input_path, chunks=chunks)
        else:
            ds = xr.open_dataset(input_path)
        return ds['scc'] if 'scc' in ds.data_vars else ds.to_array().squeeze()

    elif format == "csv":
        df = pd.read_csv(input_path)
        # Identify index columns (all non-'scc' columns)
        index_cols = [col for col in df.columns if col != 'scc']
        if index_cols:
            df = df.set_index(index_cols)
        return df.to_xarray()['scc']

    else:
        raise ValueError(f"Unknown format: {format}. Choose from: zarr, netcdf, csv")


def save_discount_factors(
    discount_factors: xr.DataArray,
    output_path: str,
    format: str = "zarr",
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save discount factors to disk.

    Parameters
    ----------
    discount_factors : xr.DataArray
        Discount factor values
    output_path : str
        Path to save discount factors
    format : str, default "zarr"
        Output format: "zarr", "netcdf", or "csv"
    metadata : dict, optional
        Additional metadata

    Examples
    --------
    >>> save_discount_factors(
    ...     discount_factors,
    ...     "outputs/discount_factors/ramsey_eta1.45_rho0.001.zarr"
    ... )

    Notes
    -----
    Discount factors are typically reused across multiple SCC calculations,
    so saving them separately can improve efficiency.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Add metadata
    if metadata is not None:
        for key, value in metadata.items():
            discount_factors.attrs[key] = value

    ds = discount_factors.to_dataset(name='discount_factors')

    if format == "zarr":
        ds.to_zarr(output_path, mode='w')

    elif format == "netcdf":
        ds.to_netcdf(output_path)

    elif format == "csv":
        df = discount_factors.to_dataframe(name='discount_factors').reset_index()
        df.to_csv(output_path, index=False)

    else:
        raise ValueError(f"Unknown format: {format}. Choose from: zarr, netcdf, csv")


def load_discount_factors(
    input_path: str,
    format: str = "zarr",
    use_dask: bool = True
) -> xr.DataArray:
    """
    Load discount factors from disk.

    Parameters
    ----------
    input_path : str
        Path to discount factors file
    format : str, default "zarr"
        Input format: "zarr", "netcdf", or "csv"
    use_dask : bool, default True
        Whether to use Dask

    Returns
    -------
    xr.DataArray
        Discount factors

    Examples
    --------
    >>> df = load_discount_factors(
    ...     "outputs/discount_factors/ramsey_eta1.45_rho0.001.zarr"
    ... )
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Discount factors file not found: {input_path}")

    if format == "zarr":
        ds = xr.open_zarr(input_path) if use_dask else xr.open_zarr(input_path, chunks=None)
        return ds['discount_factors'] if 'discount_factors' in ds.data_vars else ds.to_array().squeeze()

    elif format == "netcdf":
        ds = xr.open_dataset(input_path) if use_dask else xr.open_dataset(input_path, chunks=None)
        return ds['discount_factors'] if 'discount_factors' in ds.data_vars else ds.to_array().squeeze()

    elif format == "csv":
        df = pd.read_csv(input_path)
        index_cols = [col for col in df.columns if col != 'discount_factors']
        if index_cols:
            df = df.set_index(index_cols)
        return df.to_xarray()['discount_factors']

    else:
        raise ValueError(f"Unknown format: {format}. Choose from: zarr, netcdf, csv")


def save_scc_quantiles(
    quantiles: xr.Dataset,
    output_path: str,
    format: str = "csv",
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save SCC quantiles to disk.

    Parameters
    ----------
    quantiles : xr.Dataset
        SCC quantile values
    output_path : str
        Path to save quantiles
    format : str, default "csv"
        Output format: "csv", "zarr", or "netcdf"
    metadata : dict, optional
        Additional metadata

    Examples
    --------
    >>> save_scc_quantiles(
    ...     quantiles,
    ...     "outputs/scc/mortality_2020_quantiles.csv"
    ... )

    Notes
    -----
    Quantiles are often used for reporting uncertainty bounds,
    so CSV format is convenient for tables and visualizations.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Add metadata
    if metadata is not None:
        for key, value in metadata.items():
            quantiles.attrs[key] = value

    if format == "csv":
        # Handle both Dataset and DataArray
        if isinstance(quantiles, xr.DataArray):
            df = quantiles.to_dataframe(name='scc').reset_index()
        else:
            df = quantiles.to_dataframe().reset_index()
        df.to_csv(output_path, index=False)

    elif format == "zarr":
        quantiles.to_zarr(output_path, mode='w')

    elif format == "netcdf":
        quantiles.to_netcdf(output_path)

    else:
        raise ValueError(f"Unknown format: {format}. Choose from: csv, zarr, netcdf")


def load_scc_quantiles(
    input_path: str,
    format: str = "csv"
) -> xr.Dataset:
    """
    Load SCC quantiles from disk.

    Parameters
    ----------
    input_path : str
        Path to quantiles file
    format : str, default "csv"
        Input format: "csv", "zarr", or "netcdf"

    Returns
    -------
    xr.Dataset
        SCC quantiles

    Examples
    --------
    >>> quantiles = load_scc_quantiles(
    ...     "outputs/scc/mortality_2020_quantiles.csv"
    ... )
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Quantiles file not found: {input_path}")

    if format == "csv":
        df = pd.read_csv(input_path)
        return df.to_xarray()

    elif format == "zarr":
        return xr.open_zarr(input_path)

    elif format == "netcdf":
        return xr.open_dataset(input_path)

    else:
        raise ValueError(f"Unknown format: {format}. Choose from: csv, zarr, netcdf")


def save_scc_summary(
    scc_mean: float,
    scc_median: float,
    scc_quantiles: Optional[Dict[str, float]],
    output_path: str,
    sector: str,
    pulse_year: int,
    discount_type: str,
    eta: Optional[float] = None,
    rho: Optional[float] = None,
    discount_rate: Optional[float] = None
) -> None:
    """
    Save human-readable SCC summary.

    Parameters
    ----------
    scc_mean : float
        Mean SCC value
    scc_median : float
        Median SCC value
    scc_quantiles : dict, optional
        Quantile values (e.g., {0.05: 10.5, 0.95: 45.2})
    output_path : str
        Path to save summary (JSON or YAML)
    sector : str
        Sector name
    pulse_year : int
        Pulse year
    discount_type : str
        Type of discounting used
    eta : float, optional
        Eta parameter (for Ramsey/GWR)
    rho : float, optional
        Rho parameter (for Ramsey/GWR)
    discount_rate : float, optional
        Discount rate (for constant discounting)

    Examples
    --------
    >>> save_scc_summary(
    ...     scc_mean=42.5,
    ...     scc_median=38.2,
    ...     scc_quantiles={0.05: 15.3, 0.95: 78.9},
    ...     output_path="outputs/scc/mortality_2020_summary.json",
    ...     sector="mortality",
    ...     pulse_year=2020,
    ...     discount_type="ramsey",
    ...     eta=1.45,
    ...     rho=0.001
    ... )
    """
    import json
    from datetime import datetime

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Build summary dictionary
    summary = {
        "sector": sector,
        "pulse_year": pulse_year,
        "timestamp": datetime.now().isoformat(),
        "discounting": {
            "type": discount_type
        },
        "scc_values": {
            "mean": float(scc_mean),
            "median": float(scc_median)
        }
    }

    # Add discount parameters
    if discount_type in ["ramsey", "gwr"]:
        if eta is not None:
            summary["discounting"]["eta"] = eta
        if rho is not None:
            summary["discounting"]["rho"] = rho
    elif discount_type == "constant":
        if discount_rate is not None:
            summary["discounting"]["rate"] = discount_rate

    # Add quantiles
    if scc_quantiles:
        summary["scc_values"]["quantiles"] = {
            str(q): float(v) for q, v in scc_quantiles.items()
        }

    # Save as JSON
    if output_path.endswith('.json'):
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
    elif output_path.endswith('.yaml') or output_path.endswith('.yml'):
        import yaml
        with open(output_path, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False)
    else:
        output_path = str(path.with_suffix('.json'))
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)


def load_scc_summary(
    input_path: str
) -> Dict[str, Any]:
    """
    Load SCC summary from disk.

    Parameters
    ----------
    input_path : str
        Path to summary file

    Returns
    -------
    dict
        Summary information

    Examples
    --------
    >>> summary = load_scc_summary("outputs/scc/mortality_2020_summary.json")
    >>> print(summary["scc_values"]["mean"])
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
        try:
            with open(input_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            raise ValueError(f"Cannot determine format for: {input_path}")


def save_full_scc(
    scc: xr.DataArray,
    discount_factors: Optional[xr.DataArray],
    quantiles: Optional[xr.Dataset],
    output_dir: str,
    sector: str,
    pulse_year: int,
    discount_type: str,
    format: str = "zarr",
    **discount_params
) -> Dict[str, str]:
    """
    Save complete SCC output (SCC values, discount factors, quantiles, summary).

    Parameters
    ----------
    scc : xr.DataArray
        SCC values
    discount_factors : xr.DataArray, optional
        Discount factors used
    quantiles : xr.Dataset, optional
        SCC quantiles
    output_dir : str
        Output directory
    sector : str
        Sector name
    pulse_year : int
        Pulse year
    discount_type : str
        Type of discounting
    format : str, default "zarr"
        Format for main outputs
    **discount_params
        Additional discount parameters (eta, rho, discount_rate, etc.)

    Returns
    -------
    dict
        Paths to all saved files

    Examples
    --------
    >>> paths = save_full_scc(
    ...     scc, discount_factors, quantiles,
    ...     output_dir="outputs/scc",
    ...     sector="mortality",
    ...     pulse_year=2020,
    ...     discount_type="ramsey",
    ...     eta=1.45,
    ...     rho=0.001
    ... )
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create base filename
    base_name = f"{sector}_{pulse_year}_{discount_type}"
    if 'eta' in discount_params:
        base_name += f"_eta{discount_params['eta']}"
    if 'rho' in discount_params:
        base_name += f"_rho{discount_params['rho']}"
    if 'discount_rate' in discount_params:
        base_name += f"_dr{discount_params['discount_rate']}"

    saved_paths = {}

    # Save SCC
    ext = "zarr" if format == "zarr" else "nc"
    scc_path = output_path / f"{base_name}_scc.{ext}"
    save_scc(scc, str(scc_path), format=format)
    saved_paths["scc"] = str(scc_path)

    # Save discount factors if provided
    if discount_factors is not None:
        df_path = output_path / f"{base_name}_discount_factors.{ext}"
        save_discount_factors(discount_factors, str(df_path), format=format)
        saved_paths["discount_factors"] = str(df_path)

    # Save quantiles if provided
    if quantiles is not None:
        quant_path = output_path / f"{base_name}_quantiles.csv"
        save_scc_quantiles(quantiles, str(quant_path), format="csv")
        saved_paths["quantiles"] = str(quant_path)

    # Calculate summary statistics
    # Handle Dask arrays - compute if necessary
    if hasattr(scc.data, 'compute'):
        # Dask array - need to compute for summary stats
        scc_computed = scc.compute()
        # Compute over all remaining dimensions
        all_dims = list(scc_computed.dims)
        if all_dims:
            scc_mean = float(scc_computed.mean(dim=all_dims).values)
            scc_median = float(scc_computed.median(dim=all_dims).values)
        else:
            scc_mean = float(scc_computed.values)
            scc_median = float(scc_computed.values)
    else:
        # Regular numpy array
        scc_mean = float(scc.mean().values)
        scc_median = float(scc.median().values)

    quant_dict = None
    if quantiles is not None:
        if isinstance(quantiles, xr.DataArray) and 'probability' in quantiles.dims:
            quant_dict = {
                float(p): float(quantiles.sel(probability=p).mean().values)
                for p in quantiles.probability.values
            }
        elif isinstance(quantiles, xr.Dataset) and 'probability' in quantiles.dims:
            # For Dataset, get the first data variable
            var_name = list(quantiles.data_vars)[0]
            quant_dict = {
                float(p): float(quantiles[var_name].sel(probability=p).mean().values)
                for p in quantiles.probability.values
            }

    # Save summary
    summary_path = output_path / f"{base_name}_summary.json"
    save_scc_summary(
        scc_mean, scc_median, quant_dict,
        str(summary_path),
        sector=sector,
        pulse_year=pulse_year,
        discount_type=discount_type,
        **discount_params
    )
    saved_paths["summary"] = str(summary_path)

    return saved_paths
