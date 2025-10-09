"""
Data writing functions for DSCIM.

All file writing operations are isolated here.
"""

import xarray as xr
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path


def construct_output_path(
    base_path: str,
    sector: str,
    recipe: str,
    reduction: str,
    eta: Optional[float] = None
) -> str:
    """
    Construct output file path following DSCIM naming convention.

    Parameters
    ----------
    base_path : str
        Base directory for reduced damages library
    sector : str
        Sector name
    recipe : str
        Recipe type ("adding_up" or "risk_aversion")
    reduction : str
        Reduction type ("cc" or "no_cc")
    eta : float, optional
        Risk aversion parameter (required for risk_aversion recipe)

    Returns
    -------
    str
        Full output path

    Examples
    --------
    >>> construct_output_path(
    ...     "./dummy_data/reduced_damages",
    ...     "dummy_coastal_sector",
    ...     "adding_up",
    ...     "cc"
    ... )
    './dummy_data/reduced_damages/dummy_coastal_sector/adding_up_cc.zarr'

    >>> construct_output_path(
    ...     "./dummy_data/reduced_damages",
    ...     "dummy_coastal_sector",
    ...     "risk_aversion",
    ...     "cc",
    ...     eta=2.0
    ... )
    './dummy_data/reduced_damages/dummy_coastal_sector/risk_aversion_cc_eta2.0.zarr'

    Notes
    -----
    This matches the original DSCIM naming convention:
    - adding_up: {sector}/{recipe}_{reduction}.zarr
    - risk_aversion: {sector}/{recipe}_{reduction}_eta{eta}.zarr
    """
    sector_dir = Path(base_path) / sector
    sector_dir.mkdir(parents=True, exist_ok=True)

    if recipe == "adding_up":
        filename = f"{recipe}_{reduction}.zarr"
    elif recipe == "risk_aversion":
        if eta is None:
            raise ValueError("eta must be specified for risk_aversion recipe")
        filename = f"{recipe}_{reduction}_eta{eta}.zarr"
    else:
        raise ValueError(f"Unknown recipe: {recipe}")

    return str(sector_dir / filename)


def save_reduced_damages(
    data: xr.Dataset,
    output_path: str,
    metadata: Dict[str, Any],
    consolidated: bool = True
) -> None:
    """
    Save reduced damages to zarr file with metadata.

    Parameters
    ----------
    data : xr.Dataset
        Reduced damages dataset to save
    output_path : str
        Output file path (should end in .zarr)
    metadata : dict
        Metadata attributes to attach to dataset
        Expected keys: bottom_code, histclim_zero, filepath, eta (optional)
    consolidated : bool, optional
        Whether to consolidate zarr metadata (default: True)

    Examples
    --------
    >>> data = xr.Dataset({"cc": reduced_damages_array})
    >>> metadata = {
    ...     "bottom_code": 39.39265060424805,
    ...     "histclim_zero": False,
    ...     "filepath": "./dummy_data/sectoral/coastal_damages.zarr",
    ...     "eta": 2.0
    ... }
    >>> save_reduced_damages(data, "./output/result.zarr", metadata)

    Notes
    -----
    The function:
    1. Converts data to float32 for storage efficiency
    2. Attaches metadata as dataset attributes
    3. Writes to zarr format with optional consolidation
    4. Uses mode='w' to overwrite existing data
    """
    # Ensure data is float32 for storage efficiency
    if not all(data[var].dtype == np.float32 for var in data.data_vars):
        data = data.astype(np.float32)

    # Attach metadata
    for key, value in metadata.items():
        data.attrs[key] = value

    # Save to zarr
    data.to_zarr(
        output_path,
        consolidated=consolidated,
        mode='w'
    )