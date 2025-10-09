"""
Reduce damages pipeline implementation.

This module orchestrates the reduce_damages workflow, coordinating
I/O operations and mathematical computations. Dask usage is optional
and controlled by the user.
"""

import xarray as xr
import numpy as np
import dask.array as da
from typing import Optional, Dict, Any
from pathlib import Path

from ..core import (
    calculate_no_cc_consumption,
    calculate_cc_consumption,
    apply_bottom_coding,
    aggregate_adding_up,
    aggregate_risk_aversion,
)

from ..io import (
    load_config,
    get_sector_config,
    load_damages_data,
    load_socioeconomic_data,
    get_gdppc_for_coordinates,
    save_reduced_damages,
    construct_output_path,
)


class ReduceDamagesPipeline:
    """
    Pipeline for reducing damages with separation of I/O and computation.

    This pipeline coordinates the reduce_damages workflow while keeping
    I/O operations separate from mathematical computations. Users can
    control whether to use Dask for distributed/chunked processing.

    Parameters
    ----------
    config_path : str
        Path to YAML configuration file
    sector : str
        Sector name to process
    recipe : str
        Aggregation recipe ("adding_up" or "risk_aversion")
    reduction : str
        Reduction type ("cc" or "no_cc")
    socioec : str
        Path to socioeconomic data zarr file
    eta : float, optional
        Risk aversion parameter (required for risk_aversion recipe)
    bottom_coding_gdppc : float, optional
        Minimum GDP per capita threshold (default: 39.39265060424805)
    zero : bool, optional
        Whether to zero out historical climate (default: False)
    quantreg : bool, optional
        Whether to use quantile regression (default: False)
    use_dask : bool, optional
        Whether to use Dask for chunked processing (default: True)
        Set to False for immediate computation in memory

    Examples
    --------
    >>> # With Dask (default - lazy/chunked processing)
    >>> pipeline = ReduceDamagesPipeline(
    ...     config_path="configs/dummy_config.yaml",
    ...     sector="dummy_coastal_sector",
    ...     recipe="adding_up",
    ...     reduction="cc",
    ...     socioec="./dummy_data/econ/integration-econ-bc39.zarr"
    ... )
    >>> output_path = pipeline.run()

    >>> # Without Dask (immediate computation in memory)
    >>> pipeline = ReduceDamagesPipeline(
    ...     config_path="configs/dummy_config.yaml",
    ...     sector="dummy_coastal_sector",
    ...     recipe="adding_up",
    ...     reduction="cc",
    ...     socioec="./dummy_data/econ/integration-econ-bc39.zarr",
    ...     use_dask=False
    ... )
    >>> output_path = pipeline.run()

    Notes
    -----
    When use_dask=True:
    - Data is loaded lazily
    - Processing happens in chunks using map_blocks
    - Lower memory usage, can handle large datasets

    When use_dask=False:
    - Data is loaded into memory immediately
    - Processing happens on full arrays
    - Faster for small datasets, requires sufficient memory
    """

    def __init__(
        self,
        config_path: str,
        sector: str,
        recipe: str,
        reduction: str,
        socioec: str,
        eta: Optional[float] = None,
        bottom_coding_gdppc: float = 39.39265060424805,
        zero: bool = False,
        quantreg: bool = False,
        use_dask: bool = True,
    ):
        self.config_path = config_path
        self.sector = sector
        self.recipe = recipe
        self.reduction = reduction
        self.socioec = socioec
        self.eta = eta
        self.bottom_coding_gdppc = bottom_coding_gdppc
        self.zero = zero
        self.quantreg = quantreg
        self.use_dask = use_dask

        # Load configuration
        self.config = load_config(config_path)
        self.sector_config = get_sector_config(self.config, sector)

        # Validate parameters
        self._validate_params()

    def _validate_params(self):
        """Validate parameter combinations."""
        if self.recipe == "adding_up":
            if self.eta is not None:
                raise ValueError("Adding up recipe does not take eta parameter. Set eta=None.")

        if self.recipe == "risk_aversion":
            if self.quantreg:
                raise ValueError("Quantile regression incompatible with risk aversion. Set quantreg=False.")
            if self.eta is None:
                raise ValueError("Risk aversion recipe requires eta parameter.")

    def _get_chunking_strategy(self, damages_ds: xr.Dataset) -> Dict[str, int]:
        """
        Determine chunking strategy based on sector type.

        Parameters
        ----------
        damages_ds : xr.Dataset
            Damages dataset to determine dimensions

        Returns
        -------
        dict
            Chunking specification
        """
        if "coastal" not in self.sector:
            # Non-coastal sector
            chunks = {
                "rcp": 1,
                "region": -1,
                "gcm": 1,
                "year": 10,
                "model": 1,
                "ssp": 1,
            }
        else:
            # Coastal sector
            chunks = {
                "region": -1,
                "slr": 1,
                "year": 10,
                "model": 1,
                "ssp": 1,
            }

        if self.quantreg:
            chunks["batch"] = 1

        return chunks

    def _create_output_template(self, damages_ds: xr.Dataset, gdppc: xr.Dataset) -> xr.DataArray:
        """
        Create template for output data structure.

        Parameters
        ----------
        damages_ds : xr.Dataset
            Damages dataset
        gdppc : xr.Dataset
            GDP per capita dataset

        Returns
        -------
        xr.DataArray
            Empty template with correct dimensions and coordinates
        """
        # Determine output dimensions
        if self.quantreg:
            ce_batch_dims = [i for i in gdppc.dims] + [
                i for i in damages_ds.dims if i not in gdppc.dims
            ]
        else:
            ce_batch_dims = [i for i in gdppc.dims] + [
                i for i in damages_ds.dims if i not in gdppc.dims and i != "batch"
            ]

        # Create coordinates dictionary
        ce_batch_coords = {c: damages_ds[c].values for c in ce_batch_dims}
        ce_batch_coords["region"] = [
            r for r in gdppc.region.values if r in ce_batch_coords["region"]
        ]

        ce_shapes = [len(ce_batch_coords[c]) for c in ce_batch_dims]

        # Create template
        template = xr.DataArray(
            da.empty(ce_shapes) if self.use_dask else np.empty(ce_shapes),
            dims=ce_batch_dims,
            coords=ce_batch_coords,
        )

        if self.use_dask:
            chunks = self._get_chunking_strategy(damages_ds)
            template = template.chunk(chunks)

        return template

    def _process_chunk(self, chunk: xr.Dataset) -> xr.DataArray:
        """
        Process a single chunk of data using pure math functions.

        This is the core processing function that applies the mathematical
        operations to transform damages data.

        Parameters
        ----------
        chunk : xr.Dataset
            Chunk of damages data

        Returns
        -------
        xr.DataArray
            Processed chunk with reduced damages
        """
        # Extract coordinates from chunk
        year = chunk.year.values
        ssp = chunk.ssp.values
        model = chunk.model.values

        # Determine which regions to load
        regions = chunk.region.values.tolist()

        # Load GDP per capita for this chunk's coordinates
        gdppc = get_gdppc_for_coordinates(
            self.socioec,
            year=year,
            ssp=ssp,
            model=model,
            regions=regions,
            use_dask=self.use_dask
        )

        # Calculate consumption based on reduction type
        if self.reduction == "no_cc":
            histclim = chunk[self.sector_config.histclim]
            histclim_mean = histclim.mean("batch")
            consumption = calculate_no_cc_consumption(gdppc, histclim, histclim_mean)
        else:  # "cc"
            delta = chunk[self.sector_config.delta]
            consumption = calculate_cc_consumption(gdppc, delta)

        # Apply bottom coding
        consumption = apply_bottom_coding(consumption, self.bottom_coding_gdppc)

        # Aggregate using appropriate recipe
        if self.recipe == "adding_up":
            result = aggregate_adding_up(consumption, batch_dim="batch")
        else:  # "risk_aversion"
            result = aggregate_risk_aversion(consumption, eta=self.eta, batch_dim="batch")

        return result

    def _get_metadata(self) -> Dict[str, Any]:
        """
        Construct metadata dictionary for output file.

        Returns
        -------
        dict
            Metadata to attach to output dataset
        """
        metadata = {
            "bottom_code": self.bottom_coding_gdppc,
            "histclim_zero": self.zero,
            "filepath": str(self.sector_config.sector_path),
        }

        if self.recipe == "risk_aversion":
            metadata["eta"] = self.eta

        return metadata

    def run(self) -> str:
        """
        Execute the full reduce damages pipeline.

        Returns
        -------
        str
            Path to output zarr file

        Examples
        --------
        >>> pipeline = ReduceDamagesPipeline(...)
        >>> output_path = pipeline.run()
        >>> print(f"Results saved to: {output_path}")
        """
        # 1. Load damages data
        chunks = self._get_chunking_strategy(None) if self.use_dask else None
        damages_ds = load_damages_data(
            self.sector_config.sector_path,
            chunks=chunks,
            use_dask=self.use_dask
        )

        # 2. Load socioeconomic data (for template creation)
        gdppc_full = load_socioeconomic_data(
            self.socioec,
            use_dask=self.use_dask
        )

        # 3. Create output template
        template = self._create_output_template(damages_ds, gdppc_full)

        # 4. Process data
        if self.use_dask:
            # Use map_blocks for chunked processing
            result = damages_ds.map_blocks(
                lambda chunk: self._process_chunk(chunk),
                template=template
            )
        else:
            # Process entire dataset at once
            result = self._process_chunk(damages_ds)

        # 5. Convert to dataset and set data type
        result = result.astype(np.float32).rename(self.reduction).to_dataset()

        # 6. Construct output path
        output_path = construct_output_path(
            self.config["paths"]["reduced_damages_library"],
            self.sector,
            self.recipe,
            self.reduction,
            self.eta
        )

        # 7. Save results
        save_reduced_damages(
            result,
            output_path,
            self._get_metadata()
        )

        return output_path


def reduce_damages_refactored(
    recipe: str,
    reduction: str,
    eta: Optional[float],
    sector: str,
    config: str,
    socioec: str,
    bottom_coding_gdppc: float = 39.39265060424805,
    zero: bool = False,
    quantreg: bool = False,
    use_dask: bool = True,
) -> str:
    """
    Refactored reduce_damages function matching original signature.

    This is a convenience function that wraps the ReduceDamagesPipeline
    class, providing a functional interface similar to the original DSCIM.

    Parameters
    ----------
    recipe : str
        Aggregation recipe ("adding_up" or "risk_aversion")
    reduction : str
        Reduction type ("cc" or "no_cc")
    eta : float or None
        Risk aversion parameter (required for risk_aversion, None for adding_up)
    sector : str
        Sector name to process
    config : str
        Path to YAML configuration file
    socioec : str
        Path to socioeconomic data zarr file
    bottom_coding_gdppc : float, optional
        Minimum GDP per capita threshold (default: 39.39265060424805)
    zero : bool, optional
        Whether to zero out historical climate (default: False)
    quantreg : bool, optional
        Whether to use quantile regression (default: False)
    use_dask : bool, optional
        Whether to use Dask for chunked processing (default: True)

    Returns
    -------
    str
        Path to output zarr file

    Examples
    --------
    >>> # Using Dask (default)
    >>> output_path = reduce_damages_refactored(
    ...     recipe="adding_up",
    ...     reduction="cc",
    ...     eta=None,
    ...     sector="dummy_coastal_sector",
    ...     config="configs/dummy_config.yaml",
    ...     socioec="./dummy_data/econ/integration-econ-bc39.zarr"
    ... )

    >>> # Without Dask
    >>> output_path = reduce_damages_refactored(
    ...     recipe="adding_up",
    ...     reduction="cc",
    ...     eta=None,
    ...     sector="dummy_coastal_sector",
    ...     config="configs/dummy_config.yaml",
    ...     socioec="./dummy_data/econ/integration-econ-bc39.zarr",
    ...     use_dask=False
    ... )

    Notes
    -----
    This function provides a drop-in replacement for the original
    reduce_damages() function, with the addition of the use_dask parameter
    to control whether Dask is used for processing.
    """
    pipeline = ReduceDamagesPipeline(
        config_path=config,
        sector=sector,
        recipe=recipe,
        reduction=reduction,
        socioec=socioec,
        eta=eta,
        bottom_coding_gdppc=bottom_coding_gdppc,
        zero=zero,
        quantreg=quantreg,
        use_dask=use_dask,
    )

    return pipeline.run()