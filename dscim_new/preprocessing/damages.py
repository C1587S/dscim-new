"""
Damage reduction processing.

Provides flexible damage reduction with optional I/O and multiple output formats.
"""

import xarray as xr
import numpy as np
import dask.array as da
from typing import Optional, Dict, Any, Union, Literal
from pathlib import Path
import logging

try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    from rich.panel import Panel
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from ..core import (
    calculate_no_cc_consumption,
    calculate_cc_consumption,
    apply_bottom_coding,
    aggregate_adding_up,
    aggregate_risk_aversion,
)

from ..io import (
    load_damages_data,
    load_socioeconomic_data,
    get_gdppc_for_coordinates,
)

from ..config import DSCIMConfig, SectorConfig

logger = logging.getLogger(__name__)

OutputFormat = Literal["zarr", "netcdf", "csv"]


class DamageProcessor:
    """
    Process and aggregate sectoral damages.

    Flexible processor that can work in-memory or save outputs,
    with support for multiple formats and Dask integration.

    Parameters
    ----------
    config : DSCIMConfig
        Validated configuration object
    sector : str
        Sector name to process
    recipe : str
        Aggregation recipe ("adding_up" or "risk_aversion")
    reduction : str
        Reduction type ("cc" or "no_cc")
    socioec_path : str
        Path to socioeconomic data
    eta : float, optional
        Risk aversion parameter (required for risk_aversion)
    bottom_coding_gdppc : float, optional
        Minimum GDP per capita threshold
    zero : bool, optional
        Whether to zero out historical climate
    quantreg : bool, optional
        Whether to use quantile regression
    use_dask : bool, optional
        Whether to use Dask for chunked processing
    verbose : bool, optional
        Whether to print progress messages

    Examples
    --------
    >>> # Process in memory
    >>> processor = DamageProcessor(
    ...     config=config,
    ...     sector="coastal",
    ...     recipe="adding_up",
    ...     reduction="cc",
    ...     socioec_path="econ.zarr"
    ... )
    >>> result = processor.process()

    >>> # Save with custom path
    >>> result = processor.process(
    ...     save=True,
    ...     output_path="results/damages.zarr"
    ... )
    """

    def __init__(
        self,
        config: DSCIMConfig,
        sector: str,
        recipe: str,
        reduction: str,
        socioec_path: str,
        eta: Optional[float] = None,
        bottom_coding_gdppc: float = 39.39265060424805,
        zero: bool = False,
        quantreg: bool = False,
        use_dask: bool = True,
        verbose: bool = True,
    ):
        self.config = config
        self.sector = sector
        self.recipe = recipe
        self.reduction = reduction
        self.socioec_path = socioec_path
        self.eta = eta
        self.bottom_coding_gdppc = bottom_coding_gdppc
        self.zero = zero
        self.quantreg = quantreg
        self.use_dask = use_dask
        self.verbose = verbose

        # Initialize Rich console
        self.console = Console() if (verbose and RICH_AVAILABLE) else None

        # Get sector configuration
        self.sector_config = config.get_sector_config(sector)

        # Validate parameters
        self._validate_params()

        # Configure logging
        if verbose:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.WARNING)

    def _validate_params(self):
        """Validate parameter combinations."""
        if self.recipe == "adding_up" and self.eta is not None:
            raise ValueError(
                "adding_up recipe does not accept eta parameter. Set eta=None."
            )

        if self.recipe == "risk_aversion":
            if self.quantreg:
                raise ValueError(
                    "Quantile regression incompatible with risk_aversion. Set quantreg=False."
                )
            if self.eta is None:
                raise ValueError("risk_aversion recipe requires eta parameter.")

    def _log(self, message: str, style: str = ""):
        """Log message if verbose."""
        if self.verbose:
            if self.console:
                self.console.print(message, style=style)
            else:
                logger.info(message)

    def _get_chunking_strategy(self, damages_ds: Optional[xr.Dataset]) -> Dict[str, int]:
        """Determine chunking strategy based on sector type and available dimensions."""
        # If dataset provided, use its dimensions
        if damages_ds is not None:
            available_dims = set(damages_ds.dims)
            chunks = {}

            # Always chunk region fully
            if "region" in available_dims:
                chunks["region"] = -1

            # Year chunking
            if "year" in available_dims:
                chunks["year"] = 10

            # Model/scenario dimensions - chunk to 1
            for dim in ["model", "ssp", "rcp", "gcm", "slr"]:
                if dim in available_dims:
                    chunks[dim] = 1

            # Batch dimension
            if self.quantreg and "batch" in available_dims:
                chunks["batch"] = 1

            return chunks

        # Fallback: use sector name to guess (for initial call before loading data)
        if "coastal" not in self.sector:
            chunks = {
                "rcp": 1,
                "region": -1,
                "gcm": 1,
                "year": 10,
                "model": 1,
                "ssp": 1,
            }
        else:
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

    def _create_output_template(
        self, damages_ds: xr.Dataset, gdppc: xr.Dataset
    ) -> xr.DataArray:
        """Create template for output data structure."""
        if self.quantreg:
            ce_batch_dims = [i for i in gdppc.dims] + [
                i for i in damages_ds.dims if i not in gdppc.dims
            ]
        else:
            ce_batch_dims = [i for i in gdppc.dims] + [
                i for i in damages_ds.dims
                if i not in gdppc.dims and i != "batch"
            ]

        ce_batch_coords = {c: damages_ds[c].values for c in ce_batch_dims}
        ce_batch_coords["region"] = [
            r for r in gdppc.region.values if r in ce_batch_coords["region"]
        ]

        ce_shapes = [len(ce_batch_coords[c]) for c in ce_batch_dims]

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
        """Process a single chunk of data using core math functions."""
        year = chunk.year.values
        ssp = chunk.ssp.values
        model = chunk.model.values
        regions = chunk.region.values.tolist()

        gdppc = get_gdppc_for_coordinates(
            self.socioec_path,
            year=year,
            ssp=ssp,
            model=model,
            regions=regions,
            use_dask=self.use_dask,
        )

        if self.reduction == "no_cc":
            histclim = chunk[self.sector_config.histclim]
            histclim_mean = histclim.mean("batch")
            consumption = calculate_no_cc_consumption(gdppc, histclim, histclim_mean)
        else:
            delta = chunk[self.sector_config.delta]
            consumption = calculate_cc_consumption(gdppc, delta)

        consumption = apply_bottom_coding(consumption, self.bottom_coding_gdppc)

        if self.recipe == "adding_up":
            result = aggregate_adding_up(consumption, batch_dim="batch")
        else:
            result = aggregate_risk_aversion(
                consumption, eta=self.eta, batch_dim="batch"
            )

        return result

    def _get_metadata(self) -> Dict[str, Any]:
        """Construct metadata dictionary for output."""
        metadata = {
            "bottom_code": self.bottom_coding_gdppc,
            "histclim_zero": self.zero,
            "filepath": str(self.sector_config.sector_path),
            "recipe": self.recipe,
            "reduction": self.reduction,
            "sector": self.sector,
        }

        if self.recipe == "risk_aversion":
            metadata["eta"] = self.eta

        return metadata

    def _construct_default_output_path(self) -> str:
        """Construct default output path from configuration."""
        base_path = self.config.paths.reduced_damages_library
        sector_dir = Path(base_path) / self.sector
        sector_dir.mkdir(parents=True, exist_ok=True)

        if self.recipe == "adding_up":
            filename = f"{self.recipe}_{self.reduction}"
        else:
            filename = f"{self.recipe}_{self.reduction}_eta{self.eta}"

        # Add extension based on format
        ext_map = {"zarr": ".zarr", "netcdf": ".nc", "csv": ".csv"}
        ext = ext_map.get(self.config.processing.output_format, ".zarr")

        return str(sector_dir / f"{filename}{ext}")

    def process(
        self,
        save: bool = False,
        output_path: Optional[str] = None,
        output_format: Optional[OutputFormat] = None,
    ) -> xr.Dataset:
        """
        Execute damage processing pipeline.

        Parameters
        ----------
        save : bool, optional
            Whether to save results to disk (default: False)
        output_path : str, optional
            Custom output path. If None and save=True, uses config structure
        output_format : str, optional
            Output format when saving (default: from config)

        Returns
        -------
        xr.Dataset
            Processed damage data

        Examples
        --------
        >>> # Process in memory only
        >>> result = processor.process()

        >>> # Save to custom location
        >>> result = processor.process(
        ...     save=True,
        ...     output_path="results/damages.nc",
        ...     output_format="netcdf"
        ... )
        """
        # Print header with Rich if available
        if self.console:
            eta_str = f" | eta={self.eta}" if self.eta else ""
            self.console.print(
                f"[bold blue]Processing {self.sector}[/bold blue] "
                f"[dim]({self.recipe} | {self.reduction}{eta_str})[/dim]"
            )
        else:
            self._log(
                f"Processing: {self.sector} | {self.recipe} | {self.reduction}"
                + (f" | eta={self.eta}" if self.eta else "")
            )

        # Use Rich progress if available
        if self.console and RICH_AVAILABLE:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=self.console,
            ) as progress:
                task = progress.add_task("Loading data...", total=3)

                # Load data without chunks first
                damages_ds = load_damages_data(
                    self.sector_config.sector_path, chunks=None, use_dask=False
                )

                # Now determine chunking based on actual dimensions
                if self.use_dask:
                    chunks = self._get_chunking_strategy(damages_ds)
                    damages_ds = damages_ds.chunk(chunks)

                progress.advance(task)

                gdppc_full = load_socioeconomic_data(
                    self.socioec_path, use_dask=self.use_dask
                )
                progress.update(task, description="Computing damages...")
                progress.advance(task)

                # Create template and process
                template = self._create_output_template(damages_ds, gdppc_full)

                if self.use_dask:
                    result = damages_ds.map_blocks(
                        lambda chunk: self._process_chunk(chunk), template=template
                    )
                else:
                    result = self._process_chunk(damages_ds)

                progress.advance(task)
                progress.update(task, description="Complete", completed=3)
        else:
            # Standard processing without progress bar
            damages_ds = load_damages_data(
                self.sector_config.sector_path, chunks=None, use_dask=False
            )

            # Apply chunking if using Dask
            if self.use_dask:
                chunks = self._get_chunking_strategy(damages_ds)
                damages_ds = damages_ds.chunk(chunks)

            gdppc_full = load_socioeconomic_data(
                self.socioec_path, use_dask=self.use_dask
            )
            template = self._create_output_template(damages_ds, gdppc_full)

            if self.use_dask:
                result = damages_ds.map_blocks(
                    lambda chunk: self._process_chunk(chunk), template=template
                )
            else:
                result = self._process_chunk(damages_ds)

        # Convert to dataset
        result = result.astype(np.float32).rename(self.reduction).to_dataset()

        # Add metadata
        for key, value in self._get_metadata().items():
            result.attrs[key] = value

        # Save if requested
        if save:
            if output_path is None:
                output_path = self._construct_default_output_path()

            # Use specified format or config default
            fmt = output_format or self.config.processing.output_format

            self._save_result(result, output_path, fmt)

        # Print summary if Rich available
        if self.console:
            self._print_result_summary(result)

        return result

    def _print_result_summary(self, result: xr.Dataset):
        """Print formatted result summary using Rich."""
        if not (self.console and RICH_AVAILABLE):
            return

        table = Table(title="Result Summary", show_header=True, header_style="bold cyan")
        table.add_column("Variable", style="cyan", width=12)
        table.add_column("Shape", style="magenta", width=25)
        table.add_column("Mean", style="green", justify="right", width=15)
        table.add_column("Range", style="yellow", justify="right", width=30)

        for var in result.data_vars:
            data = result[var]
            shape_str = str(data.shape)
            mean_val = float(data.mean().values)
            min_val = float(data.min().values)
            max_val = float(data.max().values)

            table.add_row(
                var,
                shape_str,
                f"{mean_val:.2f}",
                f"[{min_val:.2f}, {max_val:.2f}]"
            )

        self.console.print(table)

    def _save_result(self, data: xr.Dataset, output_path: str, output_format: str):
        """Save result in specified format."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.console:
            self.console.print(f"[dim]Saving to {output_path} ({output_format} format)[/dim]")
        else:
            self._log(f"Saving to {output_path} ({output_format} format)")

        if output_format == "zarr":
            data.to_zarr(output_path, consolidated=True, mode="w")
        elif output_format == "netcdf":
            data.to_netcdf(output_path)
        elif output_format == "csv":
            df = data.to_dataframe().reset_index()
            df.to_csv(output_path, index=False)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

        if self.console:
            self.console.print(f"[green]Saved:[/green] {output_path}")
        else:
            self._log(f"Saved: {output_path}")


def reduce_damages(
    config: Union[str, Dict, DSCIMConfig],
    sector: str,
    recipe: str,
    reduction: str,
    socioec_path: str,
    eta: Optional[float] = None,
    bottom_coding_gdppc: float = 39.39265060424805,
    zero: bool = False,
    quantreg: bool = False,
    use_dask: bool = True,
    save: bool = False,
    output_path: Optional[str] = None,
    output_format: Optional[OutputFormat] = None,
    verbose: bool = True,
) -> xr.Dataset:
    """
    Process and aggregate sectoral damages.

    Parameters
    ----------
    config : str, dict, or DSCIMConfig
        Configuration (path to YAML, dict, or validated config object)
    sector : str
        Sector name to process
    recipe : str
        Aggregation recipe ("adding_up" or "risk_aversion")
    reduction : str
        Reduction type ("cc" or "no_cc")
    socioec_path : str
        Path to socioeconomic data
    eta : float, optional
        Risk aversion parameter (required for risk_aversion)
    bottom_coding_gdppc : float, optional
        Minimum GDP per capita threshold
    zero : bool, optional
        Whether to zero out historical climate
    quantreg : bool, optional
        Whether to use quantile regression
    use_dask : bool, optional
        Whether to use Dask for chunked processing (default: True)
    save : bool, optional
        Whether to save results to disk (default: False)
    output_path : str, optional
        Custom output path. If None and save=True, uses config structure
    output_format : str, optional
        Output format when saving ("zarr", "netcdf", or "csv")
    verbose : bool, optional
        Whether to print progress messages (default: True)

    Returns
    -------
    xr.Dataset
        Processed damage data

    Examples
    --------
    >>> # Process in memory only
    >>> result = reduce_damages(
    ...     config="config.yaml",
    ...     sector="coastal",
    ...     recipe="adding_up",
    ...     reduction="cc",
    ...     socioec_path="econ.zarr"
    ... )

    >>> # Process and save
    >>> result = reduce_damages(
    ...     config="config.yaml",
    ...     sector="coastal",
    ...     recipe="adding_up",
    ...     reduction="cc",
    ...     socioec_path="econ.zarr",
    ...     save=True
    ... )
    """
    # Load and validate config if needed
    if isinstance(config, str):
        config = DSCIMConfig.from_yaml(config)
    elif isinstance(config, dict):
        config = DSCIMConfig.from_dict(config)

    # Validate config for this operation
    config.validate_for_reduce_damages(sector)

    # Create processor
    processor = DamageProcessor(
        config=config,
        sector=sector,
        recipe=recipe,
        reduction=reduction,
        socioec_path=socioec_path,
        eta=eta,
        bottom_coding_gdppc=bottom_coding_gdppc,
        zero=zero,
        quantreg=quantreg,
        use_dask=use_dask,
        verbose=verbose,
    )

    return processor.process(save=save, output_path=output_path, output_format=output_format)
