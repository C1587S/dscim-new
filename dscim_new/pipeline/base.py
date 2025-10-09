"""
Base classes for pipeline steps.

Provides validation and execution framework for pipeline steps.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import xarray as xr
import logging

from ..config import DSCIMConfig

logger = logging.getLogger(__name__)


class PipelineStep(ABC):
    """
    Base class for all pipeline steps.

    Each step declares its required inputs and outputs, validates them,
    and provides execution logic. Steps can be run independently or
    chained together in a pipeline.

    Parameters
    ----------
    config : DSCIMConfig
        Validated DSCIM configuration
    verbose : bool, optional
        Whether to print progress messages (default: True)

    Examples
    --------
    >>> class MyStep(PipelineStep):
    ...     def required_inputs(self):
    ...         return ["input_data"]
    ...
    ...     def output_keys(self):
    ...         return ["output_data"]
    ...
    ...     def execute(self, inputs):
    ...         # Process inputs
    ...         return {"output_data": processed_data}
    """

    def __init__(self, config: DSCIMConfig, verbose: bool = True):
        self.config = config
        self.verbose = verbose
        self._outputs = {}  # Store in-memory results

        # Configure logging
        if verbose:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.WARNING)

    @abstractmethod
    def required_inputs(self) -> List[str]:
        """
        Return list of required input keys.

        Returns
        -------
        list
            List of input keys this step needs
        """
        pass

    @abstractmethod
    def output_keys(self) -> List[str]:
        """
        Return list of output keys this step produces.

        Returns
        -------
        list
            List of output keys this step generates
        """
        pass

    @abstractmethod
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the step and return outputs.

        Parameters
        ----------
        inputs : dict
            Dictionary of input data/paths

        Returns
        -------
        dict
            Dictionary of output data
        """
        pass

    def validate_inputs(self, inputs: Dict[str, Any]):
        """
        Validate that all required inputs are present and valid.

        Parameters
        ----------
        inputs : dict
            Dictionary of input data/paths

        Raises
        ------
        ValueError
            If required inputs are missing
        FileNotFoundError
            If input files don't exist
        TypeError
            If input types are incorrect
        """
        required = self.required_inputs()
        missing = [key for key in required if key not in inputs or inputs[key] is None]

        if missing:
            raise ValueError(
                f"Step {self.__class__.__name__} missing required inputs: {missing}\n"
                f"Required inputs: {required}\n"
                f"Provided inputs: {list(inputs.keys())}"
            )

        # Validate input types
        for key in required:
            self._validate_input_type(key, inputs[key])

    def _validate_input_type(self, key: str, value: Any):
        """
        Validate input is correct type and format.

        Parameters
        ----------
        key : str
            Input key name
        value : Any
            Input value

        Raises
        ------
        FileNotFoundError
            If path doesn't exist
        TypeError
            If type is incorrect
        """
        if key.endswith("_path"):
            # It's a file path
            if isinstance(value, (str, Path)):
                path = Path(value)
                if not path.exists():
                    raise FileNotFoundError(
                        f"Input file not found for '{key}': {value}\n"
                        f"Please ensure the file exists or check your configuration."
                    )
            else:
                raise TypeError(f"Expected str or Path for '{key}', got {type(value)}")

        elif key.endswith("_data"):
            # It's an xarray Dataset/DataArray
            if not isinstance(value, (xr.Dataset, xr.DataArray)):
                raise TypeError(
                    f"Expected xr.Dataset or xr.DataArray for '{key}', got {type(value)}"
                )

    def run(
        self,
        inputs: Dict[str, Any],
        save: bool = False,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run step with validation.

        Parameters
        ----------
        inputs : dict
            Dictionary of input data/paths
        save : bool, optional
            Whether to save outputs to disk (default: False)
        output_dir : str, optional
            Custom output directory. If None, uses config defaults

        Returns
        -------
        dict
            Dictionary of output data

        Examples
        --------
        >>> inputs = {"input_data": data}
        >>> outputs = step.run(inputs, save=True)
        """
        # Validate inputs
        self.validate_inputs(inputs)

        # Log start
        if self.verbose:
            logger.info(f"Running {self.__class__.__name__}...")

        # Execute
        outputs = self.execute(inputs)

        # Store in memory
        self._outputs.update(outputs)

        # Optionally save
        if save:
            self._save_outputs(outputs, output_dir)

        # Log completion
        if self.verbose:
            logger.info(f"✓ {self.__class__.__name__} complete")

        return outputs

    def _save_outputs(self, outputs: Dict[str, Any], output_dir: Optional[str] = None):
        """
        Save outputs to disk.

        Parameters
        ----------
        outputs : dict
            Dictionary of output data
        output_dir : str, optional
            Custom output directory
        """
        for key, value in outputs.items():
            if isinstance(value, (xr.Dataset, xr.DataArray)):
                output_path = self._get_output_path(key, output_dir)
                self._save_dataset(value, output_path)

                if self.verbose:
                    logger.info(f"  Saved: {output_path}")

    @abstractmethod
    def _get_output_path(self, key: str, output_dir: Optional[str] = None) -> Path:
        """
        Get output path for a key.

        Parameters
        ----------
        key : str
            Output key name
        output_dir : str, optional
            Custom output directory

        Returns
        -------
        Path
            Output file path
        """
        pass

    def _save_dataset(self, data: Union[xr.Dataset, xr.DataArray], path: Path):
        """
        Save dataset in configured format.

        Parameters
        ----------
        data : xr.Dataset or xr.DataArray
            Data to save
        path : Path
            Output file path
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        output_format = self.config.processing.output_format

        if output_format == "zarr":
            if not str(path).endswith('.zarr'):
                path = path.with_suffix('.zarr')
            data.to_zarr(path, consolidated=True, mode='w')

        elif output_format == "netcdf":
            if not str(path).endswith('.nc'):
                path = path.with_suffix('.nc')
            data.to_netcdf(path)

        elif output_format == "csv":
            if not str(path).endswith('.csv'):
                path = path.with_suffix('.csv')
            # Convert to DataFrame
            if isinstance(data, xr.DataArray):
                df = data.to_dataframe()
            else:
                df = data.to_dataframe()
            df.reset_index().to_csv(path, index=False)

    def get_outputs(self) -> Dict[str, Any]:
        """
        Get stored outputs from previous execution.

        Returns
        -------
        dict
            Dictionary of stored outputs
        """
        return self._outputs
