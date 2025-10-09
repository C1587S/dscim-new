"""
Output file naming conventions.

Centralizes naming logic to ensure consistency with original implementation.
This module ensures that output files match the exact naming format used
in the original dscim library.
"""

from pathlib import Path
from typing import Optional, Union


class OutputNaming:
    """
    Standardized naming for DSCIM outputs.

    Ensures file names match original implementation format:
    {recipe}_{discount}_{eta}eta_{rho}rho_{output_type}.{ext}

    Parameters
    ----------
    recipe : str
        Aggregation recipe (e.g., "adding_up", "risk_aversion", "equity")
    discount_type : str
        Discount method (e.g., "constant", "naive_ramsey", "euler_gwr")
    eta : float, optional
        Eta parameter (required for risk_aversion and equity recipes)
    rho : float, optional
        Rho parameter (required for Ramsey/GWR discounting)
    sector : str, optional
        Sector name (e.g., "mortality", "coastal")
    pulse_year : int, optional
        Year of carbon pulse (e.g., 2020)
    ar_version : int, optional
        Assessment report version (5 or 6), default 6
    mask_name : str, optional
        ECS mask name (e.g., "truncate_at_ecs950symmetric"), None for "unmasked"
    fair_aggregation : str, optional
        FAIR dimension aggregation method (e.g., "simulation", "simulation_rcp")

    Examples
    --------
    >>> naming = OutputNaming(
    ...     recipe="risk_aversion",
    ...     discount_type="naive_ramsey",
    ...     eta=2.0,
    ...     rho=0.001,
    ...     sector="mortality",
    ...     pulse_year=2020
    ... )
    >>> naming.get_filename("scc", ext=".nc4")
    'risk_aversion_naive_ramsey_eta2.0_rho0.001_scc.nc4'
    >>>
    >>> path = naming.get_output_path(
    ...     base_dir="results/AR6_ssp",
    ...     output_type="marginal_damages",
    ...     ext=".nc4"
    ... )
    >>> # results/AR6_ssp/mortality/2020/unmasked/risk_aversion_naive_ramsey_eta2.0_rho0.001_marginal_damages.nc4
    """

    def __init__(
        self,
        recipe: str,
        discount_type: str,
        eta: Optional[float] = None,
        rho: Optional[float] = None,
        sector: Optional[str] = None,
        pulse_year: Optional[int] = None,
        ar_version: int = 6,
        mask_name: Optional[str] = None,
        fair_aggregation: Optional[str] = None,
    ):
        self.recipe = recipe
        self.discount_type = discount_type
        self.eta = eta
        self.rho = rho
        self.sector = sector
        self.pulse_year = pulse_year
        self.ar_version = ar_version
        self.mask_name = mask_name
        self.fair_aggregation = fair_aggregation

    def get_base_name(self) -> str:
        """
        Get base filename without output type or extension.

        Returns
        -------
        str
            Base filename like "adding_up_constant_eta2.0_rho0.0001"

        Examples
        --------
        >>> naming = OutputNaming("adding_up", "constant", eta=2.0, rho=0.0001)
        >>> naming.get_base_name()
        'adding_up_constant_eta2.0_rho0.0001'
        """
        parts = [self.recipe, self.discount_type]

        # Add eta if provided
        if self.eta is not None:
            parts.append(f"eta{self.eta}")

        # Add rho if provided
        if self.rho is not None:
            parts.append(f"rho{self.rho}")

        return "_".join(parts)

    def get_filename(
        self,
        output_type: str,
        ext: Optional[str] = None,
        collapsed: bool = True
    ) -> str:
        """
        Get complete filename for output type.

        Parameters
        ----------
        output_type : str
            Type of output (e.g., "scc", "marginal_damages", "discount_factors")
        ext : str, optional
            File extension (e.g., ".nc4", ".csv", ".zarr")
            If None, uses default based on output_type
        collapsed : bool, default True
            If False, adds "uncollapsed_" prefix

        Returns
        -------
        str
            Complete filename

        Examples
        --------
        >>> naming = OutputNaming("adding_up", "constant", eta=2.0, rho=0.001)
        >>> naming.get_filename("scc")
        'adding_up_constant_eta2.0_rho0.001_scc.nc4'
        >>> naming.get_filename("marginal_damages", collapsed=False)
        'adding_up_constant_eta2.0_rho0.001_uncollapsed_marginal_damages.nc4'
        """
        base = self.get_base_name()

        # Add uncollapsed prefix if needed (avoid double-prefixing)
        if not collapsed and not output_type.startswith("uncollapsed_"):
            output_type = f"uncollapsed_{output_type}"

        # Determine extension
        if ext is None:
            ext = self._get_default_extension(output_type)

        return f"{base}_{output_type}{ext}"

    def get_output_path(
        self,
        base_dir: Union[str, Path],
        output_type: str,
        ext: Optional[str] = None,
        collapsed: bool = True
    ) -> Path:
        """
        Get full output path with directory structure.

        The directory structure follows:
        {base_dir}/{sector}/{pulse_year}/{mask_dir}/{fair_agg_dir}/{filename}

        Where:
        - mask_dir is "unmasked" or the mask name
        - fair_agg_dir is optional, based on fair_aggregation

        Parameters
        ----------
        base_dir : str or Path
            Base results directory (e.g., "results/AR6_ssp")
        output_type : str
            Type of output
        ext : str, optional
            File extension
        collapsed : bool, default True
            Whether output is collapsed

        Returns
        -------
        Path
            Full output path

        Examples
        --------
        >>> naming = OutputNaming(
        ...     recipe="adding_up",
        ...     discount_type="constant",
        ...     eta=2.0,
        ...     rho=0.001,
        ...     sector="mortality",
        ...     pulse_year=2020
        ... )
        >>> path = naming.get_output_path("results/AR6_ssp", "scc")
        >>> # results/AR6_ssp/mortality/2020/unmasked/adding_up_constant_eta2.0_rho0.001_scc.nc4
        """
        path = Path(base_dir)

        # Add sector subdirectory
        if self.sector:
            path = path / self.sector

        # Add pulse year subdirectory
        if self.pulse_year:
            path = path / str(self.pulse_year)

        # Add mask subdirectory (unmasked or mask name)
        mask_dir = self.mask_name if self.mask_name else "unmasked"
        path = path / mask_dir

        # Add FAIR aggregation subdirectory if applicable
        if self.fair_aggregation and self.fair_aggregation != "simulation":
            # Format: fair_collapsed_rcp or fair_collapsed_ssp_model
            fair_dims = self.fair_aggregation.replace("simulation_", "").replace("_", "_")
            fair_dir = f"fair_collapsed_{fair_dims}"
            path = path / fair_dir

        # Add filename
        filename = self.get_filename(output_type, ext=ext, collapsed=collapsed)
        return path / filename

    def _get_default_extension(self, output_type: str) -> str:
        """
        Get default file extension for output type.

        Parameters
        ----------
        output_type : str
            Output type

        Returns
        -------
        str
            Default extension
        """
        # CSV files
        if "points" in output_type or "summary" in output_type:
            return ".csv"

        # NetCDF files (most outputs)
        return ".nc4"

    def get_all_output_types(self) -> list:
        """
        Get list of all output types that should be generated.

        Returns
        -------
        list
            List of output type names

        Notes
        -----
        The complete set of outputs per recipe-discount combination includes:
        1. damage_function_coefficients
        2. damage_function_fit
        3. damage_function_points
        4. marginal_damages
        5. uncollapsed_marginal_damages
        6. discount_factors
        7. uncollapsed_discount_factors
        8. global_consumption
        9. global_consumption_no_pulse
        10. scc
        11. uncollapsed_sccs
        """
        outputs = [
            # Damage function outputs
            "damage_function_coefficients",
            "damage_function_fit",
            "damage_function_points",

            # Marginal damages
            "marginal_damages",
            "uncollapsed_marginal_damages",

            # Discount factors
            "discount_factors",
            "uncollapsed_discount_factors",

            # Consumption
            "global_consumption",
            "global_consumption_no_pulse",

            # SCC
            "scc",
            "uncollapsed_sccs",
        ]

        return outputs

    @classmethod
    def from_config(
        cls,
        recipe: str,
        discount_config,
        sector: str,
        pulse_year: int,
        **kwargs
    ) -> "OutputNaming":
        """
        Create OutputNaming from discount configuration.

        Parameters
        ----------
        recipe : str
            Recipe name
        discount_config : DiscountingConfig
            Discount configuration object
        sector : str
            Sector name
        pulse_year : int
            Pulse year
        **kwargs
            Additional arguments passed to OutputNaming

        Returns
        -------
        OutputNaming
            Configured naming object

        Examples
        --------
        >>> from dscim_new.config.schemas import DiscountingConfig
        >>> config = DiscountingConfig(
        ...     discount_type="ramsey",
        ...     eta=2.0,
        ...     rho=0.001
        ... )
        >>> naming = OutputNaming.from_config(
        ...     recipe="adding_up",
        ...     discount_config=config,
        ...     sector="mortality",
        ...     pulse_year=2020
        ... )
        """
        return cls(
            recipe=recipe,
            discount_type=discount_config.discount_type,
            eta=discount_config.eta,
            rho=discount_config.rho,
            sector=sector,
            pulse_year=pulse_year,
            **kwargs
        )

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"OutputNaming(recipe='{self.recipe}', "
            f"discount_type='{self.discount_type}', "
            f"eta={self.eta}, rho={self.rho}, "
            f"sector='{self.sector}', pulse_year={self.pulse_year})"
        )


def parse_filename(filename: str) -> dict:
    """
    Parse DSCIM output filename into components.

    Useful for reading existing outputs and extracting metadata.

    Parameters
    ----------
    filename : str
        Filename to parse

    Returns
    -------
    dict
        Dictionary with parsed components:
        - recipe: str
        - discount_type: str
        - eta: float or None
        - rho: float or None
        - output_type: str
        - collapsed: bool

    Examples
    --------
    >>> info = parse_filename("adding_up_constant_eta2.0_rho0.001_scc.nc4")
    >>> info['recipe']
    'adding_up'
    >>> info['discount_type']
    'constant'
    >>> info['eta']
    2.0
    >>> info['output_type']
    'scc'
    """
    # Remove extension
    name = Path(filename).stem

    # Split by underscore
    parts = name.split("_")

    result = {
        "eta": None,
        "rho": None,
        "collapsed": True,
        "recipe": None,
        "discount_type": None,
        "output_type": None,
    }

    # Check for uncollapsed and remove from parts
    if "uncollapsed" in parts:
        result["collapsed"] = False
        parts.remove("uncollapsed")

    # Known recipes (for better parsing)
    known_recipes = ["adding_up", "risk_aversion", "equity"]

    # Known discount types (order matters - check compound ones first)
    known_discount_types = [
        "euler_gwr", "euler_ramsey", "naive_gwr", "naive_ramsey",
        "constant", "ramsey", "gwr"
    ]

    # Find recipe by checking known recipes
    recipe_parts = []
    i = 0

    # Try to match known recipes first
    for known_recipe in known_recipes:
        recipe_len = len(known_recipe.split("_"))
        if i + recipe_len <= len(parts):
            candidate = "_".join(parts[i:i+recipe_len])
            if candidate == known_recipe:
                result["recipe"] = candidate
                i += recipe_len
                break

    # If no known recipe matched, take first part
    if result["recipe"] is None and i < len(parts):
        result["recipe"] = parts[i]
        i += 1

    # Find discount type by checking known types
    discount_found = False
    for known_discount in known_discount_types:
        discount_len = len(known_discount.split("_"))
        if i + discount_len <= len(parts):
            candidate = "_".join(parts[i:i+discount_len])
            if candidate == known_discount:
                result["discount_type"] = candidate
                i += discount_len
                discount_found = True
                break

    # If no known discount matched but not at eta/rho yet, take next parts
    if not discount_found:
        discount_parts = []
        while i < len(parts) and not parts[i].startswith("eta") and not parts[i].startswith("rho"):
            discount_parts.append(parts[i])
            i += 1
        if discount_parts:
            result["discount_type"] = "_".join(discount_parts)

    # Extract eta and rho
    while i < len(parts):
        if parts[i].startswith("eta"):
            result["eta"] = float(parts[i][3:])
            i += 1
        elif parts[i].startswith("rho"):
            result["rho"] = float(parts[i][3:])
            i += 1
        else:
            # Rest is output type
            result["output_type"] = "_".join(parts[i:])
            break

    return result
