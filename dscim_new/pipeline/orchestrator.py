"""
DSCIM Pipeline orchestrator.

Provides high-level interface for running DSCIM workflows with automatic
validation, resource management, and flexible I/O.
"""

from typing import Union, Dict, Any, Optional, List
from pathlib import Path
from itertools import product
import xarray as xr
import logging

from ..config import DSCIMConfig
from .steps import ReduceDamagesStep, GenerateDamageFunctionStep, CalculateSCCStep
from .resources import DaskManager

logger = logging.getLogger(__name__)


class DSCIMPipeline:
    """
    Main DSCIM pipeline orchestrator.

    Provides validated, flexible execution of DSCIM workflows with
    automatic input validation, resource management, and result tracking.

    Parameters
    ----------
    config : str, dict, or DSCIMConfig
        Configuration (path to YAML, dict, or validated config object)
    dask_config : dict, optional
        Dask configuration (n_workers, memory_limit, etc.)
    verbose : bool, optional
        Whether to print progress messages (default: True)

    Examples
    --------
    >>> # Simple usage
    >>> pipeline = DSCIMPipeline("config.yaml")
    >>> result = pipeline.reduce_damages(
    ...     sector="coastal",
    ...     recipe="adding_up",
    ...     reduction="cc"
    ... )

    >>> # With context manager (auto Dask management)
    >>> with DSCIMPipeline("config.yaml") as pipeline:
    ...     results = pipeline.reduce_all_damages(save=True)

    >>> # Full pipeline (future)
    >>> results = pipeline.run_full_pipeline()
    """

    def __init__(
        self,
        config: Union[str, Dict, DSCIMConfig],
        dask_config: Optional[Dict] = None,
        verbose: bool = True,
    ):
        # Load and validate config
        if isinstance(config, str):
            self.config = DSCIMConfig.from_yaml(config)
        elif isinstance(config, dict):
            self.config = DSCIMConfig.from_dict(config)
        else:
            self.config = config

        self.verbose = verbose
        self.dask_config = dask_config or {}
        self._dask_manager = None
        self._results = {}  # Store all intermediate results

        # Configure logging
        if verbose:
            logging.basicConfig(
                level=logging.INFO,
                format='%(message)s'
            )

    def __enter__(self):
        """Context manager entry - start Dask if configured."""
        if self.config.processing.use_dask:
            self._dask_manager = DaskManager(
                use_dask=True,
                verbose=self.verbose,
                **self.dask_config
            )
            self._dask_manager.start()
        return self

    def __exit__(self, *args):
        """Context manager exit - cleanup Dask."""
        if self._dask_manager:
            self._dask_manager.stop()

    def reduce_damages(
        self,
        sector: str,
        recipe: str,
        reduction: str,
        eta: Optional[float] = None,
        save: Optional[bool] = None,
        output_path: Optional[str] = None,
        **kwargs
    ) -> xr.Dataset:
        """
        Run reduce_damages for a single sector/recipe/reduction combination.

        Parameters
        ----------
        sector : str
            Sector to process
        recipe : str
            Aggregation recipe ("adding_up" or "risk_aversion")
        reduction : str
            Reduction type ("cc" or "no_cc")
        eta : float, optional
            Risk aversion parameter (for risk_aversion recipe)
        save : bool, optional
            Whether to save output. If None, uses config.processing.save_intermediate
        output_path : str, optional
            Custom output path. If None, uses structured path from config
        **kwargs
            Additional parameters passed to ReduceDamagesStep

        Returns
        -------
        xr.Dataset
            Reduced damages data

        Examples
        --------
        >>> # Process in memory
        >>> result = pipeline.reduce_damages(
        ...     sector="coastal",
        ...     recipe="adding_up",
        ...     reduction="cc"
        ... )

        >>> # Save to custom location
        >>> result = pipeline.reduce_damages(
        ...     sector="coastal",
        ...     recipe="adding_up",
        ...     reduction="cc",
        ...     save=True,
        ...     output_path="my_results/coastal.zarr"
        ... )
        """
        # Determine save behavior
        if save is None:
            save = self.config.processing.save_intermediate

        # Create step
        step = ReduceDamagesStep(
            config=self.config,
            sector=sector,
            recipe=recipe,
            reduction=reduction,
            eta=eta,
            verbose=self.verbose,
            **kwargs
        )

        # Prepare inputs
        inputs = {
            "sector_damages_path": self.config.sectors[sector].sector_path,
            "socioec_path": self.config.econdata.global_ssp,
        }

        # Run step
        outputs = step.run(inputs, save=save, output_dir=output_path)

        # Store result
        key = f"reduced_damages_{sector}_{recipe}_{reduction}"
        if eta:
            key += f"_eta{eta}"
        self._results[key] = outputs["reduced_damages"]

        return outputs["reduced_damages"]

    def reduce_all_damages(
        self,
        sectors: Optional[List[str]] = None,
        recipes: Optional[List[str]] = None,
        reductions: Optional[List[str]] = None,
        eta_values: Optional[List[float]] = None,
        save: Optional[bool] = None,
    ) -> Dict[str, xr.Dataset]:
        """
        Run reduce_damages for all specified combinations.

        Parameters
        ----------
        sectors : list, optional
            Sectors to process (default: from config.pipeline or all sectors)
        recipes : list, optional
            Recipes to run (default: from config.pipeline or ["adding_up", "risk_aversion"])
        reductions : list, optional
            Reductions to run (default: from config.pipeline or ["cc", "no_cc"])
        eta_values : list, optional
            Eta values for risk_aversion (default: from config.pipeline or [2.0])
        save : bool, optional
            Whether to save outputs

        Returns
        -------
        dict
            Dictionary of all reduced damages results

        Examples
        --------
        >>> # Process all combinations from config
        >>> results = pipeline.reduce_all_damages(save=True)

        >>> # Process specific combinations
        >>> results = pipeline.reduce_all_damages(
        ...     sectors=["coastal", "agriculture"],
        ...     recipes=["adding_up"],
        ...     reductions=["cc"],
        ...     save=False
        ... )
        """
        # Get defaults from config.pipeline or fallback
        if self.config.pipeline:
            sectors = sectors or self.config.pipeline.sectors_to_process
            recipes = recipes or self.config.pipeline.recipes
            reductions = reductions or self.config.pipeline.reductions
            eta_values = eta_values or self.config.pipeline.eta_values

        # Final defaults
        sectors = sectors or list(self.config.sectors.keys())
        recipes = recipes or ["adding_up", "risk_aversion"]
        reductions = reductions or ["cc", "no_cc"]
        eta_values = eta_values or [2.0]

        results = {}
        total = len(sectors) * len(recipes) * len(reductions)
        completed = 0

        if self.verbose:
            logger.info(f"Processing {total} combinations...")
            logger.info(f"  Sectors: {sectors}")
            logger.info(f"  Recipes: {recipes}")
            logger.info(f"  Reductions: {reductions}")
            if "risk_aversion" in recipes:
                logger.info(f"  Eta values: {eta_values}")
            logger.info("")

        for sector, recipe, reduction in product(sectors, recipes, reductions):
            if recipe == "adding_up":
                eta = None
                result = self.reduce_damages(sector, recipe, reduction, eta, save)
                key = f"{sector}_{recipe}_{reduction}"
                results[key] = result
                completed += 1

                if self.verbose:
                    logger.info(f"[{completed}/{total}] ✓ {key}")

            else:  # risk_aversion
                for eta in eta_values:
                    result = self.reduce_damages(sector, recipe, reduction, eta, save)
                    key = f"{sector}_{recipe}_{reduction}_eta{eta}"
                    results[key] = result
                    completed += 1

                    if self.verbose:
                        logger.info(f"[{completed}/{total}] ✓ {key}")

        if self.verbose:
            logger.info("")
            logger.info(f"✓ Completed {completed} damage reductions")

        return results

    def generate_all_damage_functions(
        self,
        sectors: Optional[List[str]] = None,
        pulse_years: Optional[List[int]] = None,
        save: Optional[bool] = None,
    ) -> Dict[str, Dict[str, xr.Dataset]]:
        """
        Generate damage functions for all sectors and pulse years.

        This step is recipe-agnostic - damage functions are generated from
        reduced damages and can be used with any discounting method.

        Parameters
        ----------
        sectors : list, optional
            Sectors to process (default: from config.pipeline or all sectors)
        pulse_years : list, optional
            Pulse years to process (default: from config.pipeline or [2020])
        save : bool, optional
            Whether to save outputs

        Returns
        -------
        dict
            Nested dictionary: {sector: {pulse_year: results}}

        Examples
        --------
        >>> # Generate for all configured combinations
        >>> damage_functions = pipeline.generate_all_damage_functions(save=True)

        >>> # Generate for specific sectors and years
        >>> damage_functions = pipeline.generate_all_damage_functions(
        ...     sectors=["mortality"],
        ...     pulse_years=[2020, 2050],
        ...     save=True
        ... )
        """
        # Get defaults
        if self.config.pipeline:
            sectors = sectors or self.config.pipeline.sectors_to_process
            pulse_years = pulse_years or self.config.pipeline.pulse_years

        sectors = sectors or list(self.config.sectors.keys())
        pulse_years = pulse_years or [2020]

        if save is None:
            save = self.config.processing.save_intermediate

        results = {}
        total = len(sectors) * len(pulse_years)
        completed = 0

        if self.verbose:
            logger.info(f"Generating {total} damage functions...")
            logger.info(f"  Sectors: {sectors}")
            logger.info(f"  Pulse years: {pulse_years}")
            logger.info("")

        for sector in sectors:
            results[sector] = {}

            for pulse_year in pulse_years:
                # Check if we have reduced damages
                # For damage functions, we typically need the baseline reduced damages
                # which would be from adding_up or risk_aversion recipe

                # Create step
                step = GenerateDamageFunctionStep(
                    config=self.config,
                    sector=sector,
                    pulse_year=pulse_year,
                    verbose=self.verbose,
                )

                # Get inputs - check if we have them in memory
                reduced_damages_key = f"reduced_damages_{sector}_adding_up_cc"
                if reduced_damages_key in self._results:
                    inputs = {"reduced_damages": self._results[reduced_damages_key]}
                else:
                    # Load from disk or skip
                    logger.warning(
                        f"No reduced damages found for {sector}. "
                        "Run reduce_all_damages first."
                    )
                    continue

                # Run step
                outputs = step.run(inputs, save=save)

                # Store results
                results[sector][pulse_year] = outputs

                # Store in memory for later steps
                for key, value in outputs.items():
                    result_key = f"{key}_{sector}_{pulse_year}"
                    self._results[result_key] = value

                completed += 1
                if self.verbose:
                    logger.info(f"[{completed}/{total}] ✓ {sector} - {pulse_year}")

        if self.verbose:
            logger.info("")
            logger.info(f"✓ Generated {completed} damage functions")

        return results

    def calculate_all_sccs(
        self,
        sectors: Optional[List[str]] = None,
        pulse_years: Optional[List[int]] = None,
        recipes: Optional[List[str]] = None,
        save: Optional[bool] = None,
    ) -> Dict[str, Dict]:
        """
        Calculate SCC for all recipe-discount combinations.

        Iterates over all combinations of recipes and discount methods,
        generating all 11 output types per combination.

        Parameters
        ----------
        sectors : list, optional
            Sectors to process
        pulse_years : list, optional
            Pulse years to process
        recipes : list, optional
            Recipes to use (default: ["adding_up", "risk_aversion", "equity"])
        save : bool, optional
            Whether to save outputs

        Returns
        -------
        dict
            Results organized by sector/pulse_year/recipe/discount

        Examples
        --------
        >>> # Calculate SCC for all combinations
        >>> sccs = pipeline.calculate_all_sccs(save=True)

        >>> # This generates outputs for all 15 combinations:
        >>> # 3 recipes × 5 discount methods
        >>> # Each combination produces 11 output types
        """
        # Get defaults
        if self.config.pipeline:
            sectors = sectors or self.config.pipeline.sectors_to_process
            pulse_years = pulse_years or self.config.pipeline.pulse_years
            recipes = recipes or self.config.pipeline.recipes

        sectors = sectors or list(self.config.sectors.keys())
        pulse_years = pulse_years or [2020]
        recipes = recipes or ["adding_up", "risk_aversion", "equity"]

        if save is None:
            save = self.config.processing.save_intermediate

        # Get discount configurations
        discount_configs = self.config.get_discounting_configs()

        # Calculate total combinations
        total = len(sectors) * len(pulse_years) * len(recipes) * len(discount_configs)
        completed = 0

        if self.verbose:
            logger.info(f"Calculating {total} SCC combinations...")
            logger.info(f"  Sectors: {sectors}")
            logger.info(f"  Pulse years: {pulse_years}")
            logger.info(f"  Recipes: {recipes}")
            logger.info(f"  Discount methods: {len(discount_configs)}")
            logger.info("")

        results = {}

        for sector in sectors:
            results[sector] = {}

            for pulse_year in pulse_years:
                results[sector][pulse_year] = {}

                # Check for marginal damages
                md_key = f"marginal_damages_{sector}_{pulse_year}"
                if md_key not in self._results:
                    logger.warning(
                        f"No marginal damages found for {sector}/{pulse_year}. "
                        "Run generate_all_damage_functions first."
                    )
                    continue

                marginal_damages = self._results[md_key]

                for recipe in recipes:
                    results[sector][pulse_year][recipe] = {}

                    for discount_idx, discount_config in enumerate(discount_configs):
                        # Create step
                        step = CalculateSCCStep(
                            config=self.config,
                            sector=sector,
                            pulse_year=pulse_year,
                            recipe=recipe,
                            discount_config_index=discount_idx,
                            verbose=self.verbose,
                        )

                        # Prepare inputs
                        inputs = {"marginal_damages": marginal_damages}

                        # Add consumption if needed for Ramsey/GWR
                        if discount_config.discount_type in ["ramsey", "gwr"]:
                            consumption_key = "consumption_data"
                            if consumption_key in self._results:
                                inputs["consumption"] = self._results[consumption_key]

                        # Run step
                        try:
                            outputs = step.run(inputs, save=save)

                            # Store results
                            discount_name = discount_config.discount_type
                            if discount_config.eta:
                                discount_name += f"_eta{discount_config.eta}"
                            if discount_config.rho:
                                discount_name += f"_rho{discount_config.rho}"

                            results[sector][pulse_year][recipe][discount_name] = outputs

                            # Store key outputs in memory
                            for key, value in outputs.items():
                                result_key = f"{key}_{sector}_{pulse_year}_{recipe}_{discount_name}"
                                self._results[result_key] = value

                            completed += 1

                            if self.verbose:
                                combo_desc = f"{sector}/{pulse_year}/{recipe}/{discount_name}"
                                logger.info(f"[{completed}/{total}] ✓ {combo_desc}")

                        except Exception as e:
                            logger.error(
                                f"Failed to calculate SCC for "
                                f"{sector}/{pulse_year}/{recipe}/{discount_config.discount_type}: {e}"
                            )
                            continue

        if self.verbose:
            logger.info("")
            logger.info(f"✓ Calculated {completed} SCC combinations")

        return results

    def run_full_pipeline(self, save: Optional[bool] = None):
        """
        Run complete DSCIM pipeline for all recipe-discount combinations.

        Executes the full workflow:
        1. Reduce damages for all sectors
        2. Generate damage functions for all pulse years
        3. Calculate SCC for all recipe-discount combinations

        This will generate outputs for all 15 recipe-discount combinations
        (3 recipes × 5 discount methods), with 11 output types each.

        Parameters
        ----------
        save : bool, optional
            Whether to save intermediate results

        Returns
        -------
        dict
            Dictionary with results from all pipeline steps

        Examples
        --------
        >>> # Run full pipeline with auto-save
        >>> with DSCIMPipeline("config.yaml") as pipeline:
        ...     results = pipeline.run_full_pipeline(save=True)
        """
        if save is None:
            save = self.config.processing.save_intermediate

        if self.verbose:
            logger.info("=" * 80)
            logger.info("RUNNING FULL DSCIM PIPELINE")
            logger.info("=" * 80)
            logger.info("")

        # Step 1: Reduce damages
        if self.verbose:
            logger.info("Step 1: Reduce Damages")
            logger.info("-" * 80)
            logger.info("")

        reduced_damages = self.reduce_all_damages(save=save)

        if self.verbose:
            logger.info("")

        # Step 2: Generate damage functions
        if self.verbose:
            logger.info("Step 2: Generate Damage Functions")
            logger.info("-" * 80)
            logger.info("")

        damage_functions = self.generate_all_damage_functions(save=save)

        if self.verbose:
            logger.info("")

        # Step 3: Calculate SCC for all recipe-discount combinations
        if self.verbose:
            logger.info("Step 3: Calculate SCC (All Recipe-Discount Combinations)")
            logger.info("-" * 80)
            logger.info("")

        sccs = self.calculate_all_sccs(save=save)

        if self.verbose:
            logger.info("")
            logger.info("=" * 80)
            logger.info("PIPELINE COMPLETE")
            logger.info("=" * 80)
            logger.info("")
            logger.info(f"Generated outputs for:")
            logger.info(f"  • {len(reduced_damages)} damage reductions")
            logger.info(f"  • {sum(len(v) for v in damage_functions.values())} damage functions")

            # Count SCC combinations
            total_sccs = 0
            for sector_data in sccs.values():
                for year_data in sector_data.values():
                    for recipe_data in year_data.values():
                        total_sccs += len(recipe_data)

            logger.info(f"  • {total_sccs} SCC combinations")

        return {
            "reduced_damages": reduced_damages,
            "damage_functions": damage_functions,
            "sccs": sccs,
        }

    def get_results(self) -> Dict[str, Any]:
        """
        Get all stored results from pipeline execution.

        Returns
        -------
        dict
            Dictionary of all results
        """
        return self._results

    def save_result(
        self,
        key: str,
        data: xr.Dataset,
        output_path: str,
        output_format: Optional[str] = None
    ):
        """
        Save a specific result to custom location.

        Parameters
        ----------
        key : str
            Result key
        data : xr.Dataset
            Data to save
        output_path : str
            Output file path
        output_format : str, optional
            Output format (default: from config)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fmt = output_format or self.config.processing.output_format

        if fmt == "zarr":
            data.to_zarr(output_path, consolidated=True, mode="w")
        elif fmt == "netcdf":
            data.to_netcdf(output_path)
        elif fmt == "csv":
            data.to_dataframe().reset_index().to_csv(output_path, index=False)

        if self.verbose:
            logger.info(f"Saved {key} to {output_path}")

    def visualize(self, format: str = "mermaid") -> str:
        """
        Generate pipeline visualization diagram.

        Parameters
        ----------
        format : str, optional
            Diagram format: "mermaid" (default) or "graphviz"

        Returns
        -------
        str
            Diagram markup that can be rendered

        Examples
        --------
        >>> pipeline = DSCIMPipeline("config.yaml")
        >>> diagram = pipeline.visualize()
        >>> print(diagram)

        >>> # In Jupyter notebook
        >>> from IPython.display import display, Markdown
        >>> display(Markdown(f"```mermaid\\n{diagram}\\n```"))
        """
        if format == "mermaid":
            return self._generate_mermaid()
        elif format == "graphviz":
            return self._generate_graphviz()
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'mermaid' or 'graphviz'")

    def _generate_mermaid(self) -> str:
        """Generate Mermaid diagram for pipeline."""
        # Get configuration info
        sectors = self.config.get_pipeline_sectors()
        recipes = self.config.pipeline.recipes if self.config.pipeline else ["adding_up", "risk_aversion"]
        reductions = self.config.pipeline.reductions if self.config.pipeline else ["cc", "no_cc"]

        diagram = ["graph LR"]
        diagram.append("    A[Configuration File] -->|validate| B[Load Economic Data]")
        diagram.append("    A -->|validate| C[Load Sector Data]")
        diagram.append("")

        # Sector nodes
        for i, sector in enumerate(sectors, 1):
            sector_node = f"S{i}[{sector}]"
            diagram.append(f"    C --> {sector_node}")

        diagram.append("")

        # Processing node
        diagram.append("    B --> D[Damage Processor]")
        for i in range(1, len(sectors) + 1):
            diagram.append(f"    S{i} --> D")

        diagram.append("")

        # Recipe branches
        if "adding_up" in recipes:
            diagram.append("    D -->|adding_up| E1[Mean Aggregation]")
        if "risk_aversion" in recipes:
            diagram.append("    D -->|risk_aversion| E2[CRRA Utility]")

        diagram.append("")

        # Output node
        diagram.append("    E1 --> F[Output Dataset]")
        if "risk_aversion" in recipes:
            diagram.append("    E2 --> F")

        diagram.append("")

        # Saving
        diagram.append("    F -->|optional| G[Save to Disk]")

        diagram.append("")

        # Styling
        diagram.append("    style A fill:#e1f5ff")
        diagram.append("    style D fill:#fff4e1")
        diagram.append("    style F fill:#e7ffe1")
        diagram.append("    style G fill:#ffe1e1")

        return "\n".join(diagram)

    def _generate_graphviz(self) -> str:
        """Generate Graphviz DOT diagram for pipeline."""
        sectors = self.config.get_pipeline_sectors()
        recipes = self.config.pipeline.recipes if self.config.pipeline else ["adding_up", "risk_aversion"]

        dot = ['digraph DSCIM {']
        dot.append('    rankdir=LR;')
        dot.append('    node [shape=box, style=rounded];')
        dot.append('')

        # Nodes
        dot.append('    config [label="Configuration", fillcolor="#e1f5ff", style="filled,rounded"];')
        dot.append('    econ [label="Economic Data", fillcolor="#e1f5ff", style="filled,rounded"];')
        dot.append('    sector_data [label="Sector Data", fillcolor="#e1f5ff", style="filled,rounded"];')
        dot.append('    processor [label="Damage Processor", fillcolor="#fff4e1", style="filled,rounded"];')
        dot.append('    output [label="Output Dataset", fillcolor="#e7ffe1", style="filled,rounded"];')
        dot.append('    save [label="Save to Disk", fillcolor="#ffe1e1", style="filled,rounded"];')
        dot.append('')

        # Edges
        dot.append('    config -> econ [label="validate"];')
        dot.append('    config -> sector_data [label="validate"];')
        dot.append('    econ -> processor;')
        dot.append('    sector_data -> processor;')

        for recipe in recipes:
            dot.append(f'    processor -> output [label="{recipe}"];')

        dot.append('    output -> save [label="optional", style=dashed];')
        dot.append('}')

        return "\n".join(dot)
