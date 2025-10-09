"""
I/O operations for DSCIM.

This module handles all file reading and writing operations,
keeping them separate from core mathematical logic.
"""

from .config import (
    SectorConfig,
    PathsConfig,
    load_config,
    get_sector_config,
)

from .loaders import (
    load_sector_config,
    load_damages_data,
    load_socioeconomic_data,
    get_gdppc_for_coordinates,
)

from .writers import (
    save_reduced_damages,
    construct_output_path,
)

from .climate_io import (
    load_gmst_data,
    load_gmsl_data,
    load_fair_temperature,
    load_fair_gmsl,
    load_pulse_conversion,
    match_climate_to_damages,
    get_climate_variable_for_sector,
)

from .damage_function_io import (
    save_damage_function_coefficients,
    load_damage_function_coefficients,
    save_marginal_damages,
    load_marginal_damages,
    save_damage_function_points,
    load_damage_function_points,
    save_damage_function_summary,
    load_damage_function_summary,
    save_full_damage_function,
)

from .scc_io import (
    save_scc,
    load_scc,
    save_discount_factors,
    load_discount_factors,
    save_scc_quantiles,
    load_scc_quantiles,
    save_scc_summary,
    load_scc_summary,
    save_full_scc,
)

from .naming import (
    OutputNaming,
    parse_filename,
)

__all__ = [
    # Config
    "SectorConfig",
    "PathsConfig",
    "load_config",
    "get_sector_config",
    # Loaders
    "load_sector_config",
    "load_damages_data",
    "load_socioeconomic_data",
    "get_gdppc_for_coordinates",
    # Writers
    "save_reduced_damages",
    "construct_output_path",
    # Climate I/O
    "load_gmst_data",
    "load_gmsl_data",
    "load_fair_temperature",
    "load_fair_gmsl",
    "load_pulse_conversion",
    "match_climate_to_damages",
    "get_climate_variable_for_sector",
    # Damage Function I/O
    "save_damage_function_coefficients",
    "load_damage_function_coefficients",
    "save_marginal_damages",
    "load_marginal_damages",
    "save_damage_function_points",
    "load_damage_function_points",
    "save_damage_function_summary",
    "load_damage_function_summary",
    "save_full_damage_function",
    # SCC I/O
    "save_scc",
    "load_scc",
    "save_discount_factors",
    "load_discount_factors",
    "save_scc_quantiles",
    "load_scc_quantiles",
    "save_scc_summary",
    "load_scc_summary",
    "save_full_scc",
    # Naming
    "OutputNaming",
    "parse_filename",
]