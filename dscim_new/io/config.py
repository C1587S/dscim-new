"""
Configuration handling for DSCIM.

Loads and validates configuration from YAML files.
"""

import yaml
from dataclasses import dataclass
from typing import Dict, Any
from pathlib import Path


@dataclass
class SectorConfig:
    """Configuration for a specific sector."""
    sector_path: str
    histclim: str
    delta: str
    formula: str


@dataclass
class PathsConfig:
    """Paths configuration."""
    reduced_damages_library: str
    ssp_damage_function_library: str
    AR6_ssp_results: str
    AR5_ssp_results: str = None


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Parameters
    ----------
    config_path : str
        Path to YAML configuration file

    Returns
    -------
    dict
        Parsed configuration dictionary

    Examples
    --------
    >>> config = load_config("configs/dummy_config.yaml")
    >>> config["sectors"]
    {...}
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_sector_config(config: Dict[str, Any], sector: str) -> SectorConfig:
    """
    Extract sector-specific configuration from main config.

    Parameters
    ----------
    config : dict
        Full configuration dictionary
    sector : str
        Sector name

    Returns
    -------
    SectorConfig
        Sector configuration object

    Raises
    ------
    KeyError
        If sector not found in configuration

    Examples
    --------
    >>> config = load_config("configs/dummy_config.yaml")
    >>> sector_config = get_sector_config(config, "dummy_coastal_sector")
    >>> sector_config.sector_path
    './dummy_data/sectoral/coastal_damages.zarr'
    """
    if sector not in config["sectors"]:
        raise KeyError(f"Sector '{sector}' not found in configuration")

    sector_data = config["sectors"][sector]

    return SectorConfig(
        sector_path=sector_data["sector_path"],
        histclim=sector_data["histclim"],
        delta=sector_data["delta"],
        formula=sector_data["formula"]
    )


def get_paths_config(config: Dict[str, Any]) -> PathsConfig:
    """
    Extract paths configuration from main config.

    Parameters
    ----------
    config : dict
        Full configuration dictionary

    Returns
    -------
    PathsConfig
        Paths configuration object

    Examples
    --------
    >>> config = load_config("configs/dummy_config.yaml")
    >>> paths = get_paths_config(config)
    >>> paths.reduced_damages_library
    './dummy_data/reduced_damages'
    """
    paths_data = config["paths"]

    return PathsConfig(
        reduced_damages_library=paths_data["reduced_damages_library"],
        ssp_damage_function_library=paths_data["ssp_damage_function_library"],
        AR6_ssp_results=paths_data["AR6_ssp_results"],
        AR5_ssp_results=paths_data.get("AR5_ssp_results")
    )