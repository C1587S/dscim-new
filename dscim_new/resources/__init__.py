"""
Resource management for DSCIM data.

Provides lazy-loading data providers for climate and economic data.
"""

from .data_providers import (
    DataProvider,
    ClimateDataProvider,
    EconomicDataProvider,
)

__all__ = [
    "DataProvider",
    "ClimateDataProvider",
    "EconomicDataProvider",
]
