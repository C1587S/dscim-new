"""
Preprocessing operations for DSCIM.

Orchestration logic that coordinates I/O and mathematical operations.
"""

from .damages import (
    DamageProcessor,
    reduce_damages,
)

from .damage_functions import (
    DamageFunctionProcessor,
    DamageFunctionResult,
)

from .scc import (
    SCCCalculator,
    SCCResult,
)

__all__ = [
    "DamageProcessor",
    "reduce_damages",
    "DamageFunctionProcessor",
    "DamageFunctionResult",
    "SCCCalculator",
    "SCCResult",
]
