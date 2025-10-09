"""
Core mathematical functions for DSCIM.

This module contains pure mathematical functions with no I/O dependencies.
All functions are deterministic and independently testable.
"""

from .utils import (
    power,
    crra_certainty_equivalent,
    mean_aggregate
)

from .damage_reduction import (
    calculate_no_cc_consumption,
    calculate_cc_consumption,
    apply_bottom_coding,
    aggregate_adding_up,
    aggregate_risk_aversion,
)

from .damage_functions import (
    fit_damage_function_ols,
    evaluate_damage_function,
    calculate_marginal_damages,
    extrapolate_damages,
    compute_damage_function_points,
)

from .discounting import (
    calculate_constant_discount_factors,
    calculate_ramsey_discount_factors,
    calculate_gwr_discount_factors,
    calculate_discount_factors,
    calculate_euler_consumption_path,
    calculate_euler_ramsey_discount_factors,
    calculate_euler_gwr_discount_factors,
    compare_discount_methods,
)

from .scc_calculation import (
    calculate_scc,
    calculate_global_consumption,
    aggregate_scc_over_fair,
    calculate_scc_quantiles,
    calculate_uncollapsed_scc,
    calculate_partial_scc,
)

from .equity import (
    calculate_equity_weights,
    aggregate_equity,
    compare_equity_vs_adding_up,
    validate_equity_weights,
)

__all__ = [
    # Utils
    "power",
    "crra_certainty_equivalent",
    "mean_aggregate",
    # Damage reduction
    "calculate_no_cc_consumption",
    "calculate_cc_consumption",
    "apply_bottom_coding",
    "aggregate_adding_up",
    "aggregate_risk_aversion",
    # Damage functions
    "fit_damage_function_ols",
    "evaluate_damage_function",
    "calculate_marginal_damages",
    "extrapolate_damages",
    "compute_damage_function_points",
    # Discounting
    "calculate_constant_discount_factors",
    "calculate_ramsey_discount_factors",
    "calculate_gwr_discount_factors",
    "calculate_discount_factors",
    "calculate_euler_consumption_path",
    "calculate_euler_ramsey_discount_factors",
    "calculate_euler_gwr_discount_factors",
    "compare_discount_methods",
    # SCC calculation
    "calculate_scc",
    "calculate_global_consumption",
    "aggregate_scc_over_fair",
    "calculate_scc_quantiles",
    "calculate_uncollapsed_scc",
    "calculate_partial_scc",
    # Equity
    "calculate_equity_weights",
    "aggregate_equity",
    "compare_equity_vs_adding_up",
    "validate_equity_weights",
]