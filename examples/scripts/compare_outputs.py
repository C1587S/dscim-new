"""
Compare outputs between original dscim and refactored dscim-new

This script compares the outputs from:
  - Original: dscim-testing/run_integration_result.py outputs
  - New: reproduce_integration_result.py outputs

The script is structured as sequential code blocks that can be copied into
Jupyter notebook cells and run step-by-step.

Key Comparisons:
1. Reduced Damages: Compare shape, dimensions, and values
2. Damage Functions: Compare coefficients, marginal damages, and fit statistics
3. SCC Results: Compare SCC values across all recipe-discount combinations

Usage:
  - Set the paths to original and new output directories
  - Run each cell sequentially
  - Inspect comparison results and visualizations
"""

import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
import warnings

warnings.filterwarnings("ignore")

# Try to detect if running in Jupyter or as script
try:
    project_root = Path(__file__).resolve().parent.parent.parent
except NameError:
    project_root = Path.cwd().resolve()

print(f"Project root: {project_root}")


# =============================================================================
# CELL 1: Configuration - Set Paths
# =============================================================================

print("\n" + "=" * 80)
print("COMPARISON SETUP")
print("=" * 80)

# Path to original dscim-testing outputs
ORIGINAL_BASE = project_root / "dscim-testing" / "dummy_data"

# Path to new dscim-new outputs
ORIGINAL_BASE = project_root / "dscim-testing" / "dummy_data"

# Path to new dscim-new outputs
NEW_BASE = project_root / "examples" / "notebooks" / "workflow_output"

# Output directory for comparison results
COMPARISON_OUTPUT = project_root / "notebooks"/ "comparison_results"
COMPARISON_OUTPUT.mkdir(parents=True, exist_ok=True)

print(f"\nOriginal outputs: {ORIGINAL_BASE}")
print(f"New outputs: {NEW_BASE}")
print(f"Comparison results: {COMPARISON_OUTPUT}")

# Verify paths exist
if not ORIGINAL_BASE.exists():
    print(f"WARNING: Original path does not exist: {ORIGINAL_BASE}")
if not NEW_BASE.exists():
    print(f"WARNING: New path does not exist: {NEW_BASE}")

# Configure comparison parameters
SECTOR = "not_coastal"  # Sector to compare
ORIGINAL_SECTOR = "dummy_not_coastl_sector"  # Original sector name
PULSE_YEAR = 2020
RECIPES = ["adding_up", "risk_aversion", "equity"]
REDUCTIONS = ["cc", "no_cc"]
DISCOUNT_METHODS = ["constant", "naive_ramsey", "euler_ramsey", "naive_gwr", "euler_gwr"]

print(f"\nComparison parameters:")
print(f"  Sector: {SECTOR} (original: {ORIGINAL_SECTOR})")
print(f"  Pulse year: {PULSE_YEAR}")
print(f"  Recipes: {RECIPES}")
print(f"  Reductions: {REDUCTIONS}")
print(f"  Discount methods: {DISCOUNT_METHODS}")


# =============================================================================
# CELL 2: Helper Functions for Comparison
# =============================================================================

def compare_arrays(
    original: xr.DataArray,
    new: xr.DataArray,
    name: str,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> Dict[str, Any]:
    """
    Compare two xarray DataArrays.

    Returns dictionary with comparison results:
    - shapes_match: bool
    - dims_match: bool
    - values_close: bool
    - max_abs_diff: float
    - max_rel_diff: float
    - correlation: float
    """
    results = {
        "name": name,
        "original_shape": original.shape if hasattr(original, "shape") else None,
        "new_shape": new.shape if hasattr(new, "shape") else None,
        "shapes_match": False,
        "dims_match": False,
        "values_close": False,
        "max_abs_diff": None,
        "max_rel_diff": None,
        "correlation": None,
    }

    # Check shapes
    if hasattr(original, "shape") and hasattr(new, "shape"):
        results["shapes_match"] = original.shape == new.shape

    # Check dimensions
    if hasattr(original, "dims") and hasattr(new, "dims"):
        results["dims_match"] = set(original.dims) == set(new.dims)
        results["original_dims"] = original.dims
        results["new_dims"] = new.dims

    # Compare values if shapes match
    if results["shapes_match"]:
        try:
            # Align coordinates if needed
            orig_aligned, new_aligned = xr.align(original, new, join="inner")

            # Convert to numpy for comparison
            orig_vals = orig_aligned.values
            new_vals = new_aligned.values

            # Remove NaNs for comparison
            mask = ~(np.isnan(orig_vals) | np.isnan(new_vals))
            orig_clean = orig_vals[mask]
            new_clean = new_vals[mask]

            if len(orig_clean) > 0:
                # Check if values are close
                results["values_close"] = np.allclose(orig_clean, new_clean, rtol=rtol, atol=atol)

                # Calculate differences
                abs_diff = np.abs(orig_clean - new_clean)
                results["max_abs_diff"] = float(np.max(abs_diff))
                results["mean_abs_diff"] = float(np.mean(abs_diff))

                # Relative difference (avoid division by zero)
                with np.errstate(divide='ignore', invalid='ignore'):
                    rel_diff = abs_diff / np.abs(orig_clean)
                    rel_diff = rel_diff[np.isfinite(rel_diff)]
                    if len(rel_diff) > 0:
                        results["max_rel_diff"] = float(np.max(rel_diff))
                        results["mean_rel_diff"] = float(np.mean(rel_diff))

                # Correlation
                if len(orig_clean) > 1:
                    results["correlation"] = float(np.corrcoef(orig_clean, new_clean)[0, 1])

        except Exception as e:
            results["error"] = str(e)

    return results


def print_comparison_result(result: Dict[str, Any], verbose: bool = True):
    """Print comparison result in a readable format."""
    print(f"\n  {result['name']}:")
    print(f"    Shapes: {result['original_shape']} vs {result['new_shape']} - {'✓' if result['shapes_match'] else '✗'}")

    if verbose and "original_dims" in result:
        print(f"    Dims: {result['original_dims']} vs {result['new_dims']} - {'✓' if result['dims_match'] else '✗'}")

    if result["values_close"] is not None:
        print(f"    Values close: {'✓' if result['values_close'] else '✗'}")

        if result["max_abs_diff"] is not None:
            print(f"    Max absolute diff: {result['max_abs_diff']:.2e}")
        if result["max_rel_diff"] is not None:
            print(f"    Max relative diff: {result['max_rel_diff']:.2e}")
        if result["correlation"] is not None:
            print(f"    Correlation: {result['correlation']:.6f}")

    if "error" in result:
        print(f"    Error: {result['error']}")


def load_zarr_safe(path: Path) -> xr.Dataset:
    """Load zarr file, handling both formats."""
    try:
        return xr.open_zarr(path)
    except Exception as e:
        print(f"    Error loading {path}: {e}")
        return None


print("\nHelper functions loaded")


# =============================================================================
# CELL 3: Compare Reduced Damages
# =============================================================================

print("\n" + "=" * 80)
print("STEP 1: COMPARE REDUCED DAMAGES")
print("=" * 80)

reduced_damages_comparisons = {}

for recipe in RECIPES:
    for reduction in REDUCTIONS:
        print(f"\nComparing: {recipe} x {reduction}")

        # Build paths for original outputs
        # Original structure: reduced_damages/{sector}/{recipe}_{reduction}.zarr (or with _eta{eta})
        # Handle both adding_up (no eta) and risk_aversion/equity (with eta)
        if recipe == "adding_up":
            original_filename = f"{recipe}_{reduction}.zarr"
        else:
            original_filename = f"{recipe}_{reduction}_eta2.0.zarr"
        original_path = ORIGINAL_BASE / "reduced_damages" / ORIGINAL_SECTOR / original_filename

        # Build paths for new outputs
        # New structure: reduced_damages/{sector}/{recipe}_{reduction}_reduced_damages.zarr (or with _eta{eta})
        if recipe == "adding_up":
            new_filename = f"{recipe}_{reduction}_reduced_damages.zarr"
        else:
            new_filename = f"{recipe}_{reduction}_eta2.0_reduced_damages.zarr"
        new_path = NEW_BASE / "reduced_damages" / SECTOR / new_filename

        print(f"  Original: {original_path}")
        print(f"  New: {new_path}")

        # Check if files exist
        if not original_path.exists():
            print(f"  WARNING: Original file not found")
            continue
        if not new_path.exists():
            print(f"  WARNING: New file not found")
            continue

        # Load data
        original_data = load_zarr_safe(original_path)
        new_data = load_zarr_safe(new_path)

        if original_data is None or new_data is None:
            continue

        # Compare datasets
        print(f"  Original variables: {list(original_data.data_vars)}")
        print(f"  New variables: {list(new_data.data_vars)}")

        # Compare each variable
        common_vars = set(original_data.data_vars) & set(new_data.data_vars)

        if len(common_vars) == 0:
            print(f"  WARNING: No common variables found")
            # Try comparing as single array if datasets have single variable
            if len(original_data.data_vars) == 1 and len(new_data.data_vars) == 1:
                orig_var = list(original_data.data_vars)[0]
                new_var = list(new_data.data_vars)[0]
                print(f"  Comparing {orig_var} vs {new_var}")

                result = compare_arrays(
                    original_data[orig_var],
                    new_data[new_var],
                    f"{recipe}_{reduction}"
                )
                print_comparison_result(result)
                reduced_damages_comparisons[(recipe, reduction)] = result
        else:
            for var in common_vars:
                result = compare_arrays(
                    original_data[var],
                    new_data[var],
                    f"{recipe}_{reduction}_{var}"
                )
                print_comparison_result(result, verbose=False)
                reduced_damages_comparisons[(recipe, reduction, var)] = result

print(f"\nReduced damages comparison complete: {len(reduced_damages_comparisons)} comparisons")


# =============================================================================
# CELL 4: Compare Damage Function Coefficients
# =============================================================================

print("\n" + "=" * 80)
print("STEP 2: COMPARE DAMAGE FUNCTION COEFFICIENTS")
print("=" * 80)

damage_function_comparisons = {}

for recipe in RECIPES:
    print(f"\nComparing damage functions for: {recipe}")

    # Original path structure: results/AR6_ssp/{sector}/{pulse_year}/unmasked/
    # Files named: {recipe}_{discount}_eta{eta}_rho{rho}_damage_function_coefficients.nc4
    # Note: Original has one file per recipe-discount combo

    # New path structure: damage_functions/{sector}/{pulse_year}/
    # Files named: damage_function_coefficients.zarr
    # Note: New has one file per recipe (discount applied later in SCC step)

    original_ar6_dir = ORIGINAL_BASE / "results" / "AR6_ssp" / ORIGINAL_SECTOR / str(PULSE_YEAR) / "unmasked"
    new_dir = NEW_BASE / "damage_functions" / SECTOR / str(PULSE_YEAR)

    print(f"  Original dir: {original_ar6_dir}")
    print(f"  New dir: {new_dir}")

    if not original_ar6_dir.exists():
        print(f"  WARNING: Original directory not found")
        continue
    if not new_dir.exists():
        print(f"  WARNING: New directory not found")
        continue

    # New coefficients path (one per recipe)
    new_coef_path = new_dir / "damage_function_coefficients.zarr"

    if not new_coef_path.exists():
        print(f"  WARNING: New coefficients not found")
        continue

    # Load new coefficients
    new_coefs = load_zarr_safe(new_coef_path)
    if new_coefs is None:
        continue

    print(f"  New coefficients loaded: {new_coefs.dims if hasattr(new_coefs, 'dims') else 'Dataset'}")

    # Find original coefficient files for this recipe
    # Pattern: {recipe}_*_damage_function_coefficients.nc4
    pattern = f"{recipe}_*_damage_function_coefficients.nc4"
    original_coef_files = list(original_ar6_dir.glob(pattern))

    if not original_coef_files:
        print(f"  WARNING: No original coefficient files found (pattern: {pattern})")
        continue

    print(f"  Found {len(original_coef_files)} original coefficient files")

    # Compare with first file (they should all have same coefficients since
    # coefficients are fitted before discount methods are applied)
    original_coef_path = original_coef_files[0]
    print(f"  Comparing with: {original_coef_path.name}")

    original_coefs = xr.open_dataset(original_coef_path)

    # NOTE: Original and new coefficient structures are different:
    # - Original: Multiple data variables (anomaly, np.power(anomaly, 2)) with dims (discount_type, ssp, model, year)
    #             These are fitted coefficient VALUES over time/scenarios
    # - New: Single 'coefficients' variable with dim (coefficient)
    #        These are the REGRESSION coefficients (slope, intercept, etc.)

    print(f"  Original structure: {original_coefs.dims}")
    print(f"  Original data vars: {list(original_coefs.data_vars)}")
    print(f"  New structure: {new_coefs.dims}")
    print(f"  New data vars: {list(new_coefs.data_vars)}")

    # For now, just note the structural difference
    # A proper comparison would need to understand what each represents
    result = {
        "name": f"{recipe}_coefficients",
        "original_shape": str(original_coefs.dims),
        "new_shape": str(new_coefs.dims),
        "shapes_match": False,
        "note": "Different structures: original has fitted values over dimensions, new has regression coefficients",
        "original_vars": list(original_coefs.data_vars),
        "new_vars": list(new_coefs.data_vars),
    }

    print(f"  NOTE: Coefficient structures differ - comparison requires interpretation")
    damage_function_comparisons[recipe] = result

print(f"\nDamage function comparison complete: {len(damage_function_comparisons)} comparisons")


# =============================================================================
# CELL 5: Compare Marginal Damages
# =============================================================================

print("\n" + "=" * 80)
print("STEP 3: COMPARE MARGINAL DAMAGES")
print("=" * 80)

marginal_damages_comparisons = {}

for recipe in RECIPES:
    print(f"\nComparing marginal damages for: {recipe}")

    # New path
    new_dir = NEW_BASE / "damage_functions" / SECTOR / str(PULSE_YEAR)
    new_marg_path = new_dir / "marginal_damages.zarr"

    # Original path (in AR6_ssp results)
    original_ar6_dir = ORIGINAL_BASE / "results" / "AR6_ssp" / ORIGINAL_SECTOR / str(PULSE_YEAR) / "unmasked"

    if not new_marg_path.exists():
        print(f"  WARNING: New marginal damages not found")
        continue

    if not original_ar6_dir.exists():
        print(f"  WARNING: Original directory not found")
        continue

    # Load new marginal damages
    new_marg = load_zarr_safe(new_marg_path)

    if new_marg is None:
        continue

    print(f"  New marginal damages: {new_marg.dims if hasattr(new_marg, 'dims') else 'loaded'}")

    # Find original marginal damages files
    # Pattern: {recipe}_*_marginal_damages.nc4
    pattern = f"{recipe}_*_marginal_damages.nc4"
    original_marg_files = list(original_ar6_dir.glob(pattern))

    if original_marg_files:
        print(f"  Found {len(original_marg_files)} original marginal damage files")
        # Compare with first file (like coefficients, marginal damages should be
        # same across discount methods for a given recipe)
        original_marg_path = original_marg_files[0]
        print(f"  Comparing with: {original_marg_path.name}")

        original_marg = xr.open_dataset(original_marg_path)

        # Extract data for comparison
        orig_data = original_marg.to_array().squeeze() if isinstance(original_marg, xr.Dataset) else original_marg
        new_data = new_marg.to_array().squeeze() if isinstance(new_marg, xr.Dataset) else new_marg

        result = compare_arrays(
            orig_data,
            new_data,
            f"{recipe}_marginal_damages"
        )
        print_comparison_result(result)
        marginal_damages_comparisons[recipe] = result
    else:
        print(f"  WARNING: No original marginal damages found (pattern: {pattern})")

print(f"\nMarginal damages comparison complete: {len(marginal_damages_comparisons)} comparisons")


# =============================================================================
# CELL 6: Compare SCC Results - All Combinations
# =============================================================================

print("\n" + "=" * 80)
print("STEP 4: COMPARE SCC RESULTS")
print("=" * 80)

scc_comparisons = {}

# Map discount method names (new vs original)
discount_name_map = {
    "constant": "constant",
    "naive_ramsey": "naive_ramsey",
    "euler_ramsey": "euler_ramsey",
    "naive_gwr": "naive_gwr",
    "euler_gwr": "euler_gwr",
}

for recipe in RECIPES:
    for discount_method in DISCOUNT_METHODS:
        print(f"\nComparing SCC: {recipe} x {discount_method}")

        # Original path: results/AR6_ssp/{sector}/{pulse_year}/unmasked/
        # File: {recipe}_{discount}_eta{eta}_rho{rho}_scc.nc4
        original_ar6_dir = ORIGINAL_BASE / "results" / "AR6_ssp" / ORIGINAL_SECTOR / str(PULSE_YEAR) / "unmasked"

        # New path: scc_results/{sector}/{pulse_year}/
        # File: {recipe}_{discount}_scc.zarr
        new_scc_dir = NEW_BASE / "scc_results" / SECTOR / str(PULSE_YEAR)

        # Build filenames
        # Original uses eta and rho in filename (e.g., adding_up_constant_eta2.0_rho0.0001_scc.nc4)
        original_pattern = f"{recipe}_{discount_method}_eta*_rho*_scc.nc4"
        new_filename = f"{recipe}_{discount_method}_scc.zarr"

        # Find original file
        if not original_ar6_dir.exists():
            print(f"  WARNING: Original directory not found")
            continue

        original_scc_files = list(original_ar6_dir.glob(original_pattern))

        if not original_scc_files:
            print(f"  WARNING: Original SCC file not found (pattern: {original_pattern})")
            continue

        original_scc_path = original_scc_files[0]
        print(f"  Original: {original_scc_path.name}")

        # Check new file
        new_scc_path = new_scc_dir / new_filename

        if not new_scc_path.exists():
            print(f"  WARNING: New SCC file not found: {new_filename}")
            continue

        print(f"  New: {new_scc_path.name}")

        # Load data
        try:
            original_scc = xr.open_dataset(original_scc_path)
            new_scc = load_zarr_safe(new_scc_path)

            if new_scc is None:
                continue

            # Compare SCC values
            # Extract the main SCC variable
            orig_scc_var = "scc" if "scc" in original_scc else list(original_scc.data_vars)[0]
            new_scc_var = "scc" if "scc" in new_scc else list(new_scc.data_vars)[0]

            result = compare_arrays(
                original_scc[orig_scc_var],
                new_scc[new_scc_var],
                f"{recipe}_{discount_method}_scc"
            )
            print_comparison_result(result)
            scc_comparisons[(recipe, discount_method)] = result

            # Calculate and print mean SCC values
            orig_mean = float(original_scc[orig_scc_var].mean())
            new_mean = float(new_scc[new_scc_var].mean())
            print(f"    Mean SCC: {orig_mean:.2f} (original) vs {new_mean:.2f} (new)")

        except Exception as e:
            print(f"  ERROR: {e}")

print(f"\nSCC comparison complete: {len(scc_comparisons)} comparisons")


# =============================================================================
# CELL 7: Summary Statistics
# =============================================================================

print("\n" + "=" * 80)
print("COMPARISON SUMMARY")
print("=" * 80)

def summarize_comparisons(comparisons: Dict, title: str):
    """Print summary of comparison results."""
    print(f"\n{title}:")
    print(f"  Total comparisons: {len(comparisons)}")

    if len(comparisons) == 0:
        return

    shapes_match = sum(1 for r in comparisons.values() if r.get("shapes_match", False))
    values_close = sum(1 for r in comparisons.values() if r.get("values_close", False))

    print(f"  Shapes match: {shapes_match}/{len(comparisons)} ({100*shapes_match/len(comparisons):.1f}%)")
    print(f"  Values close: {values_close}/{len(comparisons)} ({100*values_close/len(comparisons):.1f}%)")

    # Average differences
    max_abs_diffs = [r["max_abs_diff"] for r in comparisons.values() if r.get("max_abs_diff") is not None]
    if max_abs_diffs:
        print(f"  Average max absolute diff: {np.mean(max_abs_diffs):.2e}")

    max_rel_diffs = [r["max_rel_diff"] for r in comparisons.values() if r.get("max_rel_diff") is not None]
    if max_rel_diffs:
        print(f"  Average max relative diff: {np.mean(max_rel_diffs):.2e}")

    correlations = [r["correlation"] for r in comparisons.values() if r.get("correlation") is not None]
    if correlations:
        print(f"  Average correlation: {np.mean(correlations):.6f}")

summarize_comparisons(reduced_damages_comparisons, "Reduced Damages")
summarize_comparisons(damage_function_comparisons, "Damage Function Coefficients")
summarize_comparisons(marginal_damages_comparisons, "Marginal Damages")
summarize_comparisons(scc_comparisons, "SCC Results")


# =============================================================================
# CELL 8: Save Comparison Report
# =============================================================================

print("\n" + "=" * 80)
print("SAVING COMPARISON REPORT")
print("=" * 80)

# Create detailed report
report_lines = []
report_lines.append("# DSCIM Output Comparison Report\n")
report_lines.append(f"Generated: {pd.Timestamp.now()}\n\n")
report_lines.append(f"Original outputs: {ORIGINAL_BASE}\n")
report_lines.append(f"New outputs: {NEW_BASE}\n\n")

def format_comparison_section(comparisons: Dict, title: str) -> List[str]:
    """Format comparison results as markdown."""
    lines = [f"\n## {title}\n\n"]

    if len(comparisons) == 0:
        lines.append("No comparisons performed.\n\n")
        return lines

    # Create summary table
    lines.append("| Comparison | Shapes Match | Values Close | Max Abs Diff | Correlation |\n")
    lines.append("|------------|--------------|--------------|--------------|-------------|\n")

    for key, result in comparisons.items():
        name = result.get("name", str(key))
        shapes = "✓" if result.get("shapes_match", False) else "✗"
        values = "✓" if result.get("values_close", False) else "✗"
        max_diff = f"{result['max_abs_diff']:.2e}" if result.get("max_abs_diff") is not None else "N/A"
        corr = f"{result['correlation']:.4f}" if result.get("correlation") is not None else "N/A"

        lines.append(f"| {name} | {shapes} | {values} | {max_diff} | {corr} |\n")

    lines.append("\n")
    return lines

report_lines.extend(format_comparison_section(reduced_damages_comparisons, "Reduced Damages"))
report_lines.extend(format_comparison_section(damage_function_comparisons, "Damage Function Coefficients"))
report_lines.extend(format_comparison_section(marginal_damages_comparisons, "Marginal Damages"))
report_lines.extend(format_comparison_section(scc_comparisons, "SCC Results"))

# Save report
report_path = COMPARISON_OUTPUT / "comparison_report.md"
with open(report_path, "w") as f:
    f.writelines(report_lines)

print(f"\nComparison report saved to: {report_path}")

# Also save as CSV for easy analysis
all_comparisons = []
for comp_dict, comp_type in [
    (reduced_damages_comparisons, "reduced_damages"),
    (damage_function_comparisons, "damage_function"),
    (marginal_damages_comparisons, "marginal_damages"),
    (scc_comparisons, "scc"),
]:
    for key, result in comp_dict.items():
        row = {
            "type": comp_type,
            "comparison": result.get("name", str(key)),
            "shapes_match": result.get("shapes_match", None),
            "values_close": result.get("values_close", None),
            "max_abs_diff": result.get("max_abs_diff", None),
            "mean_abs_diff": result.get("mean_abs_diff", None),
            "max_rel_diff": result.get("max_rel_diff", None),
            "correlation": result.get("correlation", None),
        }
        all_comparisons.append(row)

if all_comparisons:
    df = pd.DataFrame(all_comparisons)
    csv_path = COMPARISON_OUTPUT / "comparison_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Comparison data saved to: {csv_path}")

print("\nComparison complete!")
