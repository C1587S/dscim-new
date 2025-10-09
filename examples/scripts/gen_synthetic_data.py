"""
Complete run_ssps workflow example.

This script demonstrates the full workflow equivalent to run_ssps from
dscim-testing/run_integration_result.py, including:
1. Generate synthetic climate data
1. Generate synthetic damages data

This is a complete end-to-end example showing all new functionality.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dscim_new.config.schemas import (
    DSCIMConfig,
    ClimateDataConfig,
    DamageFunctionConfig,
    DiscountingConfig,
    SCCConfig,
)
from dscim_new.utils import ClimateDataGenerator, DamagesDataGenerator
from dscim_new.pipeline import (
    ReduceDamagesStep,
    GenerateDamageFunctionStep,
    CalculateSCCStep,
)

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    console = Console()
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    console = None


def print_section(title: str):
    """Print a section header."""
    if RICH_AVAILABLE:
        console.print(f"\n[bold blue]{'='*60}[/bold blue]")
        console.print(f"[bold blue]{title}[/bold blue]")
        console.print(f"[bold blue]{'='*60}[/bold blue]\n")
    else:
        print(f"\n{'='*60}")
        print(title)
        print(f"{'='*60}\n")


def main():
    """Run complete DSCIM workflow: reduce damages -> damage functions -> SCC."""

    if RICH_AVAILABLE:
        console.print(Panel(
            "[bold green]DSCIM Complete Workflow Example[/bold green]\n"
            "Equivalent to run_ssps from dscim-testing",
            style="green"
        ))

    # =========================================================================
    # STEP 1: Generate Synthetic Climate Data
    # =========================================================================
    print_section("STEP 1: Generate Synthetic Climate Data")

    climate_dir = Path("examples/workflow_output/climate_data")
    climate_dir.mkdir(parents=True, exist_ok=True)

    generator = ClimateDataGenerator(seed=42, verbose=True)
    climate_paths = generator.generate_all_climate_data(str(climate_dir))

    print("\n✓ Climate data generated successfully!")
    for key, path in climate_paths.items():
        print(f"  - {key}: {path}")

    # =========================================================================
    # STEP 2: Generate Synthetic Damages Data
    # =========================================================================
    print_section("STEP 2: Generate Synthetic Damages Data")

    damages_dir = Path("examples/workflow_output/damages_data")
    damages_dir.mkdir(parents=True, exist_ok=True)

    damages_generator = DamagesDataGenerator(seed=42, verbose=True)
    damages_paths = damages_generator.generate_all_damages_data(str(damages_dir))

    print("\n✓ Damages data generated successfully!")
    for key, path in damages_paths.items():
        print(f"  - {key}: {path}")


if __name__ == "__main__":
    main()
