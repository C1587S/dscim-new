# DSCIM-New

Refactored implementation of the Data-driven Spatial Climate Impact Model with modular architecture.

## Features

- Modular pipeline architecture with distinct steps
- Type-safe configuration via Pydantic
- Pure functional core separated from I/O
- Support for constant, Ramsey, and GWR discounting
- Adding-up, risk aversion, and equity aggregation
- Dask-based distributed computation

## Quick Start

```python
from dscim_new.pipeline import DSCIMPipeline

# Load configuration
pipeline = DSCIMPipeline("config.yaml")

# Run complete pipeline
results = pipeline.run_full_pipeline(
    sectors=["mortality"],
    recipes=["adding_up", "risk_aversion"],
    discount_types=["constant", "ramsey"],
    save=True
)

# Access SCC results
scc = results["sccs"]["mortality"]["adding_up"]["constant"]["scc"]
```

## Documentation Structure

- **[Getting Started](getting-started/overview.md)** - Installation and quick start
- **[User Guide](user-guide/architecture.md)** - Architecture and configuration
- **[Examples](examples/complete-workflow.md)** - Complete workflow demonstrations
- **[API Reference](api/pipeline.md)** - Detailed API documentation
- **[Developer Guide](developer/comparison.md)** - Comparison with original DSCIM

## Structure

```
dscim_new/
├── pipeline/       # Orchestration and step definitions
├── core/           # Pure mathematical functions
├── preprocessing/  # Data transformation processors
├── config/         # Pydantic schemas and validation
├── io/             # Data loading and saving
└── resources/      # Data providers
```

## Key Difference from Original DSCIM

DSCIM-New refactors the workflow into modular, independent steps that provide flexibility:

- **Run complete pipeline** - Automated workflow for standard analyses
- **Run individual steps** - Fine-grained control for development and debugging
- **Reuse intermediate results** - Skip expensive recomputation
- **Modify between steps** - Insert custom processing

Both implementations compute identical results. The difference is how users interact with the workflow.

See [Comparison with Original DSCIM](developer/comparison.md) for details on modularity and workflow flexibility.

## Installation

```bash
# Clone repository
git clone https://github.com/C1587S/dscim-new.git
cd dscim-new

# Create environment
conda env create -f environment.yml
conda activate dscim-new

# Install package
pip install -e .
```

See [Installation Guide](getting-started/installation.md) for detailed instructions.

## Project Status

**Version**: 0.1.0 (Alpha)

Implemented features:
- Core pipeline architecture
- All aggregation recipes (adding_up, risk_aversion, equity)
- All discounting methods (constant, Ramsey, GWR)
- Comprehensive test suite
- Full documentation

See [Project Status](about/status.md) and [Pending Implementation](developer/pending.md) for planned features.

## Repository

- **Documentation**: https://dscim-new.readthedocs.io
- **GitHub**: https://github.com/C1587S/dscim-new/tree/master/dscim_new
- **Issues**: https://github.com/C1587S/dscim-new/issues
