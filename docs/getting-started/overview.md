# Overview

## Design Goals

DSCIM-New refactors the original implementation to achieve:

- **Modularity**: Separate orchestration, validation, processing, and computation
- **Type Safety**: Pydantic schemas for configuration validation
- **Testability**: Isolated components with clear interfaces
- **Maintainability**: Single-responsibility modules with reduced coupling

## Architecture Layers

The codebase is organized into four distinct layers:

**1. Orchestration** (`pipeline/orchestrator.py`)
- Coordinates workflow execution
- Manages Dask resources
- Iterates over recipe/discount combinations
- Tracks and organizes results

**2. Pipeline Steps** (`pipeline/steps.py`)
- Validate inputs and outputs
- Delegate to processors
- Handle file I/O and naming

**3. Processors** (`preprocessing/`)
- Orchestrate core functions
- Transform data between steps
- No direct I/O operations

**4. Core Functions** (`core/`)
- Pure mathematical operations
- No side effects or state
- Fully unit-testable

## Pipeline Stages

Standard workflow consists of three stages:

1. **Reduce Damages**: Apply climate data to sectoral damage projections
2. **Generate Damage Functions**: Fit OLS/quantile regression models **(NEED TO IMPLEMENT quantile regression)**
3. **Calculate SCC**: Compute discounted marginal damages

Each stage is implemented as a `PipelineStep` subclass.

## Comparison with Original

### Computational Equivalence

Both implementations use:
- Identical damage function specifications
- Same discounting formulas (constant, Ramsey, GWR)
- `statsmodels.formula.api.ols()` for regression
- Same output file formats

Results match within floating-point tolerance.

### Structural Differences

**Original:**
```python
# Scattered across multiple scripts
reduce_damages(sector, config, recipe, ...)
run_ssps(sectors, pulse_years, menu_discs, ...)
```

**DSCIM-New:**
```python
# Unified pipeline interface
pipeline = DSCIMPipeline(config)
pipeline.run_full_pipeline(sectors, recipes, discount_types)
```

Differences:
- Explicit input/output contracts
- Validation via Pydantic
- Unified interface
- Modular components

## Configuration System

Configuration uses Pydantic schemas for type checking:

```python
from dscim_new.config.schemas import DSCIMConfig

config = DSCIMConfig.from_yaml("config.yaml")  # Validates on load
```

Benefits:
- Type errors caught early
- Required fields enforced
- Clear documentation via schema definitions
- IDE autocomplete support

## Extending the Framework

The modular design makes extensions straightforward:

**Add a new recipe:**
1. Implement pure function in `core/`
2. Add processor method if needed
3. Update configuration schema
4. Add to orchestrator iteration

**Add a new discount method:**
1. Implement in `core/discounting.py`
2. Add config option to `DiscountingConfig`
3. Update SCC processor

**Add a new pipeline step:**
1. Subclass `PipelineStep`
2. Define `execute()` method
3. Specify input/output validation
4. Add to orchestrator workflow
