# Architecture

## Design Principles

DSCIM-New follows a layered architecture with clear separation of concerns:

- **Orchestration**: Workflow coordination and resource management
- **Validation**: Input/output verification and type checking
- **Processing**: Data transformation orchestration
- **Computation**: Pure mathematical functions

## Layer Structure

```
┌─────────────────────────────────────────┐
│         DSCIMPipeline                   │
│      (pipeline/orchestrator.py)         │
│  - Context manager (Dask)               │
│  - High-level API                       │
│  - Result tracking                      │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│          PipelineStep                   │
│         (pipeline/base.py)              │
│  - Input/output validation              │
│  - Abstract execute() method            │
│  - Optional file saving                 │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│        Concrete Steps                   │
│       (pipeline/steps.py)               │
│  - ReduceDamagesStep                    │
│  - GenerateDamageFunctionStep           │
│  - CalculateSCCStep                     │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│         Processors                      │
│      (preprocessing/)                   │
│  - DamageProcessor                      │
│  - DamageFunctionProcessor              │
│  - SCCCalculator                        │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│       Pure Functions                    │
│          (core/)                        │
│  - damage_reduction.py                  │
│  - damage_functions.py                  │
│  - discounting.py                       │
│  - scc_calculation.py                   │
└─────────────────────────────────────────┘
```

## Module Organization

### `pipeline/`

**orchestrator.py**: Coordinates workflow execution
- `DSCIMPipeline`: Main interface for running complete workflows
- Manages Dask resources via context manager
- Iterates over recipe/discount combinations
- Tracks results in organized dictionaries

**base.py**: Abstract base class for steps
- `PipelineStep`: Defines interface for all steps
- `validate_inputs()`: Check required inputs present
- `execute()`: Abstract method for step logic
- `run()`: Execute with optional saving

**steps.py**: Concrete step implementations
- `ReduceDamagesStep`: Apply climate data to damages
- `GenerateDamageFunctionStep`: Fit regression models
- `CalculateSCCStep`: Compute discounted SCC

### `core/`

Pure mathematical functions with no side effects:

- **damage_reduction.py**: Climate damage calculations
- **damage_functions.py**: OLS/quantile regression fitting
- **discounting.py**: Constant, Ramsey, and GWR discount factors
- **scc_calculation.py**: Marginal damage integration
- **equity.py**: Equity-weighted aggregation
- **utils.py**: Shared mathematical utilities

### `preprocessing/`

Data transformation orchestrators:

- **damages.py**: `DamageProcessor` - coordinates damage reduction
- **damage_functions.py**: `DamageFunctionProcessor` - handles regression fitting
- **scc.py**: `SCCCalculator` - orchestrates SCC computation

Processors call core functions and manage data flow between steps.

### `config/`

Configuration schemas and validation:

- **schemas.py**: Pydantic models for all configuration
  - `DSCIMConfig`: Top-level configuration
  - `DamageFunctionConfig`: Regression specifications
  - `DiscountingConfig`: Discount method parameters
  - `SectorConfig`: Sector-specific settings

### `io/`

Data loading and saving utilities:

- **loaders.py**: Read NetCDF, Zarr, CSV files
- **writers.py**: Save results in various formats
- **climate_io.py**: Climate data loading
- **scc_io.py**: SCC result saving
- **naming.py**: Standardized file naming conventions

### `resources/`

Data provider abstractions:

- **data_providers.py**: Lazy-loading data providers
  - `ClimateDataProvider`: Climate data management
  - `EconomicDataProvider`: Economic data management

## Data Flow

### Complete Pipeline

```
Input Data
    │
    ├─→ Climate (GMST, GMSL, FAIR)
    ├─→ Damages (sectoral projections)
    └─→ Economic (GDP, population, consumption)
    │
    ▼
ReduceDamagesStep
    │
    ├─→ Load climate matching data
    ├─→ Apply to damage projections
    └─→ Output: reduced_damages.zarr
    │
    ▼
GenerateDamageFunctionStep
    │
    ├─→ Fit OLS regression
    ├─→ Calculate marginal damages
    └─→ Output: coefficients.zarr, marginal_damages.zarr
    │
    ▼
CalculateSCCStep
    │
    ├─→ Calculate discount factors
    ├─→ Integrate marginal damages
    └─→ Output: scc.zarr, discount_factors.zarr
    │
    ▼
Results
```

### Step Execution

Each `PipelineStep` follows this pattern:

```python
class MyStep(PipelineStep):
    def validate_inputs(self, inputs: Dict[str, Any]):
        # Check required inputs present
        required = ["input_data"]
        missing = [k for k in required if k not in inputs]
        if missing:
            raise ValueError(f"Missing inputs: {missing}")

    def execute(self, **inputs) -> Dict[str, Any]:
        # 1. Extract inputs
        data = inputs["input_data"]

        # 2. Call processor
        processor = MyProcessor(self.config)
        result = processor.process(data)

        # 3. Return outputs
        return {"output_data": result}
```

## Configuration Flow

Configuration is loaded once and passed to all components:

```python
# Load and validate configuration
config = DSCIMConfig.from_yaml("config.yaml")

# Pass to pipeline
pipeline = DSCIMPipeline(config)

# Pipeline passes to steps
step = ReduceDamagesStep(config=config, ...)

# Steps pass to processors
processor = DamageProcessor(config.processing)

# Processors access specific settings
gmst_path = config.climate_data.gmst_path
```

## Extension Points

### Adding a New Recipe

1. Implement core function in `core/`:
```python
def aggregate_new_recipe(damages, params):
    # Pure function implementation
    return aggregated_damages
```

2. Add to processor in `preprocessing/damages.py`:
```python
def reduce_damages(self, recipe: str, ...):
    if recipe == "new_recipe":
        return aggregate_new_recipe(damages, params)
```

3. Update configuration schema in `config/schemas.py`:
```python
class PipelineConfig(BaseModel):
    recipes: List[Literal["adding_up", "risk_aversion", "equity", "new_recipe"]]
```

### Adding a New Discount Method

1. Implement in `core/discounting.py`:
```python
def calculate_new_discount(consumption, eta, rho):
    # Pure function implementation
    return discount_factors
```

2. Update `SCCCalculator` in `preprocessing/scc.py`:
```python
def calculate_discount_factors(self, method: str, ...):
    if method == "new_method":
        return calculate_new_discount(...)
```

3. Add configuration in `config/schemas.py`:
```python
class DiscountingConfig(BaseModel):
    discount_type: Literal["constant", "ramsey", "gwr", "new_method"]
```

### Adding a New Pipeline Step

1. Subclass `PipelineStep`:
```python
class MyNewStep(PipelineStep):
    def validate_inputs(self, inputs):
        # Validation logic
        pass

    def execute(self, **inputs):
        # Step logic
        return outputs
```

2. Add to orchestrator workflow:
```python
def run_full_pipeline(self):
    # Existing steps...
    new_result = MyNewStep(config=self.config).run(inputs, save=True)
```

## Testing Strategy

The layered architecture enables isolated testing:

### Unit Tests (Core Functions)
```python
def test_aggregate_damages():
    damages = create_test_damages()
    result = aggregate_damages(damages, method="adding_up")
    assert result.shape == expected_shape
```

### Integration Tests (Steps)
```python
def test_reduce_damages_step():
    config = create_test_config()
    step = ReduceDamagesStep(config, sector="test")
    result = step.run(inputs=test_inputs)
    assert "reduced_damages" in result
```

### End-to-End Tests (Pipeline)
```python
def test_full_pipeline():
    pipeline = DSCIMPipeline("test_config.yaml")
    results = pipeline.run_full_pipeline(save=False)
    assert results["sccs"]["sector"]["recipe"]["discount"] is not None
```

## Resource Management

DSCIM-New uses context managers for resource cleanup:

```python
with DSCIMPipeline(config) as pipeline:
    results = pipeline.run_full_pipeline()
# Dask cluster automatically cleaned up
```

Dask configuration is managed through `DaskManager` in `pipeline/resources.py`.
