# Pending Implementation

This page documents features and improvements that are planned but not yet implemented in DSCIM-New.

## Testing Infrastructure

### Comparative Testing Suite

**Status**: Implemented but needs cleanup and documentation

**Current state**: A functional testing suite exists that compares DSCIM-New outputs with the original DSCIM implementation using synthetic data. The suite validates computational equivalence across all pipeline steps.

**Location**: Test files in `tests/` directory and validation scripts in `examples/`

**What exists**:
- Synthetic data generation for both implementations
- Comparison of reduced damages outputs
- Validation of damage function coefficients
- SCC value comparisons across recipe-discount combinations
- Results show numerical equivalence within tolerance

**Pending work**:

- Code cleanup - Refactor test organization, remove redundant or outdated test cases, standardize test naming conventions, add clear docstrings to all test functions

- Documentation - Document test suite structure and purpose, add examples of how to run specific comparison tests, explain tolerance thresholds and why they're set, create guide for interpreting test results

- Integration - Add to main documentation, include in CI/CD pipeline, create automated comparison reports, add performance benchmarks alongside correctness checks

- Coverage expansion - Ensure all recipe-discount combinations tested, add edge case tests (missing data, extreme values), test error handling and validation, add regression tests for known issues

**Example of existing test structure**:
```python
def test_reduce_damages_matches_original():
    # Uses same synthetic data for both implementations
    original_result = run_original_dscim_reduce_damages(...)
    new_result = run_dscim_new_reduce_damages(...)

    # Compare outputs within tolerance
    xr.testing.assert_allclose(original_result, new_result, rtol=1e-6)
```

**Priority**: High - Include so other devs can verify correctness. **CURRENTLY WORKING ON CLEANING IT**.

## Computing Resources Management

### Dask Configuration

**Status**: Basic implementation exists, needs refinement

**Current state**:
- `DaskManager` class exists in `pipeline/resources.py`
- Basic context manager for Dask client
- Minimal configuration options

**Pending work**:

- Configuration file management - Expose Dask settings in `ProcessingConfig` schema
- Support for different cluster types (local, distributed, SLURM) - **Investigating how to do the SLURM part**
- Resource limits (memory, workers, threads)
- Adaptive scaling - Dynamic worker scaling based on workload, resource monitoring and adjustment, automatic cleanup on failures
- Cluster profiles - Predefined configurations for common scenarios, HPC cluster integration (SLURM, PBS, SGE)
- Performance monitoring - Dask dashboard integration, resource usage logging, bottleneck identification

## Data Management

- Chunking Strategy - Automatic chunk size determination based on data dimensions, memory-aware chunking for large datasets, configurable chunking strategies per data type

- Caching System - Persistent cache across pipeline runs, cache invalidation strategies, cache size limits and cleanup, shared cache for distributed workers

## Output Management

- Selective Output Saving - Fine-grained control over which outputs to save, output format selection (Zarr, NetCDF, CSV), compression options, metadata standardization

## Pipeline Orchestration

- Workflow Resumption - Checkpointing mechanism, state persistence, automatic detection of completed steps, skip completed steps on resume

- Parallel Recipe/Discount Execution - Parallel execution of independent recipe-discount combinations, resource allocation across parallel tasks, result aggregation from parallel executions

## Validation and Quality Control

- Input Data Validation - Validate data ranges and distributions, check coordinate alignment across datasets, detect missing or corrupted data, provide detailed validation reports

- Output Validation - Sanity checks on outputs (e.g., SCC values within reasonable ranges), comparison with historical runs, statistical summaries for quality control, automatic flagging of anomalous results

## Documentation

- API Reference Completeness - Complete docstrings for all public functions, add usage examples to docstrings, document all configuration options, add type hints throughout codebase

- User Examples - Step-by-step tutorial notebooks, custom configuration examples, advanced usage patterns, troubleshooting guide

## Advanced Features

### Quantile Regression

**Status**: OLS only, quantile regression not implemented

**Reference**: Original DSCIM supports quantile regression for damage functions - **Major implications are on collapsing batch dimension – see labor repo implementation**

**Pending work**:
- Implement quantile regression fitting option
- Support multiple quantiles in single run
- Add configuration for quantile selection

### Sector-Specific Features

- CAMEL sector support (combine mortality and energy)
- Coastal-specific features (GMSL-based damage functions)
- Energy sector temperature interactions

### Climate Scenarios

**Status**: AR6 only

**Pending work**:
- AR5 climate scenario support - **Not urgent, we should migrate to AR6**
- Custom climate scenario inputs
- ECS masking functionality
- Multiple climate model ensembles
