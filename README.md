# DSCIM-New

Refactored implementation of the Data-driven Spatial Climate Impact Model with modular architecture.

[![Documentation Status](https://readthedocs.org/projects/dscim-new/badge/?version=latest)](https://dscim-new.readthedocs.io/en/latest/)

## Overview

DSCIM-New is a refactored version of the Data-driven Spatial Climate Impact Model, designed with:

- **Modular pipeline architecture** - Independent, reusable pipeline steps
- **Type-safe configuration** - Pydantic schemas for validation
- **Computational equivalence** - Produces identical results to original DSCIM
- **Enhanced flexibility** - Run complete workflows or individual steps

## Documentation

**Full documentation**: https://dscim-new.readthedocs.io

Key sections:
- [Getting Started](https://dscim-new.readthedocs.io/en/latest/getting-started/overview/) - Installation and quick start
- [User Guide](https://dscim-new.readthedocs.io/en/latest/user-guide/architecture/) - Architecture and configuration
- [Examples](https://dscim-new.readthedocs.io/en/latest/examples/complete-workflow/) - Complete workflow demonstrations
- [API Reference](https://dscim-new.readthedocs.io/en/latest/api/pipeline/) - Detailed API documentation
- [Developer Guide](https://dscim-new.readthedocs.io/en/latest/developer/comparison/) - Comparison with original DSCIM

## Quick Install

```bash
git clone https://github.com/C1587S/dscim-new.git
cd dscim-new
conda env create -f environment.yml
conda activate dscim-new
pip install -e .
```

See the [Installation Guide](https://dscim-new.readthedocs.io/en/latest/getting-started/installation/) for detailed instructions.

## Basic Usage

```python
from dscim_new.pipeline import DSCIMPipeline

pipeline = DSCIMPipeline("config.yaml")
results = pipeline.run_full_pipeline(
    sectors=["mortality"],
    recipes=["adding_up"],
    discount_types=["constant"],
    save=True
)
```

See [Complete Workflow Example](https://dscim-new.readthedocs.io/en/latest/examples/complete-workflow/) for more details.

## Project Status

**Version**: 0.1.0 

This is a refactored implementation currently under active development. See [Project Status](https://dscim-new.readthedocs.io/en/latest/about/status/) and [Pending Implementation](https://dscim-new.readthedocs.io/en/latest/developer/pending/) for details.

## Links

- **Documentation**: https://dscim-new.readthedocs.io
- **Source Code**: https://github.com/C1587S/dscim-new/tree/master/dscim_new
- **Issues**: https://github.com/C1587S/dscim-new/issues
