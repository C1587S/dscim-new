# Installation

## Requirements

- Python 3.11 or higher
- conda or pip

## Using conda (recommended)

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

## Using pip

```bash
# Clone repository
git clone https://github.com/C1587S/dscim-new.git
cd dscim-new

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package
pip install -e .
```

## Development Installation

For development with testing and documentation tools:

```bash
pip install -e ".[dev,docs,jupyter]"
```

Optional dependencies:

- `dev`: pytest, black, mypy, pre-commit
- `docs`: sphinx, mkdocs, mkdocstrings
- `jupyter`: jupyterlab, matplotlib, seaborn
- `profiling`: memory-profiler, line-profiler

## Verify Installation

```python
from dscim_new.config.schemas import DSCIMConfig
from dscim_new.pipeline import DSCIMPipeline

print("DSCIM-New installed successfully")
```

## Configuration

DSCIM-New requires configuration files specifying data paths and parameters. See [Configuration Guide](../user-guide/configuration.md) for details.

Example configuration templates are in `examples/configs/`:
- `full_config.yaml` - Complete configuration with all options
- `minimal_config.yaml` - Minimal required fields
