# OWI-metadatabase Soil Extension

[![version](https://img.shields.io/pypi/v/owi-metadatabase-soil)](https://test.pypi.org/project/owi-metadatabase-soil/)
[![python versions](https://img.shields.io/pypi/pyversions/owi-metadatabase-soil)](https://test.pypi.org/project/owi-metadatabase-soil/)
[![license](https://img.shields.io/github/license/owi-lab/owi-metadatabase-soil-sdk)](https://github.com/OWI-Lab/owi-metadatabase-soil-sdk/blob/main/LICENSE)
[![pytest](https://img.shields.io/github/actions/workflow/status/owi-lab/owi-metadatabase-soil-sdk/ci.yml?label=pytest)](https://github.com/OWI-Lab/owi-metadatabase-soil-sdk/actions/workflows/ci.yml)
[![lint](https://img.shields.io/github/actions/workflow/status/owi-lab/owi-metadatabase-soil-sdk/ci.yml?label=lint)](https://github.com/OWI-Lab/owi-metadatabase-soil-sdk/actions/workflows/ci.yml)
[![issues](https://img.shields.io/github/issues/owi-lab/owi-metadatabase-soil-sdk)](https://github.com/OWI-Lab/owi-metadatabase-soil-sdk/issues)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17531273.svg)](https://doi.org/10.5281/zenodo.10620568)
[coverage](https://img.shields.io/badge/coverage-55%25-red)
[docs](https://img.shields.io/badge/docs-mkdocs-blue)

**Soil extension package for the OWI-metadatabase namespace SDK.**

This package extends `owi-metadatabase` with soil-focused endpoints, data processing,
and plotting helpers while keeping the same `owi.metadatabase.*` import namespace.

📚 **[Read the Documentation](https://owi-lab.github.io/owi-metadatabase-soil/)**

## Features

- **Soil API Client**: Access soil profiles, in-situ tests, geotechnical samples, and lab tests
- **Processing Utilities**: Convert API payloads to tabular/analysis-ready structures
- **Groundhog Integration**: Convert CPT/profile data into Groundhog-compatible objects
- **Visualization Helpers**: Build interactive Plotly maps and profile/fence plots
- **Namespace Extension**: Works seamlessly with the core `owi-metadatabase` package

## Installation

### Install as extension package (`owi-metadatabase-soil`)

From TestPyPI (current deployment target):

```bash
pip install --index-url https://test.pypi.org/simple --extra-index-url https://pypi.org/simple owi-metadatabase-soil
```

Using `uv`:

```bash
uv pip install --index-url https://test.pypi.org/simple --extra-index-url https://pypi.org/simple owi-metadatabase-soil
```

### Install from core package extra (`owi-metadatabase[soil]`)

If you prefer installing from the base package extras:

```bash
pip install --index-url https://test.pypi.org/simple --extra-index-url https://pypi.org/simple "owi-metadatabase[soil]"
```

Using `uv`:

```bash
uv pip install --index-url https://test.pypi.org/simple --extra-index-url https://pypi.org/simple "owi-metadatabase[soil]"
```

## Quick Start

```python
from owi.metadatabase.soil import SoilAPI

soil_api = SoilAPI(token="your-api-token")
print(soil_api.ping())
```

## Examples

### 1) Retrieve test locations

```python
from owi.metadatabase.soil import SoilAPI

soil_api = SoilAPI(token="your-api-token")
result = soil_api.get_testlocations(projectsite="Nobelwind")

if result["exists"]:
    print(result["data"].head())
```

### 2) Find the closest in-situ test

```python
closest = soil_api.get_closest_insitutest(latitude=51.707765, longitude=2.798876, radius=1.0)
print(closest["title"], f"at {closest['offset [m]']:.1f} m")
```

### 3) Retrieve CPT detail and optional Groundhog CPT object

```python
cpt = soil_api.get_cpttest_detail(
    insitutest="BH-CPT-01",
    projectsite="Nobelwind",
    location="NW-A01",
    cpt=True,
)

print(cpt["exists"])
print(cpt["rawdata"].columns)
```

### 4) Retrieve a soil profile detail

```python
profile = soil_api.get_soilprofile_detail(
    projectsite="Nobelwind",
    location="NW-A01",
    soilprofile="BH-Profile-01",
)

print(profile["exists"])
```

### 5) Plot test locations

```python
from owi.metadatabase.soil import SoilPlot

plotter = SoilPlot(soil_api)
figure = plotter.plot_testlocations(return_fig=True, projectsite="Nobelwind")
figure.show()
```

## Development

```bash
uv sync --dev
uv run invoke test.run
uv run invoke qa.all
uv run invoke docs.build
```

### Built with ❤️ and 🧠 by OWI-Lab
