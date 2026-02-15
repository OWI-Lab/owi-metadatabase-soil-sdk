# Soil Extension

The `owi-metadatabase-soil` package extends the `owi.metadatabase` namespace
with soil-specific API endpoints, processing helpers, and plotting utilities.

## Scope

- Retrieve soil data entities (locations, profiles, in-situ tests, samples, lab tests)
- Post-process profile/CPT payloads into analysis-ready tabular structures
- Build profile and CPT fence diagrams with Plotly/Groundhog

## Quick Example

```python
from owi.metadatabase.soil import SoilAPI

api = SoilAPI()
soilprofiles = api.get_soilprofiles(projectsite="Nobelwind")
print(soilprofiles["exists"])
```

## Documentation Style

API docstrings in this extension follow the NumPy docstring convention and
include doctest-style usage examples.
