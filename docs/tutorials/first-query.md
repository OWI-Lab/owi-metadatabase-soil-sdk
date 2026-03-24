# Your First Soil Query

!!! example
    This tutorial walks you through connecting to the OWI Metadatabase,
    listing test locations, retrieving a CPT test, and inspecting the
    returned data. By the end you will have queried real soil data and
    explored the result structure.

## Prerequisites

- Python 3.9+
- The SDK installed (`pip install owi-metadatabase-soil`)
- A valid API token (see [How to Authenticate](../how-to/authenticate.md))

## Step 1 — Create a SoilAPI Client

```python
from owi.metadatabase.soil import SoilAPI

api = SoilAPI(
    api_root="https://owimetadatabase-dev.azurewebsites.net/api/v1",
    token="your-api-token",
)
```

## Step 2 — List Test Locations

```python
result = api.get_testlocations(projectsite="Nobelwind")
print(result["exists"])   # True when rows are returned
print(result["data"].head())
```

The returned dict follows the standard contract:

| Key | Type | Description |
|-----|------|-------------|
| `data` | `DataFrame` | Tabular representation of test locations. |
| `exists` | `bool` | `True` when at least one row was returned. |

## Step 3 — Find an In-Situ Test

```python
insitutests = api.get_insitutests(
    projectsite="Nobelwind",
    testlocation="BH-01",
)
print(insitutests["data"][["id", "title", "testtype"]].head())
```

## Step 4 — Retrieve a CPT Detail

```python
cpt = api.get_cpttest_detail(
    insitutest="BH-CPT-01",
    projectsite="Nobelwind",
    location="NW-A01",
    cpt=True,   # also return Groundhog PCPTProcessing object
)

print(cpt["exists"])
print(cpt["rawdata"].columns.tolist())
```

When `cpt=True` the returned dict also includes a `"cpt"` key containing
a Groundhog `PCPTProcessing` object ready for further analysis.

## Step 5 — Retrieve a Soil Profile

```python
profile = api.get_soilprofile_detail(
    projectsite="Nobelwind",
    location="NW-A01",
    soilprofile="BH-Profile-01",
)

print(profile["exists"])
print(profile["data"].head())
```

## Step 6 — Plot Test Locations on a Map

```python
from owi.metadatabase.soil import SoilPlot

plotter = SoilPlot(api)
figure = plotter.plot_testlocations(
    return_fig=True,
    projectsite="Nobelwind",
)
figure.show()
```

## What You Learned

- How to create and configure a `SoilAPI` client.
- How to list test locations and in-situ tests using backend filters.
- How to retrieve a CPT detail with optional Groundhog object construction.
- How to retrieve a soil profile detail.
- How to plot test locations with `SoilPlot`.

## Next Steps

- [Proximity Search & Fence Diagrams](proximity-and-fences.md) — find
  nearby entities and build fence plots.
- [How-to: Process CPT Data](../how-to/process-cpt.md) — transform raw
  CPT payloads into analysis-ready structures.
