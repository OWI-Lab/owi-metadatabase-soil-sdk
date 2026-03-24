# Quick Start

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
