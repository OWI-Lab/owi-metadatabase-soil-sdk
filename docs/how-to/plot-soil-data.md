# Plot Soil Data

## Plot test locations on a map

```python
from owi.metadatabase.soil import SoilAPI, SoilPlot

api = SoilAPI(token="your-api-token")
plotter = SoilPlot(api)

figure = plotter.plot_testlocations(
    return_fig=True,
    projectsite="Nobelwind",
)
figure.show()
```

The Plotly figure shows a scatter map with lat/lon, title, and project
site labels on an OpenStreetMap basemap.

## Render a soil profile fence diagram

```python
profiles = api.get_soilprofile_profile(
    projectsite="Nobelwind",
    start=(51.70, 2.79),
    end=(51.72, 2.82),
    band=1000,
)

plotter.plot_soilprofile_fence(
    soilprofiles_df=profiles["data"],
    start=(51.70, 2.79),
    end=(51.72, 2.82),
    plotmap=True,
)
```

## Render a CPT fence diagram

```python
cpts = api.get_insitutests_profile(
    projectsite="Nobelwind",
    start=(51.70, 2.79),
    end=(51.72, 2.82),
    band=1000,
)

plotter.plot_cpt_fence(
    cpt_df=cpts["data"],
    start=(51.70, 2.79),
    end=(51.72, 2.82),
    band=1000,
    scale_factor=10,
)
```

## Combined profile + CPT fence

Use the static method for a single plot combining both data sources:

```python
SoilPlot.plot_combined_fence(
    profiles=profiles["data"],
    cpts=cpts["data"],
    startpoint=(51.70, 2.79),
    endpoint=(51.72, 2.82),
    band=1000,
    scale_factor=10,
)
```

## Customise fill colours

Pass a custom colour mapping for soil types in fence diagrams:

```python
colours = {
    "SAND": "#F4D03F",
    "CLAY": "#85C1E9",
    "SILT": "#A9DFBF",
}

plotter.plot_soilprofile_fence(
    soilprofiles_df=profiles["data"],
    start=(51.70, 2.79),
    end=(51.72, 2.82),
    fillcolordict=colours,
)
```
