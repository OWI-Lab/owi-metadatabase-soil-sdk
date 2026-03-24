# Process CPT Data

## Retrieve raw CPT data

```python
from owi.metadatabase.soil import SoilAPI

api = SoilAPI(token="your-api-token")

cpt = api.get_cpttest_detail(
    insitutest="BH-CPT-01",
    projectsite="Nobelwind",
    location="NW-A01",
    cpt=True,
)
```

When `cpt=True`, the returned dict includes a `"cpt"` key with a
Groundhog `PCPTProcessing` object.

## Use SoilDataProcessor directly

For finer control, use the static processor:

```python
from owi.metadatabase.soil.processing.soil_pp import SoilDataProcessor

# Build a PCPTProcessing object from a DataFrame
cpt_obj = SoilDataProcessor.process_cpt(
    rawdata=cpt["rawdata"],
    title="BH-CPT-01",
)
```

## Combine raw and processed data

When both raw and processed DataFrames are available:

```python
combined = SoilDataProcessor.combine_dfs(
    rawdata=cpt["rawdata"],
    processeddata=cpt["processeddata"],
)
print(combined.head())
```

The merge is performed on the `z [m]` column.

## Transform coordinates

Convert WGS-84 coordinates to a projected CRS (default: EPSG:25831):

```python
transformed = SoilDataProcessor.transform_coord(
    df=cpt["data"],
    target_srid=25831,
)
print(transformed[["easting [m]", "northing [m]"]].head())
```
