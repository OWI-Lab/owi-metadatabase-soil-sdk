# Build Soil Profiles

## Convert a soil profile detail to a Groundhog SoilProfile

```python
from owi.metadatabase.soil import SoilAPI
from owi.metadatabase.soil.processing.soil_pp import SoilDataProcessor

api = SoilAPI(token="your-api-token")

detail = api.get_soilprofile_detail(
    projectsite="Nobelwind",
    location="NW-A01",
    soilprofile="BH-Profile-01",
)

profile_obj = SoilDataProcessor.convert_to_profile(
    detail=detail["data"],
    title="BH-Profile-01",
)
```

The returned `SoilProfile` object integrates directly with Groundhog
plotting and engineering routines.

## Batch-load and georeference profiles

When working with many profiles at once, use `objects_to_list` to load
and georeference them in bulk:

```python
profiles_df = api.get_soilprofiles(projectsite="Nobelwind")["data"]

profile_objects = SoilDataProcessor.objects_to_list(
    df=profiles_df,
    api=api,
    object_type="profile",
)
```

Each object is enriched with projected `easting`/`northing` coordinates.

## Gather data for the closest entity

Select the closest entity from a proximity DataFrame and return its
full data alongside the id, title, and offset:

```python
nearby = api.get_proximity_soilprofiles(
    latitude=51.707765,
    longitude=2.798876,
    radius=5.0,
)

closest_data = SoilDataProcessor.gather_data_entity(
    df=nearby["data"],
    api=api,
    entity_type="soilprofile",
)
```
