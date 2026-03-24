# Query Soil Entities

## List entities

Every entity type follows the same list/detail/exists pattern:

```python
from owi.metadatabase.soil import SoilAPI

api = SoilAPI(token="your-api-token")

# Test locations
locations = api.get_testlocations(projectsite="Nobelwind")

# In-situ tests
insitu = api.get_insitutests(projectsite="Nobelwind", testlocation="BH-01")

# Soil profiles
profiles = api.get_soilprofiles(projectsite="Nobelwind")

# Batch lab tests
labtests = api.get_batchlabtests(projectsite="Nobelwind")
```

All list methods return:

```python
{"data": DataFrame, "exists": bool}
```

## Filter by project and location

Most methods accept `projectsite` and `location` keyword arguments that
map to the backend's Django filter fields:

```python
profiles = api.get_soilprofiles(
    projectsite="Nobelwind",
    location="NW-A01",
)
```

## Check existence

Use the dedicated existence methods when you only need a boolean check:

```python
api.testlocation_exists(projectsite="Nobelwind", testlocation="BH-01")
api.insitutest_exists(projectsite="Nobelwind", insitutest="BH-CPT-01")
api.soilprofile_exists(projectsite="Nobelwind", soilprofile="BH-Profile-01")
```

## Retrieve entity details

Detail methods return richer data, including nested objects:

```python
detail = api.get_soilprofile_detail(
    projectsite="Nobelwind",
    location="NW-A01",
    soilprofile="BH-Profile-01",
)
print(detail["data"].head())
```

## Proximity search

Find entities within a geographic radius:

```python
nearby = api.get_proximity_testlocations(
    latitude=51.707765,
    longitude=2.798876,
    radius=5.0,
)
```

## Closest entity

Return only the single closest match and its metric offset:

```python
closest = api.get_closest_testlocation(
    latitude=51.707765,
    longitude=2.798876,
    radius=5.0,
)
print(closest["title"], closest["offset [m]"])
```

## Pass extra keyword arguments

All methods forward unrecognized keyword arguments to the underlying
base API request, allowing you to use any backend filter field:

```python
api.get_insitutests(
    projectsite="Nobelwind",
    testtype="CPT",      # backend-specific filter
    campaign__title="2022 Q3",
)
```
