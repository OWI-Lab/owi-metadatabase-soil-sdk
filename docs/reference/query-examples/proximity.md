# Proximity & Corridor Queries

This page shows how `SoilAPI` proximity methods map to the backend's
geographic query capabilities.

---

## 2-D Proximity Search

The backend exposes a proximity endpoint that returns all entities within
a given radius from a point, ordered by distance.

### Find test locations near a coordinate

```python
# Backend ORM (simplified)
TestLocation.objects.annotate(
    distance=Distance("geom", Point(2.798876, 51.707765, srid=4326))
).filter(
    distance__lte=D(km=5.0)
).order_by("distance")
```

**SDK equivalent:**

```python
api.get_proximity_testlocations(
    latitude=51.707765,
    longitude=2.798876,
    radius=5.0,
)
```

### Available proximity methods

| Method | Entity type |
|--------|-------------|
| `get_proximity_testlocations()` | Test locations |
| `get_proximity_insitutests()` | In-situ tests |
| `get_proximity_soilprofiles()` | Soil profiles |
| `get_proximity_batchlabtests()` | Batch lab tests |
| `get_proximity_geotechnicalsamples()` | Geotechnical samples |
| `get_proximity_sampletests()` | Sample tests |

---

## Closest Entity

Returns the single nearest match with a metric `offset [m]` field.

### Find closest in-situ test

```python
api.get_closest_insitutest(
    latitude=51.707765,
    longitude=2.798876,
    radius=5.0,
)
# Returns: {"title": "BH-CPT-01", "offset [m]": 342.7, ...}
```

### Closest entity by name

For cases where you know the entity name but need the detail enriched
with distance information:

```python
api.get_closest_insitutest_byname(
    insitutest="BH-CPT-01",
    projectsite="Nobelwind",
)
```

---

## Expanding Radius Search

When the initial search radius is too narrow, `_search_any_entity` doubles
the radius iteratively up to a 500 km ceiling:

```python
# Internal flow (automatically invoked by closest methods):
# radius: 1 → 2 → 4 → 8 → 16 → ... → 500 km (max)
```

---

## Profile Corridors

The profile corridor endpoints return all entities along a line segment
with a configurable band width.

### Soil profiles along a transect

```python
api.get_soilprofile_profile(
    projectsite="Nobelwind",
    start=(51.70, 2.79),
    end=(51.72, 2.82),
    band=1000,  # corridor width in metres
)
```

### In-situ tests along a transect

```python
api.get_insitutests_profile(
    projectsite="Nobelwind",
    start=(51.70, 2.79),
    end=(51.72, 2.82),
    band=1000,
)
```

These methods are the data source for fence diagram visualisations
(see [How-to: Plot Soil Data](../../how-to/plot-soil-data.md)).

---

## Unit-Level Aggregation

### Get depth ranges for soil units

```python
api.get_soilunit_depthranges(
    projectsite="Nobelwind",
    soilprofile="BH-Profile-01",
)
```

### Get in-situ test data within unit depth ranges

```python
api.get_unit_insitutestdata(
    projectsite="Nobelwind",
    soilprofile="BH-Profile-01",
    insitutest="BH-CPT-01",
)
```

### Get sample tests within unit depth ranges

```python
api.get_unit_sampletests(
    projectsite="Nobelwind",
    soilprofile="BH-Profile-01",
)
```
