# Soil Entity Queries

This page shows Django QuerySet examples for the backend soil schema.
These are the ORM operations that run when you call `SoilAPI` methods —
understanding them helps you choose the right keyword arguments.

---

## Test Locations

### List all test locations for a project site

```python
TestLocation.objects.filter(
    projectsite__title="Nobelwind"
)
```

**SDK equivalent:**

```python
api.get_testlocations(projectsite="Nobelwind")
```

### Check whether a test location exists

```python
TestLocation.objects.filter(
    projectsite__title="Nobelwind",
    title="BH-01",
).exists()
```

**SDK equivalent:**

```python
api.testlocation_exists(projectsite="Nobelwind", testlocation="BH-01")
```

---

## In-Situ Tests

### List in-situ tests at a test location

```python
InSituTest.objects.filter(
    test_location__title="BH-01",
    test_location__projectsite__title="Nobelwind",
)
```

**SDK equivalent:**

```python
api.get_insitutests(projectsite="Nobelwind", testlocation="BH-01")
```

### Filter by test type

```python
InSituTest.objects.filter(
    test_location__projectsite__title="Nobelwind",
    testtype="CPT",
)
```

**SDK equivalent:**

```python
api.get_insitutests(projectsite="Nobelwind", testtype="CPT")
```

### Check in-situ test existence

```python
InSituTest.objects.filter(
    title="BH-CPT-01",
    test_location__projectsite__title="Nobelwind",
).exists()
```

**SDK equivalent:**

```python
api.insitutest_exists(projectsite="Nobelwind", insitutest="BH-CPT-01")
```

---

## Soil Profiles

### List all profiles for a project

```python
SoilProfile.objects.filter(
    location__projectsite__title="Nobelwind"
)
```

**SDK equivalent:**

```python
api.get_soilprofiles(projectsite="Nobelwind")
```

### Filter by location and profile name

```python
SoilProfile.objects.filter(
    location__projectsite__title="Nobelwind",
    location__title="NW-A01",
    title="BH-Profile-01",
)
```

**SDK equivalent:**

```python
api.get_soilprofile_detail(
    projectsite="Nobelwind",
    location="NW-A01",
    soilprofile="BH-Profile-01",
)
```

---

## Batch Lab Tests

### List batch lab tests for a project

```python
BatchLabTest.objects.filter(
    projectsite__title="Nobelwind"
)
```

**SDK equivalent:**

```python
api.get_batchlabtests(projectsite="Nobelwind")
```

---

## Geotechnical Samples

### List all samples at a project site

```python
GeotechnicalSample.objects.filter(
    projectsite__title="Nobelwind"
)
```

**SDK equivalent:**

```python
api.get_geotechnicalsamples(projectsite="Nobelwind")
```

---

## Sample Tests

### List sample tests at a project site

```python
SampleTest.objects.filter(
    projectsite__title="Nobelwind"
)
```

**SDK equivalent:**

```python
api.get_sampletests(projectsite="Nobelwind")
```

---

## Soil Units and Types

### List soil units for a profile

```python
SoilUnit.objects.filter(
    soil_profile__title="BH-Profile-01",
    soil_profile__location__projectsite__title="Nobelwind",
)
```

**SDK equivalent:**

```python
api.get_soilunits(
    projectsite="Nobelwind",
    soilprofile="BH-Profile-01",
)
```

### List soil types

```python
SoilType.objects.all()
```

**SDK equivalent:**

```python
api.get_soiltypes()
```

---

## Survey Campaigns

### List campaigns for a project

```python
SurveyCampaign.objects.filter(
    projectsite__title="Nobelwind"
)
```

**SDK equivalent:**

```python
api.get_surveycampaigns(projectsite="Nobelwind")
```
