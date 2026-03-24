# Data Model

The soil backend schema centres on **test locations** that anchor all
in-situ tests, soil profiles, and laboratory data.

## Entity-Relationship Diagram

```mermaid
erDiagram
    ProjectSite ||--o{ AssetLocation : contains
    ProjectSite ||--o{ TestLocation : contains
    ProjectSite ||--o{ SurveyCampaign : runs

    TestLocation ||--o{ InSituTest : hosts
    TestLocation ||--o{ GeotechnicalSample : hosts

    AssetLocation ||--o{ SoilProfile : has

    InSituTest ||--o{ CPTTest : specialises

    SoilProfile ||--o{ SoilUnit : contains
    SoilUnit }o--|| SoilType : classifies

    GeotechnicalSample ||--o{ SampleTest : tested_in
    GeotechnicalSample ||--o{ BatchLabTest : grouped_in

    SurveyCampaign ||--o{ TestLocation : covers
```

## Key Entities

### TestLocation

The geographic anchor for all field work. Identified by `title` within a
`ProjectSite`. Carries WGS-84 `latitude`/`longitude` coordinates.

### InSituTest

A field test performed at a test location (CPT, borehole, etc.).
The `testtype` discriminator determines which detail endpoint applies.

### CPTTest

Specialisation of `InSituTest` carrying raw and processed depth-series
DataFrames (`rawdata`, `processeddata`). Can be converted to a Groundhog
`PCPTProcessing` object.

### SoilProfile

An interpreted soil stratigraphy associated with an `AssetLocation`.
Contains ordered `SoilUnit` layers with depth ranges and geotechnical
properties.

### SoilUnit

A single layer within a profile, classified by `SoilType`. Properties
include total unit weight, undrained shear strength, friction angle,
relative density, G_max, and ε50.

### GeotechnicalSample / SampleTest / BatchLabTest

Laboratory entities linked to test locations. Used for parameter
characterisation and unit-level aggregation through
`SoilAPI.get_unit_sampletests()`.

### SurveyCampaign

Groups test locations into named campaigns (e.g. "2022 Q3 Geotechnical
Survey").

## Data Flow Through the SDK

```mermaid
flowchart LR
    REST[Backend REST API]
    API[SoilAPI]
    DF[DataFrame]
    GH[Groundhog objects]
    VIS[SoilPlot]
    FIG[Plotly Figure]

    REST -->|JSON| API -->|postprocess| DF
    DF -->|SoilDataProcessor| GH
    DF --> VIS
    GH --> VIS
    VIS -->|render| FIG
```
