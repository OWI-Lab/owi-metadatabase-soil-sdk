# Reference

Reference documentation is **information-oriented**. It provides precise
technical descriptions of every module, class, and function in the SDK,
alongside Django QuerySet examples for the backend soil schema.

## API Reference

Auto-generated documentation pulled from source code docstrings:

| Module | Description |
|--------|-------------|
| [Soil API](api/soil-api.md) | `SoilAPI` — the primary HTTP client for soil endpoints. |
| [Processing](api/processing.md) | `SoilDataProcessor` and `SoilprofileProcessor` — data transformation utilities. |
| [Visualization](api/visualization.md) | `SoilPlot` — Plotly-based fence diagrams and location maps. |

## Query Examples

Django QuerySet examples for the backend soil schema, showing how SDK
filters map to the underlying ORM:

| Page | Scope |
|------|-------|
| [Soil Entity Queries](query-examples/soil-entities.md) | Test locations, in-situ tests, soil profiles, CPTs, lab tests. |
| [Proximity & Corridor Queries](query-examples/proximity.md) | Geographic searches, closest-entity lookups, profile corridors. |
