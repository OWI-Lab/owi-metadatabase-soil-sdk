# OWI Metadatabase Soil Extension

!!! abstract "What is the OWI Metadatabase Soil SDK?"
    The `owi-metadatabase-soil` package extends the `owi.metadatabase` namespace
    with **soil-specific API endpoints**, processing helpers, and plotting
    utilities. Retrieve test locations, in-situ tests, CPTs, soil profiles,
    lab tests, and geotechnical samples — then transform, analyse, and
    visualise them through a single SDK.

<div class="grid cards" markdown>

-   :material-school:{ .lg .middle } **Tutorials**

    ---

    Step-by-step lessons that walk you through connecting to the soil API,
    retrieving data, and building your first fence diagram.

    [:octicons-arrow-right-24: Start learning](tutorials/index.md)

-   :material-tools:{ .lg .middle } **How-to Guides**

    ---

    Focused recipes for common tasks: install, authenticate, query soil
    entities, process CPTs, build profiles, and plot results.

    [:octicons-arrow-right-24: Find a recipe](how-to/index.md)

-   :material-book-open-variant:{ .lg .middle } **Reference**

    ---

    Auto-generated API docs for `SoilAPI`, processors, and visualisers,
    plus Django QuerySet examples for every soil entity.

    [:octicons-arrow-right-24: Browse reference](reference/index.md)

-   :material-lightbulb-on:{ .lg .middle } **Explanation**

    ---

    Deeper discussions on architecture, the soil data model,
    SSI workflows, and namespace packaging.

    [:octicons-arrow-right-24: Understand concepts](explanation/index.md)

</div>

## Quick Example

```python
from owi.metadatabase.soil import SoilAPI

api = SoilAPI(token="your-api-token")
soilprofiles = api.get_soilprofiles(projectsite="Nobelwind")
print(soilprofiles["exists"])
```
