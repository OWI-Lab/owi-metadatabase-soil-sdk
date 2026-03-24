# Prepare SSI Input

The `SoilprofileProcessor` validates and formats soil profile DataFrames
for Soil-Structure Interaction (SSI) analysis methods.

## List available options

```python
from owi.metadatabase.soil.processing.soil_pp import SoilprofileProcessor

# For lateral loading:
options = SoilprofileProcessor.get_available_options(loading="lateral")
print(options)  # ['apirp2geo', 'pisa']
```

## Prepare input for API RP 2GEO

```python
import pandas as pd

df = pd.DataFrame({
    "Depth from [m]": [0.0, 5.0],
    "Depth to [m]": [5.0, 15.0],
    "Soil type": ["SAND", "CLAY"],
    "Total unit weight [kN/m3]": [18.0, 17.5],
    "Undrained shear strength [kPa]": [None, 50.0],
    "Friction angle [deg]": [35.0, None],
    "Epsilon 50 [-]": [None, 0.005],
})

prepared = SoilprofileProcessor.lateral(
    df=df,
    option="apirp2geo",
    mudline=-25.0,    # optional: adds Elevation [mLAT] column
    pw=1.025,         # seawater density for submerged weight
)
print(prepared.head())
```

## Prepare input for PISA

```python
df_pisa = pd.DataFrame({
    "Depth from [m]": [0.0, 5.0],
    "Depth to [m]": [5.0, 15.0],
    "Soil type": ["SAND", "CLAY"],
    "Total unit weight [kN/m3]": [18.0, 17.5],
    "Gmax [kPa]": [50000.0, 30000.0],
    "Undrained shear strength [kPa]": [None, 50.0],
    "Relative density [-]": [0.75, None],
})

prepared_pisa = SoilprofileProcessor.lateral(
    df=df_pisa,
    option="pisa",
)
print(prepared_pisa.head())
```

## Mandatory and optional columns

| Method | Mandatory | Optional |
|--------|-----------|----------|
| `apirp2geo` | Depth from/to, Soil type, γ_total, Su, Φ, ε50 | Dr |
| `pisa` | Depth from/to, Soil type, γ_total, Gmax, Su, Dr | — |

!!! tip
    Column matching is **case-insensitive** and supports partial fragments
    (e.g. `"friction"` matches `"Friction angle [deg]"`).
