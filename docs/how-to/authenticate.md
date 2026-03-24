# Authenticate

## Obtain a token

Request an API token from the OWI Metadatabase administrator. The token
grants read (and optionally write) access to the soil endpoints.

## Pass the token to SoilAPI

```python
from owi.metadatabase.soil import SoilAPI

api = SoilAPI(
    api_root="https://owimetadatabase-dev.azurewebsites.net/api/v1",
    token="your-api-token",
)
```

## Store the token in an environment variable

For scripts and notebooks, keep the token out of source code:

```bash
export OWI_METADATABASE_API_TOKEN="your-api-token"
```

```python
import os
from owi.metadatabase.soil import SoilAPI

api = SoilAPI(token=os.environ["OWI_METADATABASE_API_TOKEN"])
```

## Use a `.env` file

Create a `.env` file at the project root:

```text
OWI_METADATABASE_API_TOKEN=your-api-token
```

!!! warning
    Add `.env` to `.gitignore` to avoid committing secrets.
