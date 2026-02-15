# OWI Metadatabase Soil Extension

soil extension for OWI Metadatabase SDK

## Installation

```bash
pip install owi-metadatabase-soil
```

## Usage

```python
from owi.metadatabase.soil.io import SoilAPI

api = SoilAPI(api_key="your-api-key")
print(api.ping())
```

## Development

```bash
uv sync --dev
uv run invoke test.run
uv run invoke qa.all
uv run invoke docs.build
```
