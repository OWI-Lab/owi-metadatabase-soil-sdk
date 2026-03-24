# Install the SDK

## As a standalone package

```bash
pip install owi-metadatabase-soil
```

Using `uv`:

```bash
uv pip install owi-metadatabase-soil
```

!!! note
    The package is currently published on **TestPyPI**. Once promoted, the
    same command will work against the main PyPI index.

## As a core-package extra

If you prefer pulling soil support through the base package:

```bash
pip install "owi-metadatabase[soil]"
```

Using `uv`:

```bash
uv pip install "owi-metadatabase[soil]"
```

## Development setup

Clone the repository and install all development dependencies:

```bash
git clone https://github.com/OWI-Lab/owi-metadatabase-soil-sdk.git
cd owi-metadatabase-soil-sdk
uv sync --dev
```
