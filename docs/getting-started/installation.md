# Installation

## Install as extension package (`owi-metadatabase-soil`)

From TestPyPI (current deployment target):

```bash
pip install --index-url https://test.pypi.org/simple --extra-index-url https://pypi.org/simple owi-metadatabase-soil
```

Using `uv`:

```bash
uv pip install --index-url https://test.pypi.org/simple --extra-index-url https://pypi.org/simple owi-metadatabase-soil
```

## Install from core package extra (`owi-metadatabase[soil]`)

If you prefer installing from the base package extras:

```bash
pip install --index-url https://test.pypi.org/simple --extra-index-url https://pypi.org/simple "owi-metadatabase[soil]"
```

Using `uv`:

```bash
uv pip install --index-url https://test.pypi.org/simple --extra-index-url https://pypi.org/simple "owi-metadatabase[soil]"
```

## Development setup

```bash
uv sync --dev
```
