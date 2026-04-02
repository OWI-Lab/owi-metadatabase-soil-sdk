# Contributing

## Development Setup

```bash
git clone https://github.com/OWI-Lab/owi-metadatabase-soil-sdk.git
cd owi-metadatabase-soil-sdk
uv sync --dev
```

## Code Style

The project uses **ruff** for formatting and linting (120-char lines) and
**ty** for type checking:

```bash
uv run inv qa
```

## Running Tests

```bash
uv run inv test
```

This runs pytest with coverage and doctests enabled.

## Pre-commit Hooks

Install the hooks once:

```bash
uv run pre-commit install
```

Hooks run ruff format/check, ty, and pytest automatically.

## Pull Request Workflow

1. Create a feature branch from `main`.
2. Make your changes with tests.
3. Ensure `uv run inv qa` and `uv run inv test` pass.
4. Open a PR against `main`.
