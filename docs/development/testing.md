# Testing

## Running the Test Suite

```bash
uv run inv test
```

This executes pytest with:

- **Coverage** reporting for `src/owi/metadatabase/soil`
- **Doctests** from source modules and Markdown files
- **Doctest directives**: `NORMALIZE_WHITESPACE`, `ELLIPSIS`, `NUMBER`

## Test Structure

```
tests/
├── conftest.py                  # shared fixtures
├── test_imports.py              # version assertion
├── soil/
│   ├── test_io.py               # SoilAPI method tests
│   └── processing/
│       └── test_soil_pp.py      # SoilDataProcessor + SoilprofileProcessor
```

## Mocking Strategy

Tests mock `owi.metadatabase.soil.io.API.process_data` to avoid live HTTP
calls. This boundary is stable — keep it when refactoring API methods.

## Adding Tests

1. Mirror the source module path (e.g. `io.py` → `test_io.py`).
2. Use the existing `header`, `soil_init`, and `api_soil` fixtures.
3. Parameterise with `@pytest.mark.parametrize` for schema variations.
