# Copilot instructions for owi-metadatabase-soil-sdk

## Big picture
- This package is an extension under the `owi.metadatabase` namespace; `src/owi/metadatabase/__init__.py` uses `pkgutil.extend_path` for namespace packaging.
- Soil endpoints are implemented in `src/owi/metadatabase/soil/io.py` via `SoilAPI`, which subclasses `owi.metadatabase.io.API` and appends `/soildata/` to `api_root`.
- Network/data flow is inherited from the base SDK (`send_request` → health check → JSON to DataFrame → postprocess in `process_data`); soil methods mostly provide endpoint-specific params and output shaping.
- Data transformation is split into `SoilDataProcessor` (coordinate transforms, CPT/profile object conversion, table merges) and `SoilprofileProcessor` (SSI input schema validation/prep).
- Plotting lives in `src/owi/metadatabase/soil/visualization/soil_visualizer.py` and composes `SoilAPI` + Groundhog plotting helpers + Plotly.

## Conventions and patterns
- Follow existing return contracts: list endpoints return `{ "data": DataFrame, "exists": bool }`; single/detail endpoints usually add `id` and sometimes `response`.
- Preserve the legacy metadata key spelling `existance` (from base API postprocessing) unless doing a coordinated refactor across SDK + tests.
- Endpoint wrappers consistently build `url_params`, merge `**kwargs`, then call `self.process_data(url_data_type, url_params, "list"|"single")`.
- For recoverable processing issues, current code uses `warnings.warn(...)`; for invalid user input/schema, raise `ValueError`/`InvalidParameterError` patterns already in use.
- Keep NumPy-style docstrings and runnable doctest examples in public methods; doctests are part of CI for both `.py` modules and `*.md` docs.
- Core soil modules currently use `# mypy: ignore-errors`; avoid sweeping strict-typing rewrites unless explicitly requested.

## Python generation and editing rules
- Follow PEP 8, keep edits minimal, and preserve existing module structure and naming conventions.
- Use type hints for new/changed function signatures; avoid broad typing rewrites in legacy files marked with `# mypy: ignore-errors`.
- Prefer f-strings, except logging calls where `%` formatting is preferred; use the `logging` module instead of `print`.
- Use `pathlib` for new path/file operations instead of `os.path`.
- Use explicit exception handling (`except SpecificError`) and avoid bare `except` clauses.
- Prefer `pandas`/`numpy` for tabular and numeric data manipulation, consistent with existing soil processors.
- Keep NumPy-style docstrings with doctests for public methods; this repo runs doctests in CI.
- Prefer max 80 characters for new lines where practical, but do not reflow untouched code; repo lint config currently allows 120.
- For package tests, mirror source naming where feasible (for example, `io.py` → `test_io.py`) and keep tests under `tests/soil/**`.
- Use `uv` for dependency management and execution; run via the workspace environment (`.venv/bin/python` through `uv run ...`).

## Workflows
- Setup: `uv sync --dev`
- Tests: `uv run invoke test.run` (or `uv run invoke test.all`); task includes tests plus doctests for key source files.
- Quality gate: `uv run invoke qa.all` (runs `ruff format`, `ruff check`, and `ty check`).
- Docs: `uv run invoke docs.build` (strict) and `uv run invoke docs.serve`.
- Pytest defaults in `pyproject.toml` enable coverage for `src/owi/metadatabase/soil`, doctests, and markdown doctest globbing.

## Integration points
- External base SDK (`owi-metadatabase`) supplies `API` and shared exceptions used throughout `SoilAPI`.
- `pyproject.toml` configures `owi-metadatabase` from TestPyPI via `[tool.uv.sources]`; keep this in mind when troubleshooting dependency resolution.
- Groundhog is a hard integration point (`profile_from_dataframe`, `PCPTProcessing`, longitudinal/fence plotting); preserve expected DataFrame column names.
- Coordinate conversion relies on `pyproj` from EPSG:4326 to target SRID (default `25831`) in `SoilDataProcessor.transform_coord`.
- Tests heavily mock `owi.metadatabase.soil.io.API.process_data`; keep this call boundary stable when refactoring API methods.

## Useful entry points
- `src/owi/metadatabase/soil/io.py`
- `src/owi/metadatabase/soil/processing/soil_pp.py`
- `src/owi/metadatabase/soil/visualization/soil_visualizer.py`
- `tasks/test.py`, `tasks/quality.py`, `tasks/docs.py`
- `tests/soil/test_io.py`, `tests/soil/processing/test_soil_pp.py`
