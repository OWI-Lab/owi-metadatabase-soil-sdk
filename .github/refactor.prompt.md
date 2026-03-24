# Repository Analysis & Refactoring: `owi-metadatabase-results-sdk`

You are analyzing the `owi-metadatabase-results-sdk` repository — an SDK that handles a specialized subset of metadata from the `owi-metadatabase` parent package, likely in a data analysis, experimentation, or research context.

> **Tooling requirement:** Use `uv` for all Python environment management, dependency installation, script execution, and package operations throughout every phase of this task. Never use `pip`, `poetry`, `conda`, or any other package manager directly.

---

**Phase 1 — Deep Structural Analysis**

Study the full repository in detail. Produce a rigorous, senior-engineer-level assessment covering:

- **Architecture & design patterns** — How is the codebase structured? What paradigms does it follow (OOP, functional, dataclass-driven, etc.)? Are these choices coherent and intentional?
- **Core functionality** — What does this SDK actually *do*? Trace the data flow from ingestion to output. Identify the key abstractions and how they compose.
- **Strengths** — What is done well? Consider API ergonomics, separation of concerns, extensibility, and documentation quality.
- **Weaknesses & technical debt** — Where does the design break down? Flag tight coupling, unclear interfaces, missing abstractions, or anything that would hinder a new contributor.
- **Fit within the parent ecosystem** — How does this SDK relate to `owi-metadatabase`? Are the boundaries between them well-defined?

---

**Phase 2 — Simplified Alternative Architecture and Design Pattern**

Design a cleaner, simplified package structure that preserves 100% of the existing functionality. Your redesign should:

- Reduce unnecessary complexity without sacrificing capability
- Improve discoverability and onboarding for new users
- Respect the constraints of the parent package interface

For each structural decision, explain *why* — not just what changed, but what problem it solves and what tradeoff it makes. Anticipate objections and address them.

Examples of design patterns:

* abstract factory
* factory method
* prototype
* composite
* template method
* structural subtyping (protocol)

---

**Phase 3 — Test Suite (`/tests`)**

Design and implement a comprehensive test suite in the `tests/` folder. Use `uv run pytest` to execute all tests. The suite should cover:

- **Unit tests** — Test each public function, class, and method in isolation. Mock external dependencies (e.g. the parent `owi-metadatabase` package) where necessary. Aim for full branch coverage on core logic.
- **Integration tests** — Test the interaction between internal components and, where feasible, against a real or stubbed instance of the parent package interface.
- **Edge cases & regression tests** — Explicitly test boundary conditions, malformed inputs, and any known failure modes identified during Phase 1.
- **Test configuration** — Set up `pyproject.toml` (managed via `uv`) with a `[tool.pytest.ini_options]` section, and include a `conftest.py` for shared fixtures.

For each test, explain what it is guarding against and why that behavior matters. The suite should be strict enough to catch regressions introduced by the refactoring in Phase 2.

---

**Phase 4 — Implementation Roadmap**

Provide a concrete, sequenced migration plan. Each step should be independently executable and leave the package in a working state. Include dependency order, risk areas, and backward-compatibility considerations. All dependency changes at each step must be applied using `uv add` / `uv remove`, and all code execution must go through `uv run`.

---

**Phase 5 — Notebook Suite (`/notebooks`)**

Implement a set of Jupyter notebooks in the `notebooks/` folder that together provide full coverage of the package's features. Install Jupyter and all required dependencies using `uv add --dev`. Each notebook should:

- Be self-contained and runnable top-to-bottom via `uv run jupyter nbconvert --to notebook --execute`
- Demonstrate a specific capability or workflow
- Serve as both documentation and a functional test

Design the notebook suite so it could double as an interactive tutorial for a new user unfamiliar with the SDK.

---

The additions and changes made:

- **Global `uv` requirement** is stated upfront as a non-negotiable tooling constraint, and reinforced with specific commands (`uv run`, `uv add`, `uv remove`, `uv run pytest`, `uv run jupyter`) at each relevant phase so there's no ambiguity
- **Phase 3 (Test Suite)** is a new dedicated section covering unit tests, integration tests, edge cases, `pyproject.toml` configuration, and `conftest.py` — with the same expectation of justification that applies to the other phases
- The former Phase 3 and 4 are renumbered to **Phase 4 and 5** accordingly
