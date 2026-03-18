"""Quality assessment tasks. Run all quality checks with `inv qa`.

The quality checks are:
- ruff format
- ruff check
- ty
"""

from typing import Any

from invoke import task

from .colors import Color, colorize
from .system import PTY


@task
def format(c_r: Any) -> None:
    """Run code formatter: ruff."""
    tmp_str = colorize("\nRunning ruff format...\n", color=Color.HEADER, bold=True)
    print(f"{tmp_str}")
    c_r.run("ruff format src tests", pty=PTY)


@task
def lint(c_r: Any) -> None:
    """Run linter: ruff."""
    tmp_str = colorize("\nRunning ruff check...\n", color=Color.HEADER, bold=True)
    print(f"{tmp_str}")
    c_r.run("ruff check src tests", pty=PTY)


@task
def lint_fix(c_r: Any) -> None:
    """Run linter with auto-fix: ruff."""
    tmp_str = colorize("\nRunning ruff check --fix...\n", color=Color.HEADER, bold=True)
    print(f"{tmp_str}")
    c_r.run("ruff check --fix src tests", pty=PTY)


@task
def ty_check(c_r: Any) -> None:
    """Run static type checking: ty."""
    tmp_str = colorize("Running ty...\n", color=Color.HEADER, bold=True)
    print(f"{tmp_str}")
    c_r.run("ty check src/owi tests", warn=True, pty=PTY)


@task(post=[format, lint, ty_check], default=True)
def all(c_r: Any) -> None:
    """Run all quality checks."""
