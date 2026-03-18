import os
import shutil
from typing import Any

from invoke import task

from .colors import Color, colorize
from .system import (
    PTY,
    get_current_system,
)

SYSTEM = get_current_system()

@task
def clean(c_r: Any) -> None:
    """Clean documentation build folder"""
    site_dir = "site"
    if os.path.exists(site_dir):
        shutil.rmtree(site_dir)


@task
def build(c_r: Any) -> None:
    """Build MkDocs documentation."""
    tmp_str = colorize(
        "\nBuilding MkDocs documentation...\n",
        color=Color.HEADER,
        bold=True
    )
    print(tmp_str)
    _command = "mkdocs build --strict"
    print(f">>> {colorize(_command, color=Color.OKBLUE)}\n")
    c_r.run(_command, pty=PTY)


@task
def serve(c_r: Any) -> None:
    """Serve MkDocs documentation with hot reload."""
    DOCS_PORT = c_r.start_port + 1

    tmp_str = colorize(
        "\nStarting MkDocs server with hot reload...\n",
        color=Color.HEADER,
        bold=True
    )
    print(tmp_str)

    _command = f"mkdocs serve --dev-addr localhost:{DOCS_PORT}"
    print(f">>> {colorize(_command, color=Color.OKBLUE)}\n")

    url = f"http://localhost:{DOCS_PORT}"
    print("Documentation server with hot reload hosted at:\n")
    print(f"--> {colorize(url, underline=True)}\n")
    print(f"Stop server: {colorize('Ctrl+C')}\n")

    c_r.run(_command, pty=PTY)


@task
def deploy(c_r: Any) -> None:
    """Deploy documentation to GitHub Pages."""
    tmp_str = colorize(
        "\nDeploying documentation to GitHub Pages...\n",
        color=Color.HEADER,
        bold=True
    )
    print(tmp_str)
    _command = "mkdocs gh-deploy --force"
    print(f">>> {colorize(_command, color=Color.OKBLUE)}\n")
    c_r.run(_command, pty=PTY)


@task(post=[clean, build], default=True)
def all(c_r: Any) -> None:
    """Clean and build documentation."""


@task
def deploy_version(c_r: Any, version: str, alias: str = "latest") -> None:
    """Deploy versioned documentation with mike."""
    tmp_str = colorize(
        f"\nDeploying docs version {version}...\n",
        color=Color.HEADER,
        bold=True,
    )
    print(tmp_str)
    if alias == version:
        _command = f"mike deploy --push {version}"
    else:
        _command = f"mike deploy --push {version} {alias}"
    print(f">>> {colorize(_command, color=Color.OKBLUE)}\n")
    c_r.run(_command, pty=PTY)


@task
def set_default_version(c_r: Any, version: str) -> None:
    """Set the default documentation version with mike."""
    tmp_str = colorize(
        f"\nSetting default docs version to {version}...\n",
        color=Color.HEADER,
        bold=True,
    )
    print(tmp_str)
    _command = f"mike set-default --push {version}"
    print(f">>> {colorize(_command, color=Color.OKBLUE)}\n")
    c_r.run(_command, pty=PTY)
