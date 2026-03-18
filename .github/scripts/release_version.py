"""Release version planning and file synchronization helpers."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYPROJECT_PATH = ROOT / "pyproject.toml"
PACKAGE_VERSION_PATH = ROOT / "src/owi/metadatabase/soil/__init__.py"
BUMPVERSION_PATH = ROOT / ".bumpversion.cfg"
GITVERSION_PATH = ROOT / "GitVersion.yml"
UV_LOCK_PATH = ROOT / "uv.lock"
TEST_IMPORTS_PATH = ROOT / "tests/test_imports.py"
INSTALLATION_DOC_PATH = ROOT / "docs/getting-started/installation.md"

RELEASE_LABELS = {
    "release:major": "major",
    "release:minor": "minor",
    "release:patch": "patch",
}
SEMVER_PATTERN = re.compile(r"^(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)$")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _replace_once(content: str, pattern: str, replacement: str, path: Path) -> str:
    updated, count = re.subn(pattern, replacement, content, count=1, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(f"Expected exactly one match for {path}")
    return updated


def read_project_version() -> str:
    content = _read_text(PYPROJECT_PATH)
    match = re.search(
        r'^\[project\]$(?:\n(?!\[).*)*?^version = "(?P<version>\d+\.\d+\.\d+)"$',
        content,
        flags=re.MULTILINE,
    )
    if match is None:
        raise ValueError("Could not resolve project version from pyproject.toml")
    return match.group("version")


def parse_labels(labels_json: str) -> list[str]:
    labels = json.loads(labels_json)
    if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
        raise ValueError("Labels payload must be a JSON array of strings")
    return labels


def resolve_bump_type(labels: list[str]) -> str:
    requested = [bump for label, bump in RELEASE_LABELS.items() if label in set(labels)]
    if len(requested) > 1:
        supported = ", ".join(sorted(RELEASE_LABELS))
        raise ValueError(f"Multiple release labels detected. Use at most one of: {supported}")
    if requested:
        return requested[0]
    return "patch"


def bump_version(current_version: str, bump_type: str) -> str:
    match = SEMVER_PATTERN.match(current_version)
    if match is None:
        raise ValueError(f"Unsupported version format: {current_version}")

    major = int(match.group("major"))
    minor = int(match.group("minor"))
    patch = int(match.group("patch"))

    if bump_type == "major":
        return f"{major + 1}.0.0"
    if bump_type == "minor":
        return f"{major}.{minor + 1}.0"
    if bump_type == "patch":
        return f"{major}.{minor}.{patch + 1}"
    raise ValueError(f"Unsupported bump type: {bump_type}")


def write_github_output(output_path: str | None, values: dict[str, str]) -> None:
    if output_path is None:
        return

    lines = [f"{key}={value}" for key, value in values.items()]
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def apply_version(new_version: str) -> list[str]:
    if SEMVER_PATTERN.match(new_version) is None:
        raise ValueError(f"Unsupported version format: {new_version}")

    changed_paths: list[str] = []

    replacements = [
        (
            PYPROJECT_PATH,
            r'(^\[project\]$(?:\n(?!\[).*)*?^version = ")\d+\.\d+\.\d+("$)',
            rf"\g<1>{new_version}\2",
        ),
        (
            PACKAGE_VERSION_PATH,
            r'^__version__ = "\d+\.\d+\.\d+"$',
            f'__version__ = "{new_version}"',
        ),
        (
            BUMPVERSION_PATH,
            r"^current_version = \d+\.\d+\.\d+$",
            f"current_version = {new_version}",
        ),
        (
            GITVERSION_PATH,
            r"^next-version: \d+\.\d+\.\d+$",
            f"next-version: {new_version}",
        ),
        (
            UV_LOCK_PATH,
            r'(\[\[package\]\]\nname = "owi-metadatabase-soil"\nversion = ")\d+\.\d+\.\d+("\nsource = \{ editable = "\." \})',
            rf"\g<1>{new_version}\2",
        ),
        (
            TEST_IMPORTS_PATH,
            r'^(\s*assert __version__ == ")\d+\.\d+\.\d+("\s*)$',
            rf"\g<1>{new_version}\2",
        ),
        (
            INSTALLATION_DOC_PATH,
            r"^(# Output: )\d+\.\d+\.\d+(\s*)$",
            rf"\g<1>{new_version}\2",
        ),
        (
            INSTALLATION_DOC_PATH,
            r"^(# ✓ New \(owi-metadatabase-soil v)\d+\.\d+\.\d+(\+\)\s*)$",
            rf"\g<1>{new_version}\2",
        ),
    ]

    for path, pattern, replacement in replacements:
        original = _read_text(path)
        updated = _replace_once(original, pattern, replacement, path)
        if updated != original:
            path.write_text(updated, encoding="utf-8")
            relative_path = str(path.relative_to(ROOT))
            if relative_path not in changed_paths:
                changed_paths.append(relative_path)

    return changed_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan_parser = subparsers.add_parser("plan", help="Resolve the next release version from labels")
    plan_parser.add_argument("--labels-json", default="[]", help="JSON array of GitHub label names")
    plan_parser.add_argument("--github-output", help="Path to the GitHub Actions output file")

    apply_parser = subparsers.add_parser("apply", help="Update tracked files to a new version")
    apply_parser.add_argument("--new-version", required=True, help="Version to write across tracked files")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "plan":
        labels = parse_labels(args.labels_json)
        current_version = read_project_version()
        bump_type = resolve_bump_type(labels)
        next_version = bump_version(current_version, bump_type)

        payload = {
            "current_version": current_version,
            "bump_type": bump_type,
            "next_version": next_version,
        }
        print(json.dumps(payload))
        write_github_output(args.github_output, payload)
        return 0

    if args.command == "apply":
        changed_paths = apply_version(args.new_version)
        print(json.dumps({"new_version": args.new_version, "changed_files": changed_paths}))
        return 0

    parser.error("Unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
