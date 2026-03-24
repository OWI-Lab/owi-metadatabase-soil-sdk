# Release Process

## Version Scheme

The project uses **SemVer** (`MAJOR.MINOR.PATCH`) managed by
`bump2version` and `GitVersion.yml`.

## CI Release Workflow

1. Add a `release:major`, `release:minor`, or `release:patch` label to
   a PR targeting `main`.
2. The **Release Preview** workflow resolves the next version.
3. On merge, the **Release On Merge** workflow creates a release PR,
   auto-merges it, and tags the commit.
4. The **Publish** workflow pushes the package to PyPI.

## Manual Release

```bash
uv run bump2version patch   # or minor / major
git push --follow-tags
```
