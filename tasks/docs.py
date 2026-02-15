"""Documentation tasks."""

from invoke import task


@task
def build(c):
    """Build MkDocs docs."""
    c.run("mkdocs build --strict", pty=True)


@task
def serve(c):
    """Serve MkDocs with hot reload."""
    c.run("mkdocs serve", pty=True)


@task(post=[build], default=True)
def all(c):
    """Build docs."""
