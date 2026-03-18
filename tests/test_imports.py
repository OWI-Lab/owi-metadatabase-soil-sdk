"""Test basic package imports."""

from owi.metadatabase.soil import __version__


def test_version():
    """Test that version is accessible."""
    assert __version__ == "0.1.1"
