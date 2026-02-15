"""
OWI Metadatabase SDK - Core package for geometry and location data.

This is a namespace package that provides access to the OWI-Lab
metadatabase for offshore wind installations. The package follows
pkgutil namespace conventions to allow for modular extensions.

Modules
-------
geometry : Module for handling geometry data
locations : Module for handling location data
"""

import pkgutil

__path__ = pkgutil.extend_path(__path__, __name__)

from owi.metadatabase._version import __version__

__all__ = ["__version__"]
