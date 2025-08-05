"""Sphinx configuration file for PrxteinMPNN documentation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path("../../src").resolve()))

# Project information
project = "PrxteinMPNN"
copyright = "2025, PrxteinMPNN Team"  # noqa: A001
author = "PrxteinMPNN Team"
release = "0.1.0"

# Extensions
extensions = [
  "sphinx.ext.autodoc",
  "sphinx.ext.viewcode",
  "sphinx.ext.napoleon",
  "sphinx.ext.intersphinx",
  "sphinx.ext.mathjax",
  "sphinx_autodoc_typehints",
  "myst_parser",
]

# Napoleon settings for Google-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = True

# Autodoc settings
autodoc_typehints = "description"
autodoc_member_order = "bysource"

# HTML theme
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# Source file suffixes
source_suffix = {
  ".rst": None,
  ".md": "myst_parser",
}

# Intersphinx mapping
intersphinx_mapping = {
  "python": ("https://docs.python.org/3", None),
  "jax": ("https://jax.readthedocs.io/en/latest/", None),
  "numpy": ("https://numpy.org/doc/stable/", None),
}
