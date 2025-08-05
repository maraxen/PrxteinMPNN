"""Sphinx configuration file for PrxteinMPNN documentation."""  # noqa: INP001

import sys
from pathlib import Path

sys.path.insert(0, str(Path("../../src").resolve()))

# Workaround to avoid expanding type aliases. See:
# https://github.com/sphinx-doc/sphinx/issues/6518#issuecomment-589613836
from collections.abc import Callable
from typing import ForwardRef


def _do_not_evaluate_in_jax(
  self: ForwardRef,
  globalns: dict,
  *args: tuple,
  _evaluate: Callable = ForwardRef._evaluate,  # noqa: SLF001
  **kwargs: dict,
) -> "ForwardRef":
  if globalns.get("__name__", "").startswith("jax"):
    return self
  return _evaluate(self, globalns, *args, **kwargs)


ForwardRef._evaluate = _do_not_evaluate_in_jax  # type: ignore[attr-access]   # noqa: SLF001

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
  "sphinx_copybutton",
  "myst_nb",
  "sphinx_design",
  "sphinxext.rediraffe",
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
source_suffix = [".rst", ".ipynb", ".md"]

# Intersphinx mapping
intersphinx_mapping = {
  "python": ("https://docs.python.org/3", None),
  "jax": ("https://jax.readthedocs.io/en/latest/", None),
  "numpy": ("https://numpy.org/doc/stable/", None),
}


# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The main toctree document.
main_doc = "index"

language = "en"

autosummary_generate = True

html_theme = "sphinx_book_theme"

html_theme_options = {
  "show_toc_level": 2,
  "repository_url": "https://github.com/maraxen/prxteinmpnn",
  "use_repository_button": True,  # add a "link to repository" button
  "navigation_with_keys": False,
  "article_header_start": ["toggle-primary-sidebar.html", "breadcrumbs"],
}

html_static_path = ["_static"]

html_css_files = [
  "style.css",
]

myst_heading_anchors = 3  # auto-generate 3 levels of heading anchors
myst_enable_extensions = ["dollarmath"]
myst_ref_domains = ["py"]
myst_all_links_external = False
nb_execution_mode = "force"
nb_execution_allow_errors = False
nb_merge_streams = True
nb_execution_show_tb = True

# Notebook cell execution timeout; defaults to 30.
nb_execution_timeout = 100

copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# Tell sphinx autodoc how to render type aliases.
autodoc_typehints = "description"
autodoc_typehints_description_target = "all"
autodoc_type_aliases = {
  "ArrayLike": "jax.typing.ArrayLike",
  "DTypeLike": "jax.typing.DTypeLike",
}
