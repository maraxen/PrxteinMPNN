"""Sphinx configuration file for PrxteinMPNN documentation."""  # noqa: INP001

import sys
from collections.abc import Callable
from pathlib import Path
from typing import ForwardRef

sys.path.insert(0, str(Path("../../src").resolve()))

# Workaround to avoid expanding type aliases. See:
# https://github.com/sphinx-doc/sphinx/issues/6518#issuecomment-589613836
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


ForwardRef._evaluate = _do_not_evaluate_in_jax  # type: ignore[attr-access]  # noqa: SLF001

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
autodoc_typehints_description_target = "all"
autodoc_member_order = "bysource"
autodoc_type_aliases = {
  "ArrayLike": "jax.typing.ArrayLike",
  "DTypeLike": "jax.typing.DTypeLike",
}

# HTML theme
html_theme = "sphinx_book_theme"
html_theme_options = {
  "show_toc_level": 2,
  "repository_url": "https://github.com/maraxen/prxteinmpnn",
  "use_repository_button": True,
  "navigation_with_keys": False,
}
html_static_path = ["_static"]
html_css_files = ["style.css"]

# Source file suffixes
source_suffix = [".rst", ".ipynb", ".md"]

# Intersphinx mapping
intersphinx_mapping = {
  "python": ("https://docs.python.org/3", None),
  "jax": ("https://jax.readthedocs.io/en/latest/", None),
  "numpy": ("https://numpy.org/doc/stable/", None),
}

templates_path = ["_templates"]
main_doc = "index"
language = "en"
autosummary_generate = True

# MyST / notebook settings
myst_heading_anchors = 3
myst_enable_extensions = ["dollarmath"]
myst_ref_domains = ["py"]
myst_all_links_external = False
nb_execution_mode = "off"  # don't execute notebooks on RTD (no GPU/JAX)
nb_execution_allow_errors = False
nb_merge_streams = True

copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
