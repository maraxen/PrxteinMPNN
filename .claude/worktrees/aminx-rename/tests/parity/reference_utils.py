"""Utilities for locating and importing the LigandMPNN reference repository."""

from __future__ import annotations

import os
import sys
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest


def project_root() -> Path:
  """Return repository root directory."""
  return Path(__file__).resolve().parents[2]


def _candidate_reference_paths() -> list[Path]:
  root = project_root()
  env_value = os.environ.get("REFERENCE_PATH")
  candidates: list[Path] = []
  if env_value:
    candidates.append(Path(env_value))
  candidates.extend(
    [
      root / "reference_ligandmpnn_clone",
      root.parent / "reference_ligandmpnn_clone",
    ],
  )
  return candidates


def require_reference_path() -> Path:
  """Return LigandMPNN reference repo path or skip if unavailable."""
  for candidate in _candidate_reference_paths():
    if candidate.is_dir():
      return candidate
  searched = ", ".join(str(path) for path in _candidate_reference_paths())
  pytest.skip(
    f"LigandMPNN reference repo not found. Set REFERENCE_PATH or checkout "
    f"reference_ligandmpnn_clone. Searched: {searched}",
    allow_module_level=True,
  )


def prepend_reference_to_syspath() -> Path:
  """Add reference repo path to sys.path and return it."""
  reference_path = require_reference_path()
  ref_str = str(reference_path)
  if ref_str not in sys.path:
    sys.path.insert(0, ref_str)
  return reference_path


def require_heavy_parity_prereqs(
  *,
  python_modules: list[str] | None = None,
  reference_rel_paths: list[str] | None = None,
  converted_rel_paths: list[str] | None = None,
) -> tuple[Path, Path]:
  """Validate heavy parity prerequisites and return `(reference_root, project_root)`."""
  if find_spec("torch") is None:
    pytest.skip("Heavy parity requires torch in the active environment.", allow_module_level=True)
  for module_name in python_modules or []:
    if find_spec(module_name) is None:
      pytest.skip(
        f"Heavy parity requires optional dependency '{module_name}'.",
        allow_module_level=True,
      )

  reference_root = prepend_reference_to_syspath()
  repo_root = project_root()

  for rel_path in reference_rel_paths or []:
    path = reference_root / rel_path
    if not path.exists():
      pytest.skip(
        f"Heavy parity missing reference asset: {path}",
        allow_module_level=True,
      )

  for rel_path in converted_rel_paths or []:
    path = repo_root / rel_path
    if not path.exists():
      pytest.skip(
        f"Heavy parity missing converted asset: {path}",
        allow_module_level=True,
      )

  return reference_root, repo_root


def _ensure_reference_numpy_aliases() -> None:
  """Patch NumPy aliases expected by older reference dependencies."""
  aliases: dict[str, type[object]] = {
    "bool": bool,
    "int": int,
    "object": object,
  }
  for name, target in aliases.items():
    if name not in np.__dict__:
      setattr(np, name, target)


def import_reference_module(module_name: str) -> ModuleType:
  """Import a module from the reference checkout or skip if it cannot import."""
  prepend_reference_to_syspath()
  _ensure_reference_numpy_aliases()
  try:
    return import_module(module_name)
  except (ImportError, ModuleNotFoundError) as error:
    pytest.skip(
      f"Unable to import reference module '{module_name}': {error}",
      allow_module_level=True,
    )
