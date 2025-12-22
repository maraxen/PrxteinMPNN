"""Parsing module - all parsing handled by proxide.

The proxide library provides unified parsing for:
- Structure files: PDB, mmCIF, PQR
- Trajectory files: XTC, TRR, DCD
- Dataset files: MD-CATH H5, MDTraj H5

Legacy biotite/mdtraj-based parsers have been removed.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from proxide.core.containers import Protein

# Re-export key functions
from proxide.io.parsing.rust import parse_structure as _parse_structure

from proxide import OutputSpec


def parse_structure(
  file_path: str | Path,
  k_neighbors: int = 48,
  **kwargs: Any,  # noqa: ANN401
) -> Protein:
  """Parse structure using proxide with customized OutputSpec.

  This wrapper enables RBF compilation and ensures Atom37-compatible
  Atom ordering.
  """
  # Create Spec
  spec = OutputSpec()

  # Configure RBF
  # We enable RBF computation by default for PrxteinMPNN ingestion
  spec.compute_rbf = True
  spec.rbf_num_neighbors = k_neighbors

  # Handle physics flags
  compute_physics = kwargs.get("compute_physics", False)
  compute_vdw = kwargs.get("compute_vdw", compute_physics)
  compute_estat = kwargs.get("compute_electrostatics", compute_physics)

  if compute_vdw:
    spec.compute_vdw = True
    spec.parameterize_md = True
  if compute_estat:
    spec.compute_electrostatics = True
    spec.parameterize_md = True

  # Forward other relevant kwargs if they map to OutputSpec
  # (Simplified mapping for now)
  if "remove_solvent" in kwargs:
    spec.remove_solvent = kwargs["remove_solvent"]
  if "add_hydrogens" in kwargs:
    spec.add_hydrogens = kwargs["add_hydrogens"]
  if "force_field" in kwargs:
    spec.force_field = kwargs["force_field"]

  # Call proxide rust parser directly
  return _parse_structure(file_path, spec=spec)


def parse_input(file_path: str | Path, **kwargs: Any) -> Iterator[Protein]:  # noqa: ANN401
  """Unified entry point that defaults to our configured parse_structure."""
  # For now, just route to parse_structure which returns a single Protein
  # The original parse_input yielded generator, so to maintain compat
  # we yield
  yield parse_structure(file_path, **kwargs)


def is_proxide_available() -> bool:
  """Check if proxide is available and functional."""
  try:
    import proxide  # noqa: F401

    return True
  except ImportError:
    return False


# Aliases for specific formats
parse_protein = parse_input
parse_mdcath = parse_input
parse_mdtraj_h5 = parse_input
parse_trajectory = parse_input

__all__ = [
  "is_proxide_available",
  "parse_input",
  "parse_mdcath",
  "parse_mdtraj_h5",
  "parse_protein",
  "parse_structure",
  "parse_trajectory",
]
