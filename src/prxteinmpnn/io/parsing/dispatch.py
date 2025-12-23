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

from proxide import OutputSpec
from proxide.core.containers import Protein
from proxide.io.parsing.rust import parse_structure as _parse_structure


def parse_structure(
  file_path: str | Path,
  k_neighbors: int = 48,
  **kwargs: Any,  # noqa: ANN401
) -> Protein:
  """Parse structure using proxide with customized OutputSpec.

  This wrapper enables RBF compilation and ensures Atom37-compatible
  Atom ordering.
  """
  # Handle physics flags
  compute_physics = kwargs.get("compute_physics", False)
  compute_vdw = kwargs.get("compute_vdw", compute_physics)
  compute_estat = kwargs.get("compute_electrostatics", compute_physics)

  # Build OutputSpec arguments
  spec_args = {
    "compute_rbf": True,
    "rbf_num_neighbors": k_neighbors,
    "compute_vdw": compute_vdw,
    "compute_electrostatics": compute_estat,
    "parameterize_md": compute_vdw or compute_estat,
  }

  # Forward other relevant kwargs if they map to OutputSpec
  if "remove_solvent" in kwargs:
    spec_args["remove_solvent"] = kwargs["remove_solvent"]
  if "add_hydrogens" in kwargs:
    spec_args["add_hydrogens"] = kwargs["add_hydrogens"]
  if "force_field" in kwargs:
    spec_args["force_field"] = kwargs["force_field"]

  spec = OutputSpec(**spec_args)

  # Call proxide rust parser directly
  return _parse_structure(file_path, spec=spec)


def parse_input(file_path: str | Path, **kwargs: Any) -> Iterator[Protein]:  # noqa: ANN401
  """Unified entry point that defaults to our configured parse_structure."""
  # For now, just route to parse_structure which returns a single Protein
  # The original parse_input yielded generator, so to maintain compat
  # we yield
  yield parse_structure(file_path, **kwargs)


# Aliases for specific formats
parse_protein = parse_input
parse_mdcath = parse_input
parse_mdtraj_h5 = parse_input
parse_trajectory = parse_input

__all__ = [
  "parse_input",
  "parse_mdcath",
  "parse_mdtraj_h5",
  "parse_protein",
  "parse_structure",
  "parse_trajectory",
]
