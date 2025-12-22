"""Parsing module - all parsing handled by proxide.

The proxide library provides unified parsing for:
- Structure files: PDB, mmCIF, PQR
- Trajectory files: XTC, TRR, DCD
- Dataset files: MD-CATH H5, MDTraj H5

Legacy biotite/mdtraj-based parsers have been removed.
"""

from prxteinmpnn.io.parsing.dispatch import (
  is_proxide_available,
  parse_input,
  parse_mdcath,
  parse_mdtraj_h5,
  parse_protein,
  parse_structure,
  parse_trajectory,
)

__all__ = [
  "is_proxide_available",
  "parse_input",
  "parse_mdcath",
  "parse_mdtraj_h5",
  "parse_protein",
  "parse_structure",
  "parse_trajectory",
]
