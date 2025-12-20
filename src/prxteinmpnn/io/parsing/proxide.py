"""Proxide-based structure parsing.

This module provides a high-level interface to proxide's structure parsing
capabilities, including physics feature extraction.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from prxteinmpnn.utils.data_structures import ProteinTuple
from prxteinmpnn.utils.residue_constants import atom_order

if TYPE_CHECKING:
  from collections.abc import Generator

try:
  from proxide import OutputSpec, parse_pqr, parse_structure
  from proxide.physics.features import (
    compute_electrostatic_node_features,
    compute_vdw_node_features,
  )

  PROXIDE_AVAILABLE = True
except ImportError:
  PROXIDE_AVAILABLE = False
  OutputSpec = None
  parse_pqr = None
  parse_structure = None
  compute_electrostatic_node_features = None
  compute_vdw_node_features = None

logger = logging.getLogger(__name__)


def _reshape_coordinates(
  flat_coords: np.ndarray,
  n_residues: int,
) -> np.ndarray:
  """Reshape flat coordinates to (n_residues, 5, 3) backbone format.

  Proxide returns flat coordinates. We need to extract backbone atoms
  (N, CA, C, O, CB) per residue.
  """
  n_atoms = len(flat_coords) // 3
  coords_3d = flat_coords.reshape(n_atoms, 3)

  # For now, we assume 5 backbone atoms per residue in order
  if n_atoms == n_residues * 5:
    return coords_3d.reshape(n_residues, 5, 3)

  # Otherwise, we need to extract from full coordinates
  return coords_3d[: n_residues * 5].reshape(n_residues, 5, 3)


def _atom_mask_to_37(
  flat_mask: np.ndarray,
  n_residues: int,
) -> np.ndarray:
  """Convert flat atom mask to (n_residues, 37) format.

  Proxide returns a flat mask. We need the standard 37-atom format.
  """
  mask_37 = np.zeros((n_residues, 37), dtype=np.float32)

  # Standard backbone atoms: N=0, CA=1, C=2, O=4, CB=3
  backbone_indices = [
    atom_order["N"],
    atom_order["CA"],
    atom_order["C"],
    atom_order["CB"],
    atom_order["O"],
  ]

  for idx in backbone_indices:
    mask_37[:, idx] = 1.0

  return mask_37


def parse_with_proxide(
  path: str | Path,
  *,
  compute_physics: bool = True,
  compute_rbf: bool = False,
  rbf_num_neighbors: int = 30,
) -> Generator[ProteinTuple, None, None]:
  """Parse a structure file using proxide.

  Args:
      path: Path to structure file (PDB, PQR, CIF, etc.)
      compute_physics: Whether to compute electrostatic/vdW features
      compute_rbf: Whether to compute RBF features
      rbf_num_neighbors: Number of neighbors for RBF computation

  Yields:
      ProteinTuple with structure data and optional physics features.

  Raises:
      ImportError: If proxide is not installed.
  """
  if not PROXIDE_AVAILABLE:
    msg = "proxide is not installed. Install with: pip install proxide"
    raise ImportError(msg)

  path = Path(path)
  logger.info("Parsing %s with proxide", path)

  # Configure output spec
  spec = OutputSpec(
    compute_electrostatics=compute_physics,
    compute_vdw=compute_physics,
    compute_rbf=compute_rbf,
    rbf_num_neighbors=rbf_num_neighbors,
  )

  # Parse base structure
  result = parse_structure(str(path), spec)

  # Handle PQR specially for charges/radii
  if path.suffix.lower() == ".pqr":
    pqr_data = parse_pqr(str(path))
    charges = pqr_data.get("charges")
    radii = pqr_data.get("radii")
  else:
    charges = None
    radii = None

  # Extract dimensions
  n_residues = len(result["aatype"])

  # Get coordinates in proper format
  coords_flat = result["coordinates"]
  n_atoms = len(coords_flat) // 3
  coords_3d = coords_flat.reshape(n_atoms, 3)

  # Create backbone coordinates (n_residues, 5, 3)
  backbone_coords = _reshape_coordinates(coords_flat, n_residues)

  # Create atom mask
  atom_mask = _atom_mask_to_37(result["atom_mask"], n_residues)

  # Initialize ProteinTuple first so we can use it for JAX feature computation if needed
  # (though ProteinTuple is a NamedTuple, proxide's features.py might expect its own Protein dataclass,
  # but they are structurally similar enough for dot access usually)
  protein_tuple = ProteinTuple(
    coordinates=backbone_coords.astype(np.float32),
    aatype=result["aatype"].astype(np.int8),
    atom_mask=atom_mask,
    residue_index=result["residue_index"].astype(np.int32),
    chain_index=result["chain_index"].astype(np.int32),
    full_coordinates=coords_3d.astype(np.float32),
    dihedrals=None,
    source=str(path),
    mapping=None,
    charges=charges,
    radii=radii,
    sigmas=None,
    epsilons=None,
    estat_backbone_mask=None,
    estat_resid=None,
    estat_chain_index=None,
    physics_features=None,
  )

  # Get physics features
  feats = []

  # 1. Van der Waals
  vdw_feat = result.get("vdw_features")
  if vdw_feat is None and compute_physics:
    # If not precomputed by parse_structure (e.g. missing params in rust)
    # but we have coordinates, we could compute it but it needs sigmas/epsilons.
    # For now assume if Result has it, use it.
    pass

  if vdw_feat is not None:
    feats.append(vdw_feat)

  # 2. Electrostatics
  estat_feat = result.get("electrostatic_features")
  if estat_feat is None and compute_physics and charges is not None:
    # Manually compute if we have charges (common for PQR parsed with parse_pqr)
    try:
      # Use the proxy function which handles the conversion to proxide.Protein internally or via dot-access
      estat_feat = compute_electrostatic_node_features(protein_tuple)
      estat_feat = np.array(estat_feat)
    except Exception:
      logger.exception("Failed to compute manual electrostatic features for %s", path)

  if estat_feat is not None:
    feats.append(estat_feat)

  if feats:
    merged_physics = np.concatenate(feats, axis=-1)
    protein_tuple = protein_tuple._replace(physics_features=merged_physics)

  yield protein_tuple


def is_proxide_available() -> bool:
  """Check if proxide is available."""
  return PROXIDE_AVAILABLE
