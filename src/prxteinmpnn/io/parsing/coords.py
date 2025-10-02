"""Functions for parsing and manipulating atomic coordinates."""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def extend_coordinate(
  atom_a: np.ndarray,
  atom_b: np.ndarray,
  atom_c: np.ndarray,
  bond_length: float,
  bond_angle: float,
  dihedral_angle: float,
) -> np.ndarray:
  """Compute fourth atom (D) position given three atoms (A, B, C) and internal coordinates."""
  logger.info("Computing extended coordinate (D) from A, B, C.")

  def normalize(vec: np.ndarray) -> np.ndarray:
    return vec / np.linalg.norm(vec)

  bc = normalize(atom_b - atom_c)
  normal = normalize(np.cross(atom_b - atom_a, bc))
  term1 = bond_length * np.cos(bond_angle) * bc
  term2 = bond_length * np.sin(bond_angle) * np.cos(dihedral_angle) * np.cross(normal, bc)
  term3 = bond_length * np.sin(bond_angle) * np.sin(dihedral_angle) * -normal
  return atom_c + term1 + term2 + term3


def compute_cb_precise(
  n_coord: np.ndarray,
  ca_coord: np.ndarray,
  c_coord: np.ndarray,
) -> np.ndarray:
  """Compute the C-beta atom position from backbone N, CA, and C coordinates."""
  logger.info("Computing C-beta coordinate from N, CA, C backbone atoms.")
  return extend_coordinate(
    c_coord,
    n_coord,
    ca_coord,
    bond_length=1.522,
    bond_angle=1.927,
    dihedral_angle=-2.143,
  )


def process_coordinates(
  coordinates: np.ndarray,
  num_residues: int,
  atom_37_indices: np.ndarray,
  valid_atom_mask: np.ndarray,
) -> np.ndarray:
  """Process an AtomArray to create a Protein inputs."""
  logger.debug("Processing coordinates into (N_res, 37, 3) format.")
  coords_37 = np.zeros((num_residues, 37, 3), dtype=np.float32)
  coords_37[atom_37_indices] = np.asarray(
    coordinates,
  )[valid_atom_mask]
  return coords_37
