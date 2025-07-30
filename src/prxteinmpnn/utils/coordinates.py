"""Utility functions for manipulating atomic coordinates.

prxteinmpnn.utils.coordinates
"""

from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from prxteinmpnn.utils.types import (
  AtomicCoordinate,
  BackboneCoordinates,
  StructureAtomicCoordinates,
)


@partial(jax.jit, static_argnames=("augment_eps",))
def apply_noise_to_coordinates(
  coordinates: StructureAtomicCoordinates,
  key: PRNGKeyArray,
  augment_eps: float = 0.0,
) -> StructureAtomicCoordinates:
  """Add Gaussian noise to atomic coordinates."""
  if augment_eps > 0:
    noise = augment_eps * jax.random.normal(key, coordinates.shape)
    return coordinates + noise
  return coordinates


@jax.jit
def compute_backbone_coordinates(
  coordinates: StructureAtomicCoordinates,
) -> BackboneCoordinates:
  """Compute backbone coordinates from atomic coordinates."""
  nitrogen, alpha_carbon, carbon, oxygen = (
    coordinates[:, 0, :],
    coordinates[:, 1, :],
    coordinates[:, 2, :],
    coordinates[:, 3, :],
  )
  alpha_to_nitrogen = alpha_carbon - nitrogen
  carbon_to_alpha = carbon - alpha_carbon
  beta_carbon = compute_c_beta(
    alpha_to_nitrogen,
    carbon_to_alpha,
    alpha_carbon,
  )
  return jnp.stack(
    [nitrogen, alpha_carbon, carbon, oxygen, beta_carbon],
    axis=1,
  )


@jax.jit
def compute_c_beta(
  alpha_to_nitrogen: AtomicCoordinate,
  carbon_to_alpha: AtomicCoordinate,
  alpha_carbon: AtomicCoordinate,
) -> AtomicCoordinate:
  """Compute C-beta coordinates.

  Uses a linear combination of the bond vectors to estimate C-beta.

  Coefficients are derived from empirical data and are used to ensure
  that the C-beta coordinates are consistent with the geometry of the protein backbone.

  Args:
    alpha_to_nitrogen: Bond vector from nitrogen to alpha carbon.
    carbon_to_alpha: Bond vector from alpha carbon to carbon.
    alpha_carbon: Coordinates of the alpha carbon atom.

  Returns:
    C-beta coordinates as an AtomicCoordinate.

  """
  f1, f2, f3 = -0.58273431, 0.56802827, -0.54067466
  term1 = f1 * jnp.cross(alpha_to_nitrogen, carbon_to_alpha)
  term2 = f2 * alpha_to_nitrogen
  term3 = f3 * carbon_to_alpha
  return term1 + term2 + term3 + alpha_carbon


@jax.jit
def compute_backbone_distance(backbone_coordinates: BackboneCoordinates) -> BackboneCoordinates:
  """Compute pairwise distances between backbone atoms.

  Calculate the Euclidean distance between all pairs of backbone atom coordinates based on alpha
  carbon positions.

  Assumes backbone_coordinates is a 3D array of shape (N, 5, 3), where N is the number of atoms,
  5 is the number of backbone atoms (N, CA, C, O, N), and 3 is the spatial dimension (x, y, z).

  Args:
    backbone_coordinates: A 3D array of shape (N, 5, 3) representing the coordinates of backbone
    atoms.

  Returns:
    A 2D array of shape (N, N) containing the pairwise distances between backbone atoms.

  """
  return jnp.sqrt(
    1e-6
    + jnp.sum(
      (backbone_coordinates[:, None, :, :] - backbone_coordinates[None, :, :, :]) ** 2,
      axis=-1,
    ),
  )


@jax.jit
def extend_coordinate(
  atom_a: AtomicCoordinate,
  atom_b: AtomicCoordinate,
  atom_c: AtomicCoordinate,
  bond_length: float,
  bond_angle: float,
  dihedral_angle: float,
) -> AtomicCoordinate:
  """Compute the position of a fourth atom (D) given three atoms (A, B, C) and internal coordinates.

  Given coordinates for atoms A, B, and C, and the desired bond length, bond angle, and dihedral
  angle, compute the coordinates of atom D such that:
    - |C-D| = bond_length
    - angle(B, C, D) = bond_angle
    - dihedral(A, B, C, D) = dihedral_angle

  Args:
    atom_a: Coordinates of atom A, shape (3,).
    atom_b: Coordinates of atom B, shape (3,).
    atom_c: Coordinates of atom C, shape (3,).
    bond_length: Desired bond length between C and D.
    bond_angle: Desired bond angle (in radians) at atom C.
    dihedral_angle: Desired dihedral angle (in radians) for atoms A-B-C-D.

  Returns:
    Coordinates of atom D, shape (3,).

  Example:
    >>> d = extend_coordinate(a, b, c, 1.5, 2.0, 3.14)

  """

  def normalize(vec: AtomicCoordinate) -> AtomicCoordinate:
    return vec / jnp.linalg.norm(vec)

  bc = normalize(atom_b - atom_c)
  normal = normalize(jnp.cross(atom_b - atom_a, bc))
  term1 = bond_length * jnp.cos(bond_angle) * bc
  term2 = bond_length * jnp.sin(bond_angle) * jnp.cos(dihedral_angle) * jnp.cross(normal, bc)
  term3 = bond_length * jnp.sin(bond_angle) * jnp.sin(dihedral_angle) * -normal
  return atom_c + term1 + term2 + term3


@jax.jit
def compute_cb_precise(
  n_coord: AtomicCoordinate,
  ca_coord: AtomicCoordinate,
  c_coord: AtomicCoordinate,
) -> AtomicCoordinate:
  """Compute the C-beta atom position from backbone N, CA, and C coordinates.

  Does so precisely using trigonometric relationships based on the backbone geometry.

  Specifically, the position of the C-beta atom is determined by:

  - The bond length between the alpha carbon and the C-beta atom.
  - The bond angle between the nitrogen, alpha carbon, and C-beta atoms.
  - The dihedral angle involving the nitrogen, alpha carbon, and C-beta atoms.


  Unlike the compute_c_beta function, this function does not use a linear combination of bond
  vectors with approximate fixed coefficients. This is more accurate and flexible for different
  configurations of the protein backbone, but more computationally intensive.

  It is used in preparation of the atomic coordinates for the model input.
  It is not used in the model itself, but rather in the preprocessing of the input data
  to ensure that the C-beta atom is correctly placed based on the backbone structure.

  Uses standard geometry for C-beta placement:
    - N-CA-CB bond length: 1.522 Å
    - N-CA-CB bond angle: 1.927 radians
    - C-N-CA-CB dihedral angle: -2.143 radians

  Args:
    n_coord: Coordinates of the N atom, shape (3,).
    ca_coord: Coordinates of the CA atom, shape (3,).
    c_coord: Coordinates of the C atom, shape (3,).

  Returns:
    Coordinates of the C-beta atom, shape (3,).

  Example:
    >>> cb = compute_cb_precise(n, ca, c)

  """
  return extend_coordinate(
    c_coord,
    n_coord,
    ca_coord,
    bond_length=1.522,
    bond_angle=1.927,
    dihedral_angle=-2.143,
  )
