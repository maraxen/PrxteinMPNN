"""Electrostatic force calculations using Coulomb's law."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax_md import space

from prxteinmpnn.physics.constants import COULOMB_CONSTANT, MIN_DISTANCE


def compute_pairwise_displacements(
  positions1: jax.Array,
  positions2: jax.Array,
) -> tuple[jax.Array, jax.Array]:
  """Compute pairwise displacements and distances between two sets of positions.

  Uses jax_md for efficient vectorized computation.

  Args:
      positions1: Source positions (e.g., backbone atoms), shape (n, 3)
      positions2: Target positions (e.g., all atoms including sidechains), shape (m, 3)

  Returns:
      Tuple of (displacements, distances):
      - displacements: (n, m, 3) - vector from positions1[i] to positions2[j]
      - distances: (n, m) - Euclidean distance

  Example:
      >>> pos1 = jnp.array([[0., 0., 0.], [5., 0., 0.]])
      >>> pos2 = jnp.array([[0., 0., 0.], [3., 4., 0.]])
      >>> displacements, distances = compute_pairwise_displacements(pos1, pos2)
      >>> print(distances[0, 1])  # Distance from pos1[0] to pos2[1]
      5.0

  """
  displacement_fn, _ = space.free()

  def displacement_to_all_j(pos_i: jax.Array) -> jax.Array:
    """Compute displacement from pos_i to all positions in positions2."""
    return jax.vmap(lambda pos_j: displacement_fn(pos_i, pos_j))(positions2)

  displacements = jax.vmap(displacement_to_all_j)(positions1)
  distances = space.distance(displacements)

  return displacements, distances


@partial(jax.jit, static_argnames=("exclude_self",))
def compute_coulomb_forces(
  displacements: jax.Array,
  distances: jax.Array,
  charges: jax.Array,
  coulomb_constant: float = COULOMB_CONSTANT,
  min_distance: float = MIN_DISTANCE,
  *,
  exclude_self: bool = True,
) -> jax.Array:
  """Compute Coulomb force vectors at target positions from source charges.

  Implements Coulomb's law in vacuum:
      F_ij = (k * q_j / r_ij²) * (r_ij / |r_ij|)

  Args:
      displacements: Displacement vectors from targets to sources, shape (n, m, 3)
      distances: Distances between targets and sources, shape (n, m)
      charges: Partial charges at source positions (in elementary charge units), shape (m,)
      coulomb_constant: Coulomb constant (default: 332.0636 kcal/mol·Å·e⁻²)
      min_distance: Minimum distance for numerical stability (default: 1e-7 Å)
      exclude_self: If True, zero out diagonal (self-interaction) terms

  Returns:
      Force vectors at each target position, shape (n, 3)
      Forces are in units of kcal/mol/Å

  Example:
      >>> # Two point charges: +1e at origin, -1e at (5, 0, 0)
      >>> positions = jnp.array([[0., 0., 0.], [5., 0., 0.]])
      >>> charges = jnp.array([1.0, -1.0])
      >>> displacements, distances = compute_pairwise_displacements(positions, positions)
      >>> forces = compute_coulomb_forces(displacements, distances, charges)
      >>> # Force at position 0 should point toward position 1 (attraction)

  """
  distances_safe = jnp.maximum(distances, min_distance)

  if exclude_self and distances.shape[0] == distances.shape[1]:
    mask = jnp.eye(distances.shape[0], dtype=bool)
    distances_safe = jnp.where(mask, jnp.inf, distances_safe)

  force_magnitudes = coulomb_constant * charges[None, :] / (distances_safe**2)
  unit_displacements = displacements / distances_safe[..., None]
  force_vectors = force_magnitudes[..., None] * unit_displacements
  return jnp.sum(force_vectors, axis=1)


def compute_coulomb_forces_at_backbone(
  backbone_positions: jax.Array,
  all_atom_positions: jax.Array,
  all_atom_charges: jax.Array,
  coulomb_constant: float = COULOMB_CONSTANT,
) -> jax.Array:
  """Compute Coulomb forces at all five backbone atoms from all charges.

  Computes electrostatic forces at N, CA, C, O, and CB (reconstructed) atoms
  for each residue. This matches PrxteinMPNN's representation which uses these
  5 atoms, where CB indicates the sidechain direction.

  Args:
      backbone_positions: Backbone atom positions [N, CA, C, O, CB] per residue,
        shape (n_residues, 5, 3)
      all_atom_positions: All atom positions (including sidechains),
        shape (n_atoms, 3)
      all_atom_charges: Partial charges for all atoms,
        shape (n_atoms,)
      coulomb_constant: Coulomb constant

  Returns:
      Force vectors at backbone atoms, shape (n_residues, 5, 3)
      Forces are in kcal/mol/Å

  Example:
      >>> bb_pos = jnp.ones((10, 5, 3))  # 10 residues
      >>> all_pos = jnp.ones((150, 3))   # 150 total atoms
      >>> charges = jnp.ones(150) * 0.1
      >>> forces = compute_coulomb_forces_at_backbone(bb_pos, all_pos, charges)
      >>> print(forces.shape)
      (10, 5, 3)  # Force vectors at N, CA, C, O, CB for each residue

  """
  n_residues = backbone_positions.shape[0]
  backbone_flat = backbone_positions.reshape(-1, 3)
  displacements, distances = compute_pairwise_displacements(
    backbone_flat,
    all_atom_positions,
  )
  forces_flat = compute_coulomb_forces(
    displacements,
    distances,
    all_atom_charges,
    coulomb_constant,
  )
  return forces_flat.reshape(n_residues, 5, 3)
