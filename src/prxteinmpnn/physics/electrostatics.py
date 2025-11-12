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

  Args:
      positions1: Target positions, shape (n, 3)
      positions2: Source positions, shape (m, 3)

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
  displacements = jax.vmap(
    lambda r1: jax.vmap(lambda r2: displacement_fn(r2, r1))(positions2),
  )(positions1)
  distances = space.distance(displacements)
  return displacements, distances


def compute_coulomb_potential(
  target_positions: jax.Array,
  source_positions: jax.Array,
  target_charges: jax.Array,
  source_charges: jax.Array,
  coulomb_constant: float = COULOMB_CONSTANT,
  min_distance: float = MIN_DISTANCE,
  *,
  exclude_self: bool = True,
) -> jax.Array:
  """Compute total Coulomb potential energy: U = sum_i sum_j (k * q_i * q_j / r_ij).

  Args:
      target_positions: Target positions, shape (n, 3)
      source_positions: Source positions, shape (m, 3)
      target_charges: Charges at target positions, shape (n,)
      source_charges: Charges at source positions, shape (m,)
      coulomb_constant: Coulomb constant (default: 332.0636 kcal/mol·e⁻²)
      min_distance: Minimum distance for numerical stability
      exclude_self: If True, exclude self-interactions

  Returns:
      Scalar potential energy in kcal/mol

  """
  _, distances = compute_pairwise_displacements(target_positions, source_positions)
  distances_safe = jnp.maximum(distances, min_distance)

  potentials = coulomb_constant * target_charges[:, None] * source_charges[None, :] / distances_safe

  if exclude_self:
    is_self_mask = distances < (min_distance / 10.0)
    potentials = jnp.where(is_self_mask, 0.0, potentials)

  return jnp.sum(potentials)


def compute_coulomb_forces_from_positions(
  target_positions: jax.Array,
  source_positions: jax.Array,
  target_charges: jax.Array,
  source_charges: jax.Array,
  coulomb_constant: float = COULOMB_CONSTANT,
  min_distance: float = MIN_DISTANCE,
  *,
  exclude_self: bool = True,
) -> jax.Array:
  """Compute Coulomb forces as F_i = -∇_i U via automatic differentiation.

  Args:
      target_positions: Target positions, shape (n, 3)
      source_positions: Source positions, shape (m, 3)
      target_charges: Charges at target positions, shape (n,)
      source_charges: Charges at source positions, shape (m,)
      coulomb_constant: Coulomb constant
      min_distance: Minimum distance for numerical stability
      exclude_self: If True, exclude self-interactions

  Returns:
      Force vectors at target positions, shape (n, 3) in kcal/mol/Å

  """
  grad_fn = jax.grad(
    lambda pos: compute_coulomb_potential(
      pos,
      source_positions,
      target_charges,
      source_charges,
      coulomb_constant,
      min_distance,
      exclude_self=exclude_self,
    ),
  )
  return -grad_fn(target_positions)


@partial(jax.jit, static_argnames=("exclude_self",))
def compute_coulomb_forces(
  displacements: jax.Array,
  distances: jax.Array,
  target_charges: jax.Array,
  source_charges: jax.Array,
  coulomb_constant: float = COULOMB_CONSTANT,
  min_distance: float = MIN_DISTANCE,
  *,
  exclude_self: bool = True,
) -> jax.Array:
  """Compute Coulomb force vectors (manual calculation equivalent to -∇U).

  This function maintains backward compatibility with the existing API that takes
  precomputed displacements/distances. The calculation is mathematically equivalent
  to computing forces as the negative gradient of the potential energy.

  Args:
      displacements: Displacement vectors from targets to sources, shape (n, m, 3)
      distances: Distances between targets and sources, shape (n, m)
      target_charges: Charges at target positions, shape (n,)
      source_charges: Charges at source positions, shape (m,)
      coulomb_constant: Coulomb constant (default: 332.0636 kcal/mol·Å·e⁻²)
      min_distance: Minimum distance for numerical stability
      exclude_self: If True, exclude self-interactions

  Returns:
      Force vectors at each target position, shape (n, 3) in kcal/mol/Å

  Example:
      >>> positions = jnp.array([[0., 0., 0.], [5., 0., 0.]])
      >>> charges = jnp.array([1.0, 1.0])
      >>> displacements, distances = compute_pairwise_displacements(positions, positions)
      >>> forces = compute_coulomb_forces(displacements, distances, charges, charges)
      >>> print(forces[0, 0] < 0)
      True

  """
  distances_safe = jnp.maximum(distances, min_distance)

  force_magnitudes = (
    coulomb_constant * target_charges[:, None] * source_charges[None, :] / (distances_safe**2)
  )
  unit_force_direction = -displacements / distances_safe[..., None]
  force_vectors = force_magnitudes[..., None] * unit_force_direction

  if exclude_self:
    is_self_mask = distances < (min_distance / 10.0)
    force_vectors = jnp.where(is_self_mask[..., None], 0.0, force_vectors)

  return jnp.sum(force_vectors, axis=1)


def compute_coulomb_forces_at_backbone(
  backbone_positions: jax.Array,
  all_atom_positions: jax.Array,
  backbone_charges: jax.Array,
  all_atom_charges: jax.Array,
  coulomb_constant: float = COULOMB_CONSTANT,
) -> jax.Array:
  """Compute Coulomb forces at all five backbone atoms from all charges.

  Computes electrostatic forces at N, CA, C, O, and CB atoms for each residue.
  This matches PrxteinMPNN's representation which uses these 5 atoms, where CB
  indicates the sidechain direction.

  For Glycine residues (which lack CB), the CB position is typically set to the
  hydrogen position, and the charge at that position should be the hydrogen charge.
  Self-interactions are automatically excluded (force from an atom on itself is zero).

  Args:
      backbone_positions: Backbone atom positions [N, CA, C, O, CB/H] per residue,
        shape (n_residues, 5, 3)
      all_atom_positions: All atom positions (including sidechains),
        shape (n_atoms, 3)
      backbone_charges: Partial charges at backbone positions [N, CA, C, O, CB/H],
        shape (n_residues, 5) - use H charge for Glycine CB position
      all_atom_charges: Partial charges for all atoms,
        shape (n_atoms,)
      coulomb_constant: Coulomb constant

  Returns:
      Force vectors at backbone atoms, shape (n_residues, 5, 3)
      Forces are in kcal/mol/Å

  Example:
      >>> bb_pos = jnp.ones((10, 5, 3))  # 10 residues
      >>> bb_charges = jnp.ones((10, 5)) * 0.2  # Backbone charges
      >>> all_pos = jnp.ones((150, 3))   # 150 total atoms
      >>> all_charges = jnp.ones(150) * 0.1
      >>> forces = compute_coulomb_forces_at_backbone(
      ...     bb_pos, all_pos, bb_charges, all_charges
      ... )
      >>> print(forces.shape)
      (10, 5, 3)  # Force vectors at N, CA, C, O, CB/H for each residue

  """
  n_residues = backbone_positions.shape[0]

  # Flatten to (n_residues * 5, 3) for vectorized computation
  backbone_flat = backbone_positions.reshape(-1, 3)
  backbone_charges_flat = backbone_charges.reshape(-1)

  displacements, distances = compute_pairwise_displacements(
    backbone_flat,
    all_atom_positions,
  )

  forces_flat = compute_coulomb_forces(
    displacements,
    distances,
    backbone_charges_flat,
    all_atom_charges,
    coulomb_constant,
    exclude_self=True,
  )

  return forces_flat.reshape(n_residues, 5, 3)
