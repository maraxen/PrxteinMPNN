"""Van der Waals (Lennard-Jones) interactions using jax_md."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from prxteinmpnn.physics.constants import MIN_DISTANCE
from prxteinmpnn.physics.electrostatics import compute_pairwise_displacements


def combine_lj_parameters(
  sigma_i: jax.Array,
  sigma_j: jax.Array,
  epsilon_i: jax.Array,
  epsilon_j: jax.Array,
) -> tuple[jax.Array, jax.Array]:
  r"""Combine Lennard-Jones parameters using Lorentz-Berthelot rules.

  Lorentz-Berthelot combining rules:

  .. math::

    \\sigma_{ij} = \\frac{\\sigma_i + \\sigma_j}{2}

    \\varepsilon_{ij} = \\sqrt{\\varepsilon_i \\cdot \\varepsilon_j}

  These are the most common combining rules in molecular mechanics.

  Args:
      sigma_i (jax.Array): LJ sigma parameters for atoms i, shape (n,) or scalar.
      sigma_j (jax.Array): LJ sigma parameters for atoms j, shape (m,) or scalar.
      epsilon_i (jax.Array): LJ epsilon parameters for atoms i, shape (n,) or scalar.
      epsilon_j (jax.Array): LJ epsilon parameters for atoms j, shape (m,) or scalar.

  Returns:
      tuple[jax.Array, jax.Array]: Tuple of (sigma_ij, epsilon_ij) combined parameters,
      each of shape (n, m).

  Example:
      >>> sigma_i = jnp.array([3.5, 3.0])
      >>> sigma_j = jnp.array([3.0, 2.5])
      >>> epsilon_i = jnp.array([0.1, 0.15])
      >>> epsilon_j = jnp.array([0.2, 0.1])
      >>> sigma_ij, epsilon_ij = combine_lj_parameters(
      ...     sigma_i[:, None], sigma_j[None, :],
      ...     epsilon_i[:, None], epsilon_j[None, :]
      ... )
      >>> print(sigma_ij.shape)
      (2, 2)

  """
  sigma_ij = (sigma_i + sigma_j) / 2.0
  epsilon_ij = jnp.sqrt(epsilon_i * epsilon_j)
  return sigma_ij, epsilon_ij


def broadcast_and_combine_lj_parameters(
  sigma_i: jax.Array,
  sigma_j: jax.Array,
  epsilon_i: jax.Array,
  epsilon_j: jax.Array,
) -> tuple[jax.Array, jax.Array]:
  """Broadcast 1D parameters and combine for pairwise calculations.

  Convenience function that handles the common pattern of broadcasting
  1D parameter arrays to 2D and combining them using Lorentz-Berthelot rules.

  Args:
      sigma_i (jax.Array): LJ sigma parameters for target atoms, shape (n,).
      sigma_j (jax.Array): LJ sigma parameters for source atoms, shape (m,).
      epsilon_i (jax.Array): LJ epsilon parameters for target atoms, shape (n,).
      epsilon_j (jax.Array): LJ epsilon parameters for source atoms, shape (m,).

  Returns:
      tuple[jax.Array, jax.Array]: Combined (sigma_ij, epsilon_ij), both shape (n, m).

  Example:
      >>> sigma_i = jnp.array([3.5, 3.0])
      >>> sigma_j = jnp.array([3.0, 2.5, 2.0])
      >>> epsilon_i = jnp.array([0.1, 0.15])
      >>> epsilon_j = jnp.array([0.2, 0.1, 0.05])
      >>> sigma_ij, epsilon_ij = broadcast_and_combine_lj_parameters(
      ...     sigma_i, sigma_j, epsilon_i, epsilon_j
      ... )
      >>> print(sigma_ij.shape)
      (2, 3)

  """
  return combine_lj_parameters(
    sigma_i[:, None],  # (n, 1)
    sigma_j[None, :],  # (1, m)
    epsilon_i[:, None],  # (n, 1)
    epsilon_j[None, :],  # (1, m)
  )


def clamp_distances(
  distances: jax.Array,
  min_distance: float = MIN_DISTANCE,
) -> jax.Array:
  """Clamp distances to a minimum value for numerical stability.

  Args:
      distances (jax.Array): Pairwise distances between atoms, shape (n, m).
      min_distance (float): Minimum distance to clamp to.

  Returns:
      jax.Array: Clamped distances, shape (n, m).

  Example:
      >>> distances = jnp.array([[0.5, 1.0], [2.0, 0.1]])
      >>> clamped = clamp_distances(distances, min_distance=0.8)
      >>> print(clamped)
      [[0.8 1. ]
       [2.  0.8]]

  """
  return jnp.maximum(distances, min_distance)


def compute_inverse_powers(r: jax.Array, sigma: jax.Array) -> tuple[jax.Array, jax.Array]:
  r"""Compute $(\sigma/r)^6$ and $(\sigma/r)^{12}$.

  Computes the inverse power terms used in the Lennard-Jones potential.

  Args:
      r (jax.Array): Pairwise distances, shape (n, m).
      sigma (jax.Array): Combined LJ sigma parameters, shape (n, m).

  Returns:
      tuple[jax.Array, jax.Array]: Tuple of $(\sigma/r)^6$ and $(\sigma/r)^{12}$,
          each shape (n, m).

  Example:
      >>> r = jnp.array([[3.0, 4.0], [5.0, 6.0]])
      >>> sigma = jnp.ones((2, 2)) * 3.5
      >>> sigma_6, sigma_12 = compute_inverse_powers(r, sigma)
      >>> print(sigma_6.shape)
      (2, 2)

  """
  sigma_over_distance = sigma / r
  sigma_over_distance_6 = sigma_over_distance**6
  sigma_over_distance_12 = sigma_over_distance_6**2
  return sigma_over_distance_6, sigma_over_distance_12


def sigma_over_r(
  r: jax.Array,
  sigma: jax.Array,
  min_distance: float = MIN_DISTANCE,
) -> tuple[jax.Array, jax.Array, jax.Array]:
  r"""Compute $(\sigma/r)^6$ and $(\sigma/r)^{12}$ with clamping for numerical stability."""
  safe_distance = clamp_distances(r, min_distance)
  sigma_6, sigma_12 = compute_inverse_powers(safe_distance, sigma)
  return sigma_6, sigma_12, safe_distance


def apply_self_exclusion(
  values: jax.Array,
  *,
  exclude_self: bool,
) -> jax.Array:
  """Zero out diagonal elements for self-interaction exclusion.

  Args:
      values (jax.Array): Pairwise values (energies or forces), shape (n, m) or (n, m, d).
      exclude_self (bool): Whether to exclude self-interactions.

  Returns:
      jax.Array: Values with diagonal zeroed if exclude_self=True and matrix is square.

  Note:
      Only applies masking if the first two dimensions are equal (square matrix).

  """
  if not exclude_self:
    return values

  n, m = values.shape[0], values.shape[1]
  if n != m:
    return values

  mask = jnp.eye(n, dtype=bool)
  # Broadcast mask to match values shape
  while mask.ndim < values.ndim:
    mask = mask[..., None]

  return jnp.where(mask, 0.0, values)


def compute_lj_energy_pairwise(
  distances: jax.Array,
  sigma_ij: jax.Array,
  epsilon_ij: jax.Array,
  min_distance: float = MIN_DISTANCE,
) -> jax.Array:
  r"""Compute Lennard-Jones energy for pairwise interactions.

  Implements the 12-6 Lennard-Jones potential:

  .. math::

    E_{LJ}(r) = 4 \\varepsilon \\left[ \\left( \\frac{\\sigma}{r} \\right)^{12}
    - \\left( \\frac{\\sigma}{r} \\right)^6 \\right]

  The 12th power term represents short-range repulsion (Pauli exclusion),
  and the 6th power term represents long-range attraction (dispersion).

  Args:
      distances (jax.Array): Pairwise distances between atoms, shape (n, m).
      sigma_ij (jax.Array): Combined LJ sigma parameters, shape (n, m).
      epsilon_ij (jax.Array): Combined LJ epsilon parameters, shape (n, m).
      min_distance (float): Minimum distance for numerical stability.

  Returns:
      jax.Array: LJ energy for each pair, shape (n, m).
      Energies are in the same units as epsilon (typically kcal/mol).

  Example:
      >>> distances = jnp.array([[3.0, 4.0], [5.0, 6.0]])
      >>> sigma = jnp.ones((2, 2)) * 3.5
      >>> epsilon = jnp.ones((2, 2)) * 0.1
      >>> energy = compute_lj_energy_pairwise(distances, sigma, epsilon)
      >>> print(energy.shape)
      (2, 2)

  """
  sigma_6, sigma_12, _ = sigma_over_r(distances, sigma_ij, min_distance)
  return 4.0 * epsilon_ij * (sigma_12 - sigma_6)


def compute_lj_force_magnitude_pairwise(
  distances: jax.Array,
  sigma_ij: jax.Array,
  epsilon_ij: jax.Array,
  min_distance: float = MIN_DISTANCE,
) -> jax.Array:
  r"""Compute magnitude of Lennard-Jones force for pairwise interactions.

  The force is the negative derivative of the LJ potential:

  .. math::

    F_{LJ}(r) = -\\frac{dE}{dr}

  For the 12-6 Lennard-Jones potential:

  .. math::

    E_{LJ}(r) = 4 \\varepsilon \\left[ \\left( \\frac{\\sigma}{r}
    \\right)^{12} - \\left( \\frac{\\sigma}{r} \\right)^6 \\right]

  The force magnitude simplifies to:

  .. math::

    F_{LJ}(r) = \\frac{24 \\varepsilon}{r} \\left[ 2 \\left(
    \\frac{\\sigma}{r} \\right)^{12} - \\left( \\frac{\\sigma}{r}
    \\right)^6 \\right]

  Positive force = repulsive, negative force = attractive.

  Args:
      distances (jax.Array): Pairwise distances between atoms, shape (n, m).
      sigma_ij (jax.Array): Combined LJ \\sigma parameters, shape (n, m).
      epsilon_ij (jax.Array): Combined LJ \\varepsilon parameters, shape (n, m).
      min_distance (float): Minimum distance for numerical stability.

  Returns:
      jax.Array: Force magnitudes for each pair, shape (n, m).
      Forces are in units of \\varepsilon/distance (e.g., kcal/mol/Å).

  Example:
      >>> distances = jnp.array([[3.0, 4.0]])
      >>> sigma = jnp.ones((1, 2)) * 3.5
      >>> epsilon = jnp.ones((1, 2)) * 0.1
      >>> force_mag = compute_lj_force_magnitude_pairwise(distances, sigma, epsilon)
      >>> # At equilibrium (r ≈ 2^(1/6) * \\sigma), force should be near zero

  """
  sigma_6, sigma_12, safe_distance = sigma_over_r(distances, sigma_ij, min_distance)
  return 24.0 * epsilon_ij * (2.0 * sigma_12 - sigma_6) / safe_distance


@partial(jax.jit, static_argnames=("exclude_self",))
def compute_lj_forces(
  displacements: jax.Array,
  distances: jax.Array,
  sigma_i: jax.Array,
  sigma_j: jax.Array,
  epsilon_i: jax.Array,
  epsilon_j: jax.Array,
  min_distance: float = MIN_DISTANCE,
  *,
  exclude_self: bool = False,
) -> jax.Array:
  """Compute Lennard-Jones force vectors at target positions.

  Computes the total LJ force at each target position (i) due to all
  source atoms (j). Forces are vectors pointing in the direction of
  the displacement.

  Args:
      displacements (jax.Array): Displacement vectors from targets to sources,
          shape (n, m, 3).
      distances (jax.Array): Distances between targets and sources, shape (n, m).
      sigma_i (jax.Array): LJ sigma parameters for target atoms, shape (n,).
      sigma_j (jax.Array): LJ sigma parameters for source atoms, shape (m,).
      epsilon_i (jax.Array): LJ epsilon parameters for target atoms, shape (n,).
      epsilon_j (jax.Array): LJ epsilon parameters for source atoms, shape (m,).
      min_distance (float): Minimum distance for numerical stability.
      exclude_self (bool): If True, zero out diagonal (self-interaction) terms.

  Returns:
      jax.Array: Force vectors at each target position, shape (n, 3).
      Forces are in units of epsilon/distance (e.g., kcal/mol/Å).

  Example:
      >>> # Two atoms with LJ interaction
      >>> positions = jnp.array([[0., 0., 0.], [3.5, 0., 0.]])
      >>> from prxteinmpnn.physics.electrostatics import (
      ...     compute_pairwise_displacements
      ... )
      >>> displacements, distances = compute_pairwise_displacements(
      ...     positions, positions
      ... )
      >>> sigma = jnp.ones(2) * 3.5
      >>> epsilon = jnp.ones(2) * 0.1
      >>> forces = compute_lj_forces(
      ...     displacements, distances, sigma, sigma, epsilon, epsilon
      ... )
      >>> # Forces should be repulsive at short distance

  """
  sigma_ij, epsilon_ij = broadcast_and_combine_lj_parameters(
    sigma_i,
    sigma_j,
    epsilon_i,
    epsilon_j,
  )

  force_magnitudes = compute_lj_force_magnitude_pairwise(
    distances,
    sigma_ij,
    epsilon_ij,
    min_distance,
  )

  force_magnitudes = apply_self_exclusion(force_magnitudes, exclude_self=exclude_self)
  distances_safe = clamp_distances(distances, min_distance)
  unit_displacements = -displacements / distances_safe[..., None]
  force_vectors = force_magnitudes[..., None] * unit_displacements
  return jnp.sum(force_vectors, axis=1)


def compute_lj_energy_at_positions(
  _displacements: jax.Array,
  distances: jax.Array,
  sigma_i: jax.Array,
  sigma_j: jax.Array,
  epsilon_i: jax.Array,
  epsilon_j: jax.Array,
  min_distance: float = MIN_DISTANCE,
  *,
  exclude_self: bool = False,
) -> jax.Array:
  """Compute total Lennard-Jones energy at target positions.

  Sums the LJ energy contributions from all source atoms to each target atom.

  Args:
      displacements (jax.Array): Displacement vectors from targets to sources,
          shape (n, m, 3).
      distances (jax.Array): Distances between targets and sources, shape (n, m).
      sigma_i (jax.Array): LJ sigma parameters for target atoms, shape (n,).
      sigma_j (jax.Array): LJ sigma parameters for source atoms, shape (m,).
      epsilon_i (jax.Array): LJ epsilon parameters for target atoms, shape (n,).
      epsilon_j (jax.Array): LJ epsilon parameters for source atoms, shape (m,).
      min_distance (float): Minimum distance for numerical stability.
      exclude_self (bool): If True, zero out diagonal (self-interaction) terms.

  Returns:
      jax.Array: Total LJ energy at each target position, shape (n,).
      Energies are in the same units as epsilon (typically kcal/mol).

  """
  sigma_ij, epsilon_ij = broadcast_and_combine_lj_parameters(
    sigma_i,
    sigma_j,
    epsilon_i,
    epsilon_j,
  )
  energy_pairwise = compute_lj_energy_pairwise(
    distances,
    sigma_ij,
    epsilon_ij,
    min_distance,
  )
  energy_pairwise = apply_self_exclusion(energy_pairwise, exclude_self=exclude_self)
  return jnp.sum(energy_pairwise, axis=1)


def compute_lj_forces_at_backbone(
  backbone_positions: jax.Array,
  all_atom_positions: jax.Array,
  backbone_sigmas: jax.Array,
  backbone_epsilons: jax.Array,
  all_atom_sigmas: jax.Array,
  all_atom_epsilons: jax.Array,
  *,
  noise_scale: float | jax.Array = 0.0,
  key: jax.Array | None = None,
) -> jax.Array:
  """Compute Lennard-Jones forces at all five backbone atoms.

  Computes LJ forces at N, CA, C, O, and CB atoms for each residue.
  This matches the electrostatics interface for consistency.

  Args:
      backbone_positions (jax.Array): Backbone atom positions [N, CA, C, O, CB]
          per residue, shape (n_residues, 5, 3).
      all_atom_positions (jax.Array): All atom positions (including sidechains),
          shape (n_atoms, 3).
      backbone_sigmas (jax.Array): LJ sigma parameters for backbone atoms,
          shape (n_residues, 5).
      backbone_epsilons (jax.Array): LJ epsilon parameters for backbone atoms,
          shape (n_residues, 5).
      all_atom_sigmas (jax.Array): LJ sigma parameters for all atoms,
          shape (n_atoms,).
      all_atom_epsilons (jax.Array): LJ epsilon parameters for all atoms,
          shape (n_atoms,).
      noise_scale: Scale of Gaussian noise to add to forces.
      key: PRNG key for noise generation.

  Returns:
      jax.Array: Force vectors at backbone atoms, shape (n_residues, 5, 3).
      Forces are in units of epsilon/distance (e.g., kcal/mol/Å).

  Example:
      >>> bb_pos = jnp.ones((10, 5, 3))
      >>> all_pos = jnp.ones((150, 3))
      >>> bb_sigma = jnp.ones((10, 5)) * 3.5
      >>> bb_eps = jnp.ones((10, 5)) * 0.1
      >>> all_sigma = jnp.ones(150) * 3.0
      >>> all_eps = jnp.ones(150) * 0.15
      >>> forces = compute_lj_forces_at_backbone(
      ...     bb_pos, all_pos, bb_sigma, bb_eps, all_sigma, all_eps
      ... )
      >>> print(forces.shape)
      (10, 5, 3)

  """
  n_residues = backbone_positions.shape[0]

  backbone_flat = backbone_positions.reshape(-1, 3)
  backbone_sigmas_flat = backbone_sigmas.reshape(-1)
  backbone_epsilons_flat = backbone_epsilons.reshape(-1)

  displacements, distances = compute_pairwise_displacements(
    backbone_flat,
    all_atom_positions,
  )

  forces_flat = compute_lj_forces(
    displacements,
    distances,
    backbone_sigmas_flat,
    all_atom_sigmas,
    backbone_epsilons_flat,
    all_atom_epsilons,
  )

  if noise_scale > 0.0:
    if key is None:
      raise ValueError("Must provide key when noise_scale > 0")
    noise = jax.random.normal(key, forces_flat.shape)
    forces_flat = forces_flat + noise * noise_scale

  return forces_flat.reshape(n_residues, 5, 3)


def compute_lj_energy_at_backbone(
  backbone_positions: jax.Array,
  all_atom_positions: jax.Array,
  backbone_sigmas: jax.Array,
  backbone_epsilons: jax.Array,
  all_atom_sigmas: jax.Array,
  all_atom_epsilons: jax.Array,
) -> jax.Array:
  """Compute total Lennard-Jones energy at all five backbone atoms.

  Args:
      backbone_positions (jax.Array): Backbone atom positions [N, CA, C, O, CB]
          per residue, shape (n_residues, 5, 3).
      all_atom_positions (jax.Array): All atom positions, shape (n_atoms, 3).
      backbone_sigmas (jax.Array): LJ sigma parameters for backbone atoms,
          shape (n_residues, 5).
      backbone_epsilons (jax.Array): LJ epsilon parameters for backbone atoms,
          shape (n_residues, 5).
      all_atom_sigmas (jax.Array): LJ sigma parameters for all atoms,
          shape (n_atoms,).
      all_atom_epsilons (jax.Array): LJ epsilon parameters for all atoms,
          shape (n_atoms,).

  Returns:
      jax.Array: LJ energy at each backbone atom, shape (n_residues, 5).
      Energies are in the same units as epsilon (typically kcal/mol).

  """
  n_residues = backbone_positions.shape[0]
  backbone_flat = backbone_positions.reshape(-1, 3)
  backbone_sigmas_flat = backbone_sigmas.reshape(-1)
  backbone_epsilons_flat = backbone_epsilons.reshape(-1)

  displacements, distances = compute_pairwise_displacements(
    backbone_flat,
    all_atom_positions,
  )

  energy_flat = compute_lj_energy_at_positions(
    displacements,
    distances,
    backbone_sigmas_flat,
    all_atom_sigmas,
    backbone_epsilons_flat,
    all_atom_epsilons,
  )

  return energy_flat.reshape(n_residues, 5)
