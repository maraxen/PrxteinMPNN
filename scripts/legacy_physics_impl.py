from __future__ import annotations
COULOMB_CONSTANT = 332.0636
MIN_DISTANCE = 1e-07

"""Force projection onto backbone geometry for SE(3)-equivariant features."""


import jax
import jax.numpy as jnp

from prxteinmpnn.utils.atom_ordering import C_INDEX, CA_INDEX, CB_PDB_INDEX, N_INDEX


def compute_backbone_frame(
  backbone_positions: jax.Array,
) -> jax.Array:
  """Compute local backbone coordinate frame for each residue.

  Defines four important unit vectors per residue:
  - forward: CA → C (along backbone toward C-terminus)
  - backward: CA → N (toward N-terminus)
  - sidechain: CA → CB (sidechain direction)
  - normal: perpendicular to N-CA-C plane (via cross product)

  All vectors are normalized to unit length.

  Args:
      backbone_positions: Positions of [N, CA, C, O, CB] atoms per residue.
        Shape: (n_residues, 5, 3).

  Returns:
      Stack of (forward_hat, backward_hat, sidechain_hat, normal_hat).
      Shape: (4, n_residues, 3).

  Example:
      >>> positions = jnp.array([
      ...     [[0., 0., 0.], [1., 0., 0.], [2., 0., 0.],
      ...      [2., 1., 0.], [1., 1., 0.]]  # One residue
      ... ])
      >>> frame = compute_backbone_frame(positions)
      >>> print(frame.shape)
      (4, 1, 3)

  """

  def get_atom(index: int) -> jax.Array:
    return backbone_positions[:, index, :]

  def get_bond_vector(index_from: int, index_to: int) -> jax.Array:
    return get_atom(index_to) - get_atom(index_from)

  def normalize_bond_vector(vector: jax.Array) -> jax.Array:
    norm = jnp.linalg.norm(vector, axis=-1, keepdims=True)
    return vector / norm

  def get_normal_plane_vector(
    forward: jax.Array,
    backward: jax.Array,
  ) -> jax.Array:
    cross_product = jnp.cross(forward, backward, axis=-1)
    normal_norm = jnp.linalg.norm(cross_product, axis=-1, keepdims=True)
    forward_norm = jnp.linalg.norm(forward, axis=-1, keepdims=True)
    backward_norm = jnp.linalg.norm(backward, axis=-1, keepdims=True)
    epsilon = jnp.maximum(forward_norm, backward_norm) * 1e-7 + 1e-8
    return cross_product / jnp.maximum(normal_norm, epsilon)

  forward, backward = get_bond_vector(CA_INDEX, C_INDEX), get_bond_vector(CA_INDEX, N_INDEX)
  normal = get_normal_plane_vector(forward, backward)
  forward, backward, sidechain = (
    normalize_bond_vector(forward),
    normalize_bond_vector(backward),
    normalize_bond_vector(get_bond_vector(CA_INDEX, CB_PDB_INDEX)),
  )
  return jnp.stack([forward, backward, sidechain, normal], axis=0)


def project_forces_onto_backbone(
  force_vectors: jax.Array,
  backbone_positions: jax.Array,
  aggregation: str = "mean",
) -> jax.Array:
  """Project force vectors onto local backbone geometry.

  Computes five SE(3)-equivariant scalar features per residue by:
  1. Aggregating forces across the 5 backbone atoms (N, CA, C, O, CB)
  2. Projecting onto the local backbone frame

  Features per residue:
  1. f_forward: Force component along CA→C bond
  2. f_backward: Force component along CA→N bond
  3. f_sidechain: Force component along CA→CB (sidechain direction)
  4. f_out_of_plane: Force component perpendicular to N-CA-C plane
  5. f_magnitude: Total force magnitude

  All features are rotation-invariant (scalars that don't change under rotation).

  Args:
      force_vectors: Force vectors at all 5 backbone atoms.
        Shape: (n_residues, 5, 3).
      backbone_positions: Backbone atom positions.
        Shape: (n_residues, 5, 3).
      aggregation: How to aggregate forces ("mean" or "sum").

  Returns:
      Projected features. Shape: (n_residues, 5).
      Features: [f_forward, f_backward, f_sidechain, f_oop, f_mag].

  Raises:
      ValueError: If aggregation method is not "mean" or "sum".

  Example:
      >>> forces = jnp.ones((10, 5, 3))
      >>> positions = jnp.ones((10, 5, 3))
      >>> features = project_forces_onto_backbone(forces, positions)
      >>> print(features.shape)
      (10, 5)

  """
  if aggregation == "mean":
    aggregated_forces = jnp.mean(force_vectors, axis=1)
  elif aggregation == "sum":
    aggregated_forces = jnp.sum(force_vectors, axis=1)
  else:
    msg = f"Unknown aggregation method: {aggregation}"
    raise ValueError(msg)

  frames = compute_backbone_frame(
    backbone_positions,
  )

  def project_residue(agg_force: jax.Array, residue_frames: jax.Array) -> jax.Array:
    """Project aggregated force for one residue onto its 4 frame vectors.

    Args:
        agg_force: Aggregated force for one residue. Shape: (3,).
        residue_frames: 4 frame vectors for one residue. Shape: (4, 3).

    Returns:
        4 projected forces. Shape: (4,).

    """
    return jnp.sum(agg_force[jnp.newaxis, :] * residue_frames, axis=-1)

  forces = jax.vmap(project_residue, in_axes=(0, 1))(aggregated_forces, frames)
  magnitude = jnp.linalg.norm(aggregated_forces, axis=-1, keepdims=True)
  return jnp.concatenate([forces, magnitude], axis=-1)


def project_forces_onto_backbone_per_atom(
  force_vectors: jax.Array,
  backbone_positions: jax.Array,
) -> jax.Array:
  """Project forces at each backbone atom onto local frame (alternative approach).

  Projects forces at N, CA, C, O, CB separately onto the backbone frame.
  Produces 25 features per residue (5 projections x 5 atoms).

  This provides the most detailed information but highest dimensionality.
  Use this if you want to preserve per-atom force information.

  Args:
      force_vectors: Force vectors at all 5 backbone atoms.
        Shape: (n_residues, 5, 3).
      backbone_positions: Backbone atom positions.
        Shape: (n_residues, 5, 3).

  Returns:
      Projected features. Shape: (n_residues, 25).
      Layout: [N_forward, N_backward, N_sidechain, N_oop, N_mag,
                CA_forward, CA_backward, CA_sidechain, CA_oop, CA_mag, ...].

  """
  frames = compute_backbone_frame(backbone_positions)

  def project_per_residue(force_vector: jax.Array, residue_frames: jax.Array) -> jax.Array:
    """Project each backbone atom's force onto all frame vectors.

    Args:
        force_vector: Forces at 5 atoms. Shape: (5, 3).
        residue_frames: 4 frame vectors. Shape: (4, 3).

    Returns:
        Projections of 5 atoms onto 4 frames. Shape: (4, 5).

    """
    return jnp.dot(residue_frames, force_vector.T)  # (4, 3) @ (3, 5) -> (4, 5)

  forces = jax.vmap(project_per_residue, in_axes=(0, 1))(
    force_vectors,
    frames,
  )  # (n_residues, 4, 5)
  magnitude = jnp.linalg.norm(force_vectors, axis=-1)  # (n_residues, 5)
  result = jnp.concatenate(
    [forces.squeeze(), magnitude],
    axis=0,
  )
  return result.reshape(force_vectors.shape[0], -1)
"""Electrostatic force calculations using Coulomb's law."""


from functools import partial

import jax
import jax.numpy as jnp
from jax_md import space



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
  *,
  noise_scale: float | jax.Array = 0.0,
  key: jax.Array | None = None,
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
      noise_scale: Scale of Gaussian noise to add to forces (simulating thermal fluctuations).
      key: PRNG key for noise generation (required if noise_scale > 0).

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

  if noise_scale > 0.0:
    if key is None:
      msg = "Must provide key when noise_scale > 0"
      raise ValueError(msg)
    noise = jax.random.normal(key, forces_flat.shape)
    forces_flat = forces_flat + noise * noise_scale

  return forces_flat.reshape(n_residues, 5, 3)
"""Van der Waals (Lennard-Jones) interactions using jax_md."""


from functools import partial

import jax
import jax.numpy as jnp



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

  mask = jnp.eye(n, dtype=jnp.bool_)
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
      msg = "Must provide key when noise_scale > 0"
      raise ValueError(msg)
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
"""Compute physics-based node features for protein structures."""


from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from prxteinmpnn.utils.coordinates import compute_backbone_coordinates

if TYPE_CHECKING:
  from collections.abc import Sequence

  from prxteinmpnn.utils.data_structures import ProteinTuple


def _resolve_sigma(
  value: float | jax.Array | None,
  mode: str,
) -> float | jax.Array:
  """Resolve the noise standard deviation (sigma) from the input value and mode.

  Args:
      value: The noise parameter.
             - If mode='direct', this is the raw sigma.
             - If mode='thermal', this is T (Kelvin).
      mode: 'direct' or 'thermal'.

  Returns:
      The calculated standard deviation (sigma).

  """
  # Treat None as 0.0
  if value is None:
    return 0.0

  val = jnp.asarray(value)

  if mode == "direct":
    return val

  if mode == "thermal":
    # Physics Formula: sigma = sqrt(0.5 * R * T)
    # We clamp T to 0.0 to prevent NaN from negative sqrt
    thermal_energy = jnp.maximum(0.5 * BOLTZMANN_KCAL * val, 0.0)
    return jnp.sqrt(thermal_energy)

  msg = f"Unknown noise mode: {mode}"
  raise ValueError(msg)


def compute_electrostatic_node_features(
  protein: ProteinTuple,
  *,
  noise_scale: float | jax.Array | None = None,
  noise_mode: str = "direct",
  key: jax.Array | None = None,
) -> jax.Array:
  """Compute SE(3)-invariant electrostatic features for each residue.

  Computes electrostatic forces at backbone atoms (N, CA, C, O, CB/H) from all
  charged atoms, then projects these forces onto the backbone frame to create
  5D SE(3)-invariant features per residue.

  For Glycine residues (which lack CB), the 5th backbone position contains a
  hydrogen atom. The force calculation naturally handles this since the position
  and charge are already set correctly in the ProteinTuple.

  Self-interactions are automatically excluded in the force calculation - an atom
  does not exert force on itself.

  Args:
      protein: ProteinTuple containing structure and charge information.
        Must have:
        - coordinates: (n_residues, n_atom_types, 3) backbone atom positions
        - charges: (n_residues, n_atom_types) partial charges for all atoms
        - aatype: (n_residues,) amino acid type indices
      noise_scale: Scale of Gaussian noise to add to forces (default: 0.0).
      noise_mode: 'direct' or 'thermal'.
      key: PRNG key for noise generation (required if noise_scale > 0).

  Returns:
      Electrostatic features, shape (n_residues, 5):
      - f_forward: Force component along CA→next_N (forward chain direction)
      - f_backward: Force component along CA→prev_C (backward chain direction)
      - f_sidechain: Force component along CA→CB (sidechain direction)
      - f_oop: Out-of-plane force component (perpendicular to backbone plane)
      - f_mag: Total force magnitude

  Raises:
      ValueError: If protein.charges is None (PQR data required)

  Example:
      >>> protein = load_pqr_file("protein.pqr")
      >>> features = compute_electrostatic_node_features(protein)
      >>> print(features.shape)
      (n_residues, 5)

  """
  if protein.charges is None:
    msg = "ProteinTuple must have charges (PQR data) to compute electrostatic features"
    raise ValueError(msg)

  if protein.full_coordinates is None:
    msg = "ProteinTuple must have full_coordinates to compute electrostatic features"
    raise ValueError(msg)

  # compute_backbone_coordinates returns (n_residues, 5, 3) in this exact order
  backbone_positions = compute_backbone_coordinates(
    jnp.array(protein.coordinates),
  )
  all_positions = jnp.array(protein.full_coordinates)
  all_charges = jnp.array(protein.charges)

  n_residues = backbone_positions.shape[0]
  backbone_positions_flat = backbone_positions.reshape(-1, 3)  # (n_residues*5, 3)

  distances = jnp.linalg.norm(
    backbone_positions_flat[:, None, :] - all_positions[None, :, :],
    axis=-1,
  )

  closest_indices = jnp.argmin(distances, axis=1)
  backbone_charges_flat = all_charges[closest_indices]
  backbone_charges = backbone_charges_flat.reshape(n_residues, 5)

  sigma = _resolve_sigma(noise_scale, noise_mode)

  forces_at_backbone = compute_coulomb_forces_at_backbone(
    backbone_positions,
    all_positions,
    backbone_charges,
    all_charges,
    noise_scale=sigma,
    key=key,
  )

  return project_forces_onto_backbone(
    forces_at_backbone,
    backbone_positions,
  )


def compute_vdw_node_features(
  protein: ProteinTuple,
  *,
  noise_scale: float | jax.Array | None = None,
  noise_mode: str = "direct",
  key: jax.Array | None = None,
) -> jax.Array:
  """Compute SE(3)-invariant Van der Waals features for each residue.

  Computes LJ forces at backbone atoms from all atoms, then projects these
  forces onto the backbone frame.

  Args:
      protein: ProteinTuple containing structure and LJ parameters.
        Must have:
        - coordinates: (n_residues, n_atom_types, 3) backbone atom positions
        - full_coordinates: (n_atoms, 3) all atom positions
        - sigmas: (n_atoms,) LJ sigma parameters
        - epsilons: (n_atoms,) LJ epsilon parameters
      noise_scale: Scale of Gaussian noise to add to forces.
      noise_mode: 'direct' or 'thermal'.
      key: PRNG key for noise generation.

  Returns:
      vdW features, shape (n_residues, 5).

  """
  if protein.sigmas is None or protein.epsilons is None:
    msg = "ProteinTuple must have sigmas and epsilons to compute vdW features"
    raise ValueError(msg)

  if protein.full_coordinates is None:
    msg = "ProteinTuple must have full_coordinates to compute vdW features"
    raise ValueError(msg)

  backbone_positions = compute_backbone_coordinates(
    jnp.array(protein.coordinates),
  )
  all_positions = jnp.array(protein.full_coordinates)
  all_sigmas = jnp.array(protein.sigmas)
  all_epsilons = jnp.array(protein.epsilons)

  # Map sigmas/epsilons to backbone atoms by finding closest atom in 'full_coordinates'
  n_residues = backbone_positions.shape[0]
  backbone_positions_flat = backbone_positions.reshape(-1, 3)

  distances = jnp.linalg.norm(
    backbone_positions_flat[:, None, :] - all_positions[None, :, :],
    axis=-1,
  )
  closest_indices = jnp.argmin(distances, axis=1)

  backbone_sigmas_flat = all_sigmas[closest_indices]
  backbone_epsilons_flat = all_epsilons[closest_indices]

  backbone_sigmas = backbone_sigmas_flat.reshape(n_residues, 5)
  backbone_epsilons = backbone_epsilons_flat.reshape(n_residues, 5)

  sigma = _resolve_sigma(noise_scale, noise_mode)

  forces_at_backbone = compute_lj_forces_at_backbone(
    backbone_positions,
    all_positions,
    backbone_sigmas,
    backbone_epsilons,
    all_sigmas,
    all_epsilons,
    noise_scale=sigma,
    key=key,
  )

  return project_forces_onto_backbone(
    forces_at_backbone,
    backbone_positions,
  )


def compute_electrostatic_features_batch(
  proteins: Sequence[ProteinTuple],
  max_length: int | None = None,
  *,
  pad_value: float = 0.0,
) -> tuple[jax.Array, jax.Array]:
  """Compute electrostatic features for a batch of proteins with padding.

  Args:
      proteins: List of ProteinTuple instances
      max_length: Maximum sequence length for padding. If None, uses the
        longest sequence in the batch.
      pad_value: Value to use for padding (default: 0.0)

  Returns:
      features: (batch_size, max_length, 5) padded feature arrays
      mask: (batch_size, max_length) binary mask (1.0 for real residues, 0.0 for padding)

  Example:
      >>> proteins = [load_pqr_file(f"protein_{i}.pqr") for i in range(4)]
      >>> features, mask = compute_electrostatic_features_batch(proteins, max_length=128)
      >>> print(features.shape, mask.shape)
      (4, 128, 5) (4, 128)

  """
  if not proteins:
    msg = "Must provide at least one protein"
    raise ValueError(msg)

  features_list = [compute_electrostatic_node_features(p) for p in proteins]

  lengths = [f.shape[0] for f in features_list]
  if max_length is None:
    max_length = max(lengths)
  elif max_length < max(lengths):
    msg = f"max_length={max_length} is less than longest sequence ({max(lengths)})"
    raise ValueError(msg)

  batch_size = len(proteins)
  n_features = 5

  features_padded = jnp.full((batch_size, max_length, n_features), pad_value)
  mask = jnp.zeros((batch_size, max_length))

  for i, (features, length) in enumerate(zip(features_list, lengths, strict=False)):
    features_padded = features_padded.at[i, :length, :].set(features)
    mask = mask.at[i, :length].set(1.0)

  return features_padded, mask
