"""Force projection onto backbone geometry for SE(3)-equivariant features."""

from __future__ import annotations

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
  # frames shape: (4, n_residues, 3)
  # aggregated_forces shape: (n_residues, 3)

  def project_residue(agg_force: jax.Array, residue_frames: jax.Array) -> jax.Array:
    """Project aggregated force for one residue onto its 4 frame vectors.

    Args:
        agg_force: Aggregated force for one residue. Shape: (3,).
        residue_frames: 4 frame vectors for one residue. Shape: (4, 3).

    Returns:
        4 projected forces. Shape: (4,).

    """
    return jnp.sum(agg_force[jnp.newaxis, :] * residue_frames, axis=-1)

  # Vmap over residues: frames has shape (4, n_residues, 3)
  # We want to vmap over axis 1 of frames to get each residue's 4 frames
  forces = jax.vmap(project_residue, in_axes=(0, 1))(aggregated_forces, frames)
  # forces shape after vmap: (n_residues, 4)
  magnitude = jnp.linalg.norm(aggregated_forces, axis=-1, keepdims=True)
  # magnitude shape: (n_residues, 1)

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
  # frames shape: (4, n_residues, 3)

  def project_per_residue(force_vector: jax.Array, residue_frames: jax.Array) -> jax.Array:
    """Project each backbone atom's force onto all frame vectors.

    Args:
        force_vector: Forces at 5 atoms. Shape: (5, 3).
        residue_frames: 4 frame vectors. Shape: (4, 3).

    Returns:
        Projections of 5 atoms onto 4 frames. Shape: (4, 5).

    """
    # For each atom, project onto all 4 frames
    # force_vector: (5, 3), residue_frames: (4, 3)
    # Result: (4, 5) - each row is projections of all atoms onto one frame
    return jnp.dot(residue_frames, force_vector.T)  # (4, 3) @ (3, 5) -> (4, 5)

  # Vmap over residues: frames has shape (4, n_residues, 3)
  forces = jax.vmap(project_per_residue, in_axes=(0, 1))(force_vectors, frames)
  # forces shape after vmap: (n_residues, 4, 5)

  magnitude = jnp.linalg.norm(force_vectors, axis=-1)  # (n_residues, 5)

  # Stack projections with magnitude for each atom
  # Result shape: (n_residues, 4, 5) + (n_residues, 5) -> (n_residues, 5, 5)
  result = jnp.stack(
    [forces[:, 0, :], forces[:, 1, :], forces[:, 2, :], forces[:, 3, :], magnitude], axis=-1
  )
  return result.reshape(force_vectors.shape[0], -1)
