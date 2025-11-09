"""Feature extraction module for PrxteinMPNN.

This module contains the ProteinFeatures class that extracts and projects
features from raw protein coordinates.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from prxteinmpnn.utils.coordinates import (
  apply_noise_to_coordinates,
  compute_backbone_coordinates,
  compute_backbone_distance,
)
from prxteinmpnn.utils.graph import compute_neighbor_offsets
from prxteinmpnn.utils.radial_basis import compute_radial_basis
from prxteinmpnn.utils.types import (
  AlphaCarbonMask,
  BackboneNoise,
  ChainIndex,
  EdgeFeatures,
  NeighborIndices,
  ResidueIndex,
  StructureAtomicCoordinates,
)

# Type alias for PRNG keys
PRNGKeyArray = jax.Array

# Layer normalization
LayerNorm = eqx.nn.LayerNorm

# Feature extraction constants
MAXIMUM_RELATIVE_FEATURES = 32
POS_EMBED_DIM = 16
top_k = jax.jit(jax.lax.top_k, static_argnames=("k",))


class ProteinFeatures(eqx.Module):
  """Extracts and projects features from raw protein coordinates.

  This module encapsulates k-NN, RBF, positional encodings, and edge projections.
  Note: W_e projection is NOT here - it's in the main model (matches ColabDesign).
  """

  w_pos: eqx.nn.Linear
  w_e: eqx.nn.Linear
  norm_edges: LayerNorm
  k_neighbors: int = eqx.field(static=True)
  rbf_dim: int = eqx.field(static=True)
  pos_embed_dim: int = eqx.field(static=True)

  def __init__(
    self,
    node_features: int,
    edge_features: int,
    k_neighbors: int,
    *,
    key: PRNGKeyArray,
  ) -> None:
    """Initialize feature extraction layers.

    Args:
      node_features: Dimension of node features (not directly used, kept for API compat).
      edge_features: Dimension of edge features.
      k_neighbors: Number of nearest neighbors to consider.
      key: PRNG key for initialization.

    """
    keys = jax.random.split(key, 2)

    self.k_neighbors = k_neighbors
    self.rbf_dim = 16
    self.pos_embed_dim = POS_EMBED_DIM

    pos_one_hot_dim = 2 * MAXIMUM_RELATIVE_FEATURES + 2  # 66
    edge_embed_in_dim = 416  # Match original model's edge embedding input size

    self.w_pos = eqx.nn.Linear(pos_one_hot_dim, POS_EMBED_DIM, key=keys[0])
    self.w_e = eqx.nn.Linear(edge_embed_in_dim, edge_features, use_bias=False, key=keys[1])
    self.norm_edges = LayerNorm(edge_features)
    # NOTE: w_e_proj removed - W_e projection is in main model (matches ColabDesign)

  def __call__(
    self,
    prng_key: PRNGKeyArray,
    structure_coordinates: StructureAtomicCoordinates,
    mask: AlphaCarbonMask,
    residue_index: ResidueIndex,
    chain_index: ChainIndex,
    backbone_noise: BackboneNoise | None,
  ) -> tuple[EdgeFeatures, NeighborIndices, PRNGKeyArray]:
    """Extract and project features from protein structure.

    Args:
      prng_key: PRNG key for coordinate noise.
      structure_coordinates: Atomic coordinates (N, CA, C, O).
      mask: Alpha carbon mask.
      residue_index: Residue indices.
      chain_index: Chain indices.
      backbone_noise: Noise to add to backbone coordinates.

    Returns:
      Tuple of (edge_features, neighbor_indices, updated_prng_key).

    """
    if backbone_noise is None:
      backbone_noise = jnp.array(0.0, dtype=jnp.float32)

    noised_coordinates, prng_key = apply_noise_to_coordinates(
      prng_key,
      structure_coordinates,
      backbone_noise=backbone_noise,
    )
    backbone_atom_coordinates = compute_backbone_coordinates(noised_coordinates)
    distances = compute_backbone_distance(backbone_atom_coordinates)

    distances_masked = jnp.array(
      jnp.where(
        (mask[:, None] * mask[None, :]).astype(bool),
        distances,
        jnp.inf,
      ),
    )

    k = min(self.k_neighbors, structure_coordinates.shape[0])
    _, neighbor_indices = top_k(-distances_masked, k)
    neighbor_indices = jnp.array(neighbor_indices, dtype=jnp.int32)

    rbf = compute_radial_basis(backbone_atom_coordinates, neighbor_indices)
    neighbor_offsets = compute_neighbor_offsets(residue_index, neighbor_indices)

    # Get edge chains neighbors
    edge_chains = (chain_index[:, None] == chain_index[None, :]).astype(int)
    edge_chains_neighbors = jnp.take_along_axis(
      edge_chains,
      neighbor_indices,
      axis=1,
    )

    # Encode positions
    neighbor_offset_factor = jnp.clip(
      neighbor_offsets + MAXIMUM_RELATIVE_FEATURES,
      0,
      2 * MAXIMUM_RELATIVE_FEATURES,
    )
    edge_chain_factor = (1 - edge_chains_neighbors) * (2 * MAXIMUM_RELATIVE_FEATURES + 1)
    encoded_offset = neighbor_offset_factor * edge_chains_neighbors + edge_chain_factor
    encoded_offset_one_hot = jax.nn.one_hot(
      encoded_offset,
      2 * MAXIMUM_RELATIVE_FEATURES + 2,
    )

    # vmap over (N, K)
    encoded_positions = jax.vmap(jax.vmap(self.w_pos))(encoded_offset_one_hot)

    # Embed edges
    edges = jnp.concatenate([encoded_positions, rbf], axis=-1)
    jax.debug.print("🔍 PrxteinMPNN ProteinFeatures.__call__")
    jax.debug.print("  edges.shape: {}", edges.shape)
    jax.debug.print("  edges[0,0,:5]: {}", edges[0,0,:5])

    edge_features = jax.vmap(jax.vmap(self.w_e))(edges)
    jax.debug.print("  After w_e (edge_embedding), edge_features.shape: {}", edge_features.shape)
    jax.debug.print("  After w_e, edge_features[0,0,:5]: {}", edge_features[0,0,:5])

    edge_features = jax.vmap(jax.vmap(self.norm_edges))(edge_features)
    jax.debug.print("  After norm_edges (FINAL), edge_features.shape: {}", edge_features.shape)
    jax.debug.print("  After norm_edges (FINAL), edge_features[0,0,:5]: {}", edge_features[0,0,:5])

    # NOTE: W_e projection is applied in the main model, not here!
    # This matches ColabDesign architecture where ProteinFeatures returns
    # edge_embedding + norm output, and W_e is applied in score() method.

    return edge_features, neighbor_indices, prng_key
