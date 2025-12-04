"""Modified feature extraction module using direct matrix operations instead of vmap.

This version aims to match ColabDesign's numerical behavior more closely.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
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


class ProteinFeaturesDirect(eqx.Module):
  """Extracts and projects features using direct matrix operations.

  This version replaces vmap operations with direct matrix multiplications
  to match ColabDesign's numerical behavior more closely.
  """

  w_pos: eqx.nn.Linear
  w_e: eqx.nn.Linear
  norm_edges: LayerNorm
  w_e_proj: eqx.nn.Linear
  k_neighbors: int = eqx.field(static=True)
  rbf_dim: int = eqx.field(static=True)
  pos_embed_dim: int = eqx.field(static=True)

  def __init__(
    self,
    node_features: int,  # noqa: ARG002
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
    keys = jax.random.split(key, 3)

    self.k_neighbors = k_neighbors
    self.rbf_dim = 16
    self.pos_embed_dim = POS_EMBED_DIM

    pos_one_hot_dim = 2 * MAXIMUM_RELATIVE_FEATURES + 2  # 66
    edge_embed_in_dim = 416  # Match original model's edge embedding input size

    self.w_pos = eqx.nn.Linear(pos_one_hot_dim, POS_EMBED_DIM, key=keys[0])
    self.w_e = eqx.nn.Linear(edge_embed_in_dim, edge_features, use_bias=False, key=keys[1])
    self.norm_edges = LayerNorm(edge_features)
    self.w_e_proj = eqx.nn.Linear(edge_features, edge_features, key=keys[2])

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
        (mask[:, None] * mask[None, :]).astype(jnp.bool_),
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
    edge_chains = (chain_index[:, None] == chain_index[None, :]).astype(jnp.int32)
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

    # Direct matrix multiply instead of vmap (shape: N, K, 66 @ 66, 16 = N, K, 16)
    w_pos_weight = self.w_pos.weight  # (16, 66)
    w_pos_bias = self.w_pos.bias  # (16,)
    if w_pos_bias is None:
      encoded_positions = encoded_offset_one_hot @ w_pos_weight.T
    else:
      encoded_positions = encoded_offset_one_hot @ w_pos_weight.T + w_pos_bias

    # Embed edges (shape: N, K, 416 @ 416, 128 = N, K, 128)
    edges = jnp.concatenate([encoded_positions, rbf], axis=-1)
    w_e_weight = self.w_e.weight  # (128, 416)
    edge_features = edges @ w_e_weight.T  # No bias for w_e

    # LayerNorm: normalize across the feature dimension (axis=-1)
    # This matches ColabDesign's approach
    mean = edge_features.mean(axis=-1, keepdims=True)
    var = edge_features.var(axis=-1, keepdims=True)
    edge_features_normalized = (edge_features - mean) / jnp.sqrt(var + 1e-5)

    # Apply scale and offset from LayerNorm
    # Note: eqx.nn.LayerNorm stores scale as 'weight' and offset as 'bias'
    if hasattr(self.norm_edges, "weight") and self.norm_edges.weight is not None:
      scale = self.norm_edges.weight
    else:
      scale = jnp.ones(edge_features.shape[-1])
    if hasattr(self.norm_edges, "bias") and self.norm_edges.bias is not None:
      offset = self.norm_edges.bias
    else:
      offset = jnp.zeros(edge_features.shape[-1])
    edge_features = edge_features_normalized * scale + offset

    # Project features (shape: N, K, 128 @ 128, 128 = N, K, 128)
    w_e_proj_weight = self.w_e_proj.weight  # (128, 128)
    w_e_proj_bias = self.w_e_proj.bias  # (128,)
    if w_e_proj_bias is None:
      edge_features = edge_features @ w_e_proj_weight.T
    else:
      edge_features = edge_features @ w_e_proj_weight.T + w_e_proj_bias

    return edge_features, neighbor_indices, prng_key
