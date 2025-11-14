"""Feature extraction module for PrxteinMPNN.

This module contains the ProteinFeatures class that extracts and projects
features from raw protein coordinates.
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
    NodeFeatures,
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
  w_e_proj: eqx.nn.Linear  # Final edge projection
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
    structure_mapping: jnp.ndarray | None = None,
    initial_node_features: jnp.ndarray | None = None,
  ) -> tuple[EdgeFeatures, NeighborIndices, NodeFeatures | None, PRNGKeyArray]:
    """Extract and project features from protein structure.

    Args:
      prng_key: PRNG key for coordinate noise.
      structure_coordinates: Atomic coordinates (N, CA, C, O).
      mask: Alpha carbon mask.
      residue_index: Residue indices.
      chain_index: Chain indices.
      backbone_noise: Noise to add to backbone coordinates.
      structure_mapping: Optional (N,) array mapping each residue to a structure ID.
                        When provided (multi-state mode), prevents cross-structure
                        neighbors to avoid information leakage between conformational states.
      initial_node_features: Optional initial node features.
      debug_mode: If True, enables debug prints.

    Returns:
      Tuple of (edge_features, neighbor_indices, updated_prng_key).

    """
    node_features = None if initial_node_features is None else initial_node_features

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

    if structure_mapping is not None:
      same_structure = structure_mapping[:, jnp.newaxis] == structure_mapping[jnp.newaxis, :]
      distances_masked = jnp.array(
        jnp.where(
          same_structure.astype(bool),
          distances_masked,
          jnp.inf,
        ),
      ).squeeze()

    k = min(self.k_neighbors, structure_coordinates.shape[0])
    _, neighbor_indices = top_k(-distances_masked, k)
    neighbor_indices = jnp.array(neighbor_indices, dtype=jnp.int32)

    rbf = compute_radial_basis(backbone_atom_coordinates, neighbor_indices)
    neighbor_offsets = compute_neighbor_offsets(residue_index, neighbor_indices)

    edge_chains = (chain_index[:, None] == chain_index[None, :]).astype(int)
    edge_chains_neighbors = jnp.take_along_axis(
      edge_chains,
      neighbor_indices,
      axis=1,
    )

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

    encoded_positions = jax.vmap(jax.vmap(self.w_pos))(encoded_offset_one_hot)

    edges = jnp.concatenate([encoded_positions, rbf], axis=-1)

    edge_features = jax.vmap(jax.vmap(self.w_e))(edges)

    edge_features = jax.vmap(jax.vmap(self.norm_edges))(edge_features)

    edge_features = jax.vmap(jax.vmap(self.w_e_proj))(edge_features)

    return edge_features, neighbor_indices, node_features, prng_key
