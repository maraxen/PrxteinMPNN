"""Concatenation utilities.

prxteinmpnn.utils.concatenate
"""

import jax
import jax.numpy as jnp

from .types import EdgeFeatures, NeighborIndices, NodeFeatures


@jax.jit
def concatenate_neighbor_nodes(
  node_features: NodeFeatures,
  edge_features: EdgeFeatures,
  neighbor_indices: NeighborIndices,
) -> EdgeFeatures:
  """Concatenate node features with neighbor edge features.

  Args:
    node_features: (L, C_V) node features
    edge_features: (L, K, C_E) edge features
    neighbor_indices: (L, K) neighbor indices

  Returns:
    (L, K, C_V + C_E) concatenated features for neighbors

  """
  neighbor_features = node_features[neighbor_indices]  # (L, K, C_V)
  return jnp.concatenate([edge_features, neighbor_features], axis=-1)
