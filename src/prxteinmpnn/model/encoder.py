"""Encoder module for PrxteinMPNN.

This module contains the EncoderLayer and Encoder classes.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp

from prxteinmpnn.utils.concatenate import concatenate_neighbor_nodes

if TYPE_CHECKING:
  from prxteinmpnn.utils.types import (
    AlphaCarbonMask,
    EdgeFeatures,
    NeighborIndices,
    NodeFeatures,
  )

# Type alias for PRNG keys
PRNGKeyArray = jax.Array

# Layer normalization
LayerNorm = eqx.nn.LayerNorm
_gelu = partial(jax.nn.gelu, approximate=False)


class EncoderLayer(eqx.Module):
  """A single encoder layer for the ProteinMPNN model."""

  edge_message_mlp: eqx.nn.MLP
  norm1: LayerNorm
  dense: eqx.nn.MLP
  norm2: LayerNorm
  edge_update_mlp: eqx.nn.MLP
  norm3: LayerNorm
  node_features_dim: int = eqx.field(static=True)
  edge_features_dim: int = eqx.field(static=True)

  def __init__(
    self,
    node_features: int,
    edge_features: int,
    hidden_features: int,
    *,
    key: PRNGKeyArray,
  ) -> None:
    """Initialize the encoder layer.

    Args:
      node_features: Dimension of node features.
      edge_features: Dimension of edge features.
      hidden_features: Dimension of hidden features in feedforward network.
      key: PRNG key for initialization.

    """
    self.node_features_dim = node_features
    self.edge_features_dim = edge_features

    keys = jax.random.split(key, 4)
    self.edge_message_mlp = eqx.nn.MLP(
      in_size=384,
      out_size=128,
      width_size=128,
      depth=2,
      activation=_gelu,
      key=keys[0],
    )
    self.norm1 = LayerNorm(node_features)
    self.dense = eqx.nn.MLP(
      in_size=node_features,
      out_size=node_features,
      width_size=hidden_features,
      depth=1,
      activation=_gelu,
      key=keys[1],
    )
    self.norm2 = LayerNorm(node_features)
    self.edge_update_mlp = eqx.nn.MLP(
      in_size=node_features * 2 + edge_features,
      out_size=edge_features,
      width_size=edge_features,
      depth=2,
      activation=_gelu,
      key=keys[2],
    )
    self.norm3 = LayerNorm(edge_features)

  def _get_mlp_input(
    self,
    h: NodeFeatures,
    e: EdgeFeatures,
    neighbor_indices: NeighborIndices,
  ) -> jax.Array:
    """Return the input tensor [h_i, e_ij, h_j] for edge_message_mlp."""
    e_with_neighbors = concatenate_neighbor_nodes(h, e, neighbor_indices)
    node_expanded = jnp.tile(jnp.expand_dims(h, -2), [1, e_with_neighbors.shape[-2], 1])
    return jnp.concatenate([node_expanded, e_with_neighbors], -1)

  def __call__(
    self,
    node_features: NodeFeatures,
    edge_features: EdgeFeatures,
    neighbor_indices: NeighborIndices,
    mask: AlphaCarbonMask,
    mask_attend: jnp.ndarray | None = None,
    scale: float = 30.0,
  ) -> tuple[NodeFeatures, EdgeFeatures]:
    """Forward pass for the encoder layer."""
    mlp_input = self._get_mlp_input(node_features, edge_features, neighbor_indices)
    message = jax.vmap(jax.vmap(self.edge_message_mlp))(mlp_input)

    # Apply attention mask to zero out messages from invalid neighbors
    if mask_attend is not None:
      message = jnp.expand_dims(mask_attend, -1) * message

    aggregated_message = jnp.sum(message, -2) / scale
    node_features = node_features + aggregated_message
    node_features = jax.vmap(self.norm1)(node_features)
    node_features = node_features + jax.vmap(self.dense)(node_features)
    node_features = jax.vmap(self.norm2)(node_features)
    node_features = mask[:, None] * node_features

    edge_features_cat = concatenate_neighbor_nodes(node_features, edge_features, neighbor_indices)
    node_features_expand = jnp.tile(
      jnp.expand_dims(node_features, -2),
      [1, edge_features_cat.shape[-2], 1],
    )
    mlp_input_edge_update = jnp.concatenate([node_features_expand, edge_features_cat], -1)
    edge_message = jax.vmap(jax.vmap(self.edge_update_mlp))(mlp_input_edge_update)
    edge_features = edge_features + edge_message
    edge_features = jax.vmap(jax.vmap(self.norm3))(edge_features)

    return node_features, edge_features


class Encoder(eqx.Module):
  """The complete encoder module for ProteinMPNN."""

  layers: tuple[EncoderLayer, ...]
  node_feature_dim: int = eqx.field(static=True)

  def __init__(
    self,
    node_features: int,
    edge_features: int,
    hidden_features: int,
    num_layers: int = 3,
    *,
    key: PRNGKeyArray,
  ) -> None:
    """Initialize the encoder.

    Args:
      node_features: Dimension of node features.
      edge_features: Dimension of edge features.
      hidden_features: Dimension of hidden features in feedforward network.
      num_layers: Number of encoder layers.
      key: PRNG key for initialization.

    """
    self.node_feature_dim = node_features
    keys = jax.random.split(key, num_layers)
    self.layers = tuple(
      EncoderLayer(node_features, edge_features, hidden_features, key=k) for k in keys
    )

  def __call__(
    self,
    edge_features: EdgeFeatures,
    neighbor_indices: NeighborIndices,
    mask: AlphaCarbonMask,
  ) -> tuple[NodeFeatures, EdgeFeatures]:
    """Forward pass for the encoder."""
    node_features = jnp.zeros((edge_features.shape[0], self.node_feature_dim))

    # Compute attention mask: mask[i] * mask[j] for all pairs, then gather along neighbors
    # Shape: (N, N) -> (N, K) where K is number of neighbors
    mask_2d = mask[:, None] * mask[None, :]  # (N, N)
    mask_attend = jnp.take_along_axis(mask_2d, neighbor_indices, axis=1)  # (N, K)

    for layer in self.layers:
      node_features, edge_features = layer(
        node_features,
        edge_features,
        neighbor_indices,
        mask,
        mask_attend,
      )
    return node_features, edge_features
