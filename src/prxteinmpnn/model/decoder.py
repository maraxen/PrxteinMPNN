"""Decoder module for PrxteinMPNN.

This module contains the Equinox-based decoder implementation for ProteinMPNN.
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
    Array,
    AutoRegressiveMask,
    EdgeFeatures,
    NeighborIndices,
    NodeFeatures,
    OneHotProteinSequence,
    PRNGKeyArray,
  )

# Layer normalization with a standard epsilon
LayerNorm = eqx.nn.LayerNorm
_gelu = partial(jax.nn.gelu, approximate=False)


class DecoderLayer(eqx.Module):
  """A single decoder layer for the ProteinMPNN model."""

  message_mlp: eqx.nn.MLP
  norm1: LayerNorm
  dense: eqx.nn.MLP  # Use eqx.nn.MLP directly
  norm2: LayerNorm
  dropout1: eqx.nn.Dropout
  dropout2: eqx.nn.Dropout

  def __init__(
    self,
    node_features: int,
    edge_context_features: int,
    _hidden_features: int,
    dropout_rate: float = 0.1,
    *,
    key: PRNGKeyArray,
  ) -> None:
    """Initialize the decoder layer.

    Args:
      node_features: Dimension of node features (e.g., 128).
      edge_context_features: Dimension of edge context (e.g., 384).
      hidden_features: Dimension of hidden layer in dense MLP.
      key: PRNG key for initialization.

    Returns:
      None

    Raises:
      None

    Example:
      >>> key = jax.random.PRNGKey(0)
      >>> layer = DecoderLayer(128, 384, 128, key=key)

    """
    keys = jax.random.split(key, 4)

    self.dropout1 = eqx.nn.Dropout(dropout_rate)
    self.dropout2 = eqx.nn.Dropout(dropout_rate)

    # Input dim is [h_i (128), e_context (384)] = 512
    mlp_input_dim = node_features + edge_context_features

    # Message MLP: 512 -> 128 -> 128 -> 128 (width=node_features, not hidden_features)
    self.message_mlp = eqx.nn.MLP(
      in_size=mlp_input_dim,
      out_size=node_features,
      width_size=node_features,  # 128, matches functional W1/W2/W3
      depth=2,
      activation=_gelu,
      key=keys[2],
    )
    self.norm1 = LayerNorm(node_features)
    # Use eqx.nn.MLP for the dense layer
    self.dense = eqx.nn.MLP(
      in_size=node_features,
      out_size=node_features,
      width_size=mlp_input_dim,
      depth=1,
      activation=_gelu,
      key=keys[3],
    )
    self.norm2 = LayerNorm(node_features)

  def __call__(
    self,
    node_features: NodeFeatures,
    layer_edge_features: EdgeFeatures,  # This is the (N, K, 384) context
    mask: AlphaCarbonMask,
    scale: float = 30.0,
    attention_mask: Array | None = None,  # Optional attention mask for conditional decoding
    *,
    key: PRNGKeyArray | None = None,
  ) -> NodeFeatures:
    """Forward pass for the decoder layer.

    Works for both N-batch (N, C) and single-node (1, C) inputs.

    Args:
      node_features: Node features tensor of shape (N, C).
      layer_edge_features: Edge context features of shape (N, K, 384).
      mask: Alpha carbon mask of shape (N,).
      scale: Scaling factor for message aggregation (default: 30.0).
      attention_mask: Optional attention mask for conditional decoding.

    Returns:
      Updated node features of shape (N, C).

    Raises:
      None

    Example:
      >>> key = jax.random.PRNGKey(0)
      >>> layer = DecoderLayer(128, 384, 128, key=key)
      >>> node_feats = jnp.ones((10, 128))
      >>> edge_feats = jnp.ones((10, 30, 384))
      >>> mask = jnp.ones((10,))
      >>> output = layer(node_feats, edge_feats, mask)

    """
    keys = jax.random.split(key, 2) if key is not None else (None, None)

    # Tile central node features [h_i (N, 1, C)]
    node_features_expand = jnp.tile(
      jnp.expand_dims(node_features, -2),
      [1, layer_edge_features.shape[1], 1],
    )

    # Concat with context [h_i (N, K, C), e_context (N, K, 384)]
    mlp_input = jnp.concatenate([node_features_expand, layer_edge_features], -1)

    # Apply MLP to each (atom, neighbor) pair: vmap over atoms, then over neighbors
    message = jax.vmap(jax.vmap(self.message_mlp))(mlp_input)

    # Apply attention mask if provided (for conditional decoding)
    if attention_mask is not None:
      message = jnp.expand_dims(attention_mask, -1) * message

    # Aggregate messages
    aggregated_message = jnp.sum(message, -2) / scale

    # dropout1
    aggregated_message = self.dropout1(aggregated_message, key=keys[0])

    node_features = node_features + aggregated_message


    # vmap over N
    node_features_norm1 = jax.vmap(self.norm1)(node_features)
    dense_output = jax.vmap(self.dense)(node_features_norm1)  # This works

    # dropout2
    dense_output = self.dropout2(dense_output, key=keys[1])

    node_features = node_features_norm1 + dense_output
    node_features_norm2 = jax.vmap(self.norm2)(node_features)

    # Handle both batched (N,) mask and scalar mask
    if jnp.ndim(mask) == 0:
      return mask * node_features_norm2
    return mask[:, None] * node_features_norm2


class Decoder(eqx.Module):
  """The complete decoder module for ProteinMPNN."""

  layers: tuple[DecoderLayer, ...]
  node_features_dim: int = eqx.field(static=True)
  edge_features_dim: int = eqx.field(static=True)

  def __init__(
    self,
    node_features: int,
    edge_features: int,  # This is the raw edge_features dim (128)
    hidden_features: int,
    num_layers: int = 3,
    dropout_rate: float = 0.1,
    *,
    key: PRNGKeyArray,
  ) -> None:
    """Initialize the decoder.

    Args:
      node_features: Dimension of node features (e.g., 128).
      edge_features: Dimension of edge features (e.g., 128).
      hidden_features: Dimension of hidden layer in decoder layers.
      num_layers: Number of decoder layers (default: 3).
      key: PRNG key for initialization.

    Returns:
      None

    Raises:
      None

    Example:
      >>> key = jax.random.PRNGKey(0)
      >>> decoder = Decoder(128, 128, 128, num_layers=3, key=key)

    """
    self.node_features_dim = node_features
    self.edge_features_dim = edge_features

    keys = jax.random.split(key, num_layers)

    # The context dim is [h_i, e_ij, h_j] = node_features + edge_features + node_features
    edge_context_features = 2 * node_features + edge_features

    self.layers = tuple(
      DecoderLayer(
        node_features,
        edge_context_features,
        hidden_features,
        dropout_rate=dropout_rate,
        key=k,
      )
      for k in keys
    )

  def __call__(
    self,
    node_features: NodeFeatures,
    edge_features: EdgeFeatures,  # Raw 128-dim edges
    neighbor_indices: NeighborIndices,
    mask: AlphaCarbonMask,
    *,
    key: PRNGKeyArray | None = None,
  ) -> NodeFeatures:
    """Forward pass for UNCONDITIONAL decoding.

    Args:
      node_features: Node features from encoder of shape (N, 128).
      edge_features: Edge features from encoder of shape (N, K, 128).
      neighbor_indices: Indices of neighbors for each node.
      mask: Alpha carbon mask of shape (N,).

    Returns:
      Decoded node features of shape (N, 128).

    Raises:
      None

    Example:
      >>> key = jax.random.PRNGKey(0)
      >>> decoder = Decoder(128, 128, 128, num_layers=3, key=key)
      >>> node_feats = jnp.ones((10, 128))
      >>> edge_feats = jnp.ones((10, 30, 128))
      >>> neighbor_idx = jnp.arange(300).reshape(10, 30)
      >>> mask = jnp.ones((10,))
      >>> output = decoder(node_feats, edge_feats, neighbor_idx, mask)

    """
    keys = jax.random.split(key, len(self.layers)) if key is not None else [None] * len(self.layers)

    # Prepare 384-dim context tensor *once*
    # For unconditional: [0, h_E_ij, h_V_j] where j is the neighbor
    # First concatenate zeros with edge features
    zeros_with_edges = concatenate_neighbor_nodes(
      jnp.zeros_like(node_features),
      edge_features,
      neighbor_indices,
    )  # Shape: (N, K, 128 + 128) = (N, K, 256)

    # Then concatenate node features with the above
    layer_edge_features = concatenate_neighbor_nodes(
      node_features,
      zeros_with_edges,
      neighbor_indices,
    )  # Shape: (N, K, 256 + 128) = (N, K, 384)

    loop_node_features = node_features
    for i, layer in enumerate(self.layers):
      loop_node_features = layer(
        loop_node_features,
        layer_edge_features,
        mask,
        key=keys[i],
      )
    return loop_node_features

  def call_conditional(
    self,
    node_features: NodeFeatures,  # h_i from encoder
    edge_features: EdgeFeatures,  # e_ij from encoder
    neighbor_indices: NeighborIndices,
    mask: AlphaCarbonMask,
    ar_mask: AutoRegressiveMask,
    one_hot_sequence: OneHotProteinSequence,
    w_s_weight: Array,  # Sequence embedding weight
    *,
    key: PRNGKeyArray | None = None,
  ) -> NodeFeatures:
    """Forward pass for CONDITIONAL decoding (scoring).

    Args:
      node_features: Node features from encoder of shape (N, 128).
      edge_features: Edge features from encoder of shape (N, K, 128).
      neighbor_indices: Indices of neighbors for each node.
      mask: Alpha carbon mask of shape (N,).
      ar_mask: Autoregressive mask for conditional decoding.
      one_hot_sequence: One-hot encoded protein sequence.
      w_s_weight: Sequence embedding weight matrix.

    Returns:
      Decoded node features of shape (N, 128).

    Raises:
      None

    Example:
      >>> key = jax.random.PRNGKey(0)
      >>> decoder = Decoder(128, 128, 128, num_layers=3, key=key)
      >>> node_feats = jnp.ones((10, 128))
      >>> edge_feats = jnp.ones((10, 30, 128))
      >>> neighbor_indices = jnp.arange(300).reshape(10, 30)
      >>> mask = jnp.ones((10,))
      >>> ar_mask = jnp.ones((10, 10))
      >>> seq = jax.nn.one_hot(jnp.arange(10), 21)
      >>> w_s = jnp.ones((21, 128))
      >>> output = decoder.call_conditional(
      ...     node_feats, edge_feats, neighbor_indices, mask, ar_mask, seq, w_s
      ... )

    """
    keys = jax.random.split(key, len(self.layers)) if key is not None else [None] * len(self.layers)

    # 1. Embed the sequence
    embedded_sequence = jnp.atleast_2d(one_hot_sequence) @ w_s_weight  # s_i

    # 2. Initialize context features
    # Following functional implementation (decoder.py lines 127-141)

    # First: [0, e_ij, h_j] -> (N, K, 256)
    temp_node_edge = concatenate_neighbor_nodes(
      jnp.zeros_like(node_features),
      edge_features,
      neighbor_indices,
    )
    # Second: [h_i, [0, e_ij, h_j]] -> (N, K, 384)
    node_edge_features = concatenate_neighbor_nodes(
      node_features,
      temp_node_edge,
      neighbor_indices,
    )

    # [e_ij, s_j] -> (N, K, 256)
    # Note: concatenate_neighbor_nodes returns [edge_features, neighbor_features]
    sequence_edge_features = concatenate_neighbor_nodes(
      embedded_sequence,
      edge_features,
      neighbor_indices,
    )

    # 3. Prepare masks
    attention_mask = jnp.take_along_axis(ar_mask, neighbor_indices, axis=1)
    mask_bw = mask[:, None] * attention_mask
    mask_fw = mask[:, None] * (1 - attention_mask)
    masked_node_edge_features = mask_fw[..., None] * node_edge_features

    # 4. Run the decoder loop
    # Following functional implementation (decoder.py lines 480-497)
    loop_node_features = node_features
    for i, layer in enumerate(self.layers):
      # Construct the decoder context for this layer by gathering neighbor features
      # and concatenating with sequence edge features
      current_features = concatenate_neighbor_nodes(
        loop_node_features,  # (N, 128) -> gather neighbors -> (N, K, 128) = h_j
        sequence_edge_features,  # (N, K, 256) = [e_ij, s_j]
        neighbor_indices,
      )  # Result: (N, K, 384) = [e_ij, s_j, h_j]

      layer_edge_features = (mask_bw[..., None] * current_features) + masked_node_edge_features

      # Run the layer (masking already applied to layer_edge_features)
      loop_node_features = layer(
        loop_node_features,
        layer_edge_features,
        mask,
        key=keys[i],
      )

    return loop_node_features
