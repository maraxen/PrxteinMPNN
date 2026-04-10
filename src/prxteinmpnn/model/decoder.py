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
      dropout_rate: Dropout rate (default: 0.1).
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

    # Input dim is [h_i, e_context]
    mlp_input_dim = node_features + edge_context_features

    self.message_mlp = eqx.nn.MLP(
      in_size=mlp_input_dim,
      out_size=node_features,
      width_size=node_features,
      depth=2,
      activation=_gelu,
      key=keys[2],
    )
    self.norm1 = LayerNorm(node_features)
    self.dense = eqx.nn.MLP(
      in_size=node_features,
      out_size=node_features,
      width_size=node_features * 4,
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
    attention_mask: Array | None = None,
    inference: bool = False,
    *,
    key: PRNGKeyArray | None = None,
  ) -> NodeFeatures:
    """Forward pass for the decoder layer."""
    # Pass the key to jax.random.split for potential dropout use
    if key is None:
      inference = True
    keys = jax.random.split(key, 2) if key is not None else (None, None)

    node_features_expand = jnp.tile(
      jnp.expand_dims(node_features, -2),
      [1, layer_edge_features.shape[1], 1],
    )

    # Concat with context [h_i, e_context]
    mlp_input = jnp.concatenate([node_features_expand, layer_edge_features], -1)

    # Apply MLP to each (atom, neighbor) pair: vmap over atoms, then over neighbors
    message = jax.vmap(jax.vmap(self.message_mlp))(mlp_input)

    # Apply attention mask if provided (for conditional decoding)
    if attention_mask is not None:
      mask_cast = attention_mask.astype(message.dtype)
      message = jnp.expand_dims(mask_cast, -1) * message

    # Stability fix: Accumulate message sums in float32
    message_f32 = message.astype(jnp.float32)
    aggregated_message_f32 = jnp.sum(message_f32, -2) / scale
    aggregated_message = aggregated_message_f32.astype(message.dtype)

    # Aggregate messages and apply dropout
    h_V = node_features + self.dropout1(aggregated_message, key=keys[0], inference=inference)
    h_V = jax.vmap(self.norm1)(h_V)

    # Dense layer and residue connection
    h_dense = jax.vmap(self.dense)(h_V)
    h_V = h_V + self.dropout2(h_dense, key=keys[1], inference=inference)
    h_V = jax.vmap(self.norm2)(h_V)

    # Handle both batched (N,) mask and scalar mask
    if jnp.ndim(mask) == 0:
      return mask * h_V
    return mask[:, None] * h_V


class DecoderLayerJ(eqx.Module):
    """Specialized decoder layer for LigandMPNN context atoms."""
    w1: eqx.nn.Linear
    w2: eqx.nn.Linear
    w3: eqx.nn.Linear
    dense: eqx.nn.MLP
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm
    dropout1: eqx.nn.Dropout
    dropout2: eqx.nn.Dropout
    scale: float = eqx.field(static=True)

    def __init__(
        self,
        hidden_dim: int,
        in_dim: int,
        dropout: float = 0.1,
        scale: float = 30.0,
        *,
        key: PRNGKeyArray,
    ):
        keys = jax.random.split(key, 5)
        self.w1 = eqx.nn.Linear(hidden_dim + in_dim, hidden_dim, key=keys[0])
        self.w2 = eqx.nn.Linear(hidden_dim, hidden_dim, key=keys[1])
        self.w3 = eqx.nn.Linear(hidden_dim, hidden_dim, key=keys[2])
        self.dense = eqx.nn.MLP(hidden_dim, hidden_dim, hidden_dim * 4, depth=1, activation=_gelu, key=keys[3])
        self.norm1 = eqx.nn.LayerNorm(hidden_dim)
        self.norm2 = eqx.nn.LayerNorm(hidden_dim)
        self.dropout1 = eqx.nn.Dropout(dropout)
        self.dropout2 = eqx.nn.Dropout(dropout)
        self.scale = scale

    def __call__(
        self,
        h_v: NodeFeatures,
        h_e: EdgeFeatures,
        mask_v: AlphaCarbonMask | None = None,
        mask_attend: Array | None = None,
        inference: bool = False,
        *,
        key: PRNGKeyArray | None = None,
    ) -> NodeFeatures:
        if key is None:
            inference = True
        keys = jax.random.split(key, 2) if key is not None else (None, None)

        # h_v: [L, M, D]
        # h_e: [L, M, M, D]

        # Expand h_v to match h_e for local context
        h_v_expand = jnp.expand_dims(h_v, axis=-2)
        h_v_expand = jnp.broadcast_to(h_v_expand, h_v_expand.shape[:-2] + (h_e.shape[-2], h_v.shape[-1]))

        h_ev = jnp.concatenate([h_v_expand, h_e], axis=-1)

        # Message passing
        h_message = jax.vmap(jax.vmap(jax.vmap(self.w1)))(h_ev)
        h_message = _gelu(h_message)
        h_message = jax.vmap(jax.vmap(jax.vmap(self.w2)))(h_message)
        h_message = _gelu(h_message)
        h_message = jax.vmap(jax.vmap(jax.vmap(self.w3)))(h_message)

        if mask_attend is not None:
            h_message = jnp.expand_dims(mask_attend, axis=-1) * h_message

        dh = jnp.sum(h_message, axis=-2) / self.scale

        h_v = jax.vmap(jax.vmap(self.norm1))(h_v + self.dropout1(dh, key=keys[0], inference=inference))

        # MLP
        dh_dense = jax.vmap(jax.vmap(self.dense))(h_v)
        h_v = jax.vmap(jax.vmap(self.norm2))(h_v + self.dropout2(dh_dense, key=keys[1], inference=inference))

        if mask_v is not None:
            h_v = jnp.expand_dims(mask_v, axis=-1) * h_v

        return h_v


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
      dropout_rate: Dropout rate (default: 0.1).
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

    # The context dim is [h_i, e_ij, h_j]
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
      key: PRNG key for dropout (optional).

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
    if key is None:
      inference = True
    keys = jax.random.split(key, len(self.layers)) if key is not None else [None] * len(self.layers)

    # Prepare context tensor *once*
    # For unconditional: [0, h_E_ij, h_V_j] where j is the neighbor
    # First concatenate zeros with edge features
    zeros_with_edges = concatenate_neighbor_nodes(
      jnp.zeros_like(node_features),
      edge_features,
      neighbor_indices,
    )

    # Then concatenate node features with the above
    layer_edge_features = concatenate_neighbor_nodes(
      node_features,
      zeros_with_edges,
      neighbor_indices,
    )

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
    w_s_weight: jnp.ndarray,  # Sequence embedding weight
    inference: bool = False,
    *,
    key: PRNGKeyArray | None = None,
  ) -> NodeFeatures:
    """Run conditional decoding (scoring).

    Args:
      node_features: Node features from encoder of shape (N, 128).
      edge_features: Edge features from encoder of shape (N, K, 128).
      neighbor_indices: Indices of neighbors for each node.
      mask: Alpha carbon mask of shape (N,).
      ar_mask: Autoregressive mask for conditional decoding.
      one_hot_sequence: One-hot encoded protein sequence.
      w_s_weight: Sequence embedding weight matrix.
      inference: Whether to run in inference mode (disables dropout).
      key: PRNG key for dropout (optional).

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
    if key is None:
      inference = True
    keys = jax.random.split(key, len(self.layers)) if key is not None else [None] * len(self.layers)

    embedded_sequence = jnp.atleast_2d(one_hot_sequence) @ w_s_weight

    temp_node_edge = concatenate_neighbor_nodes(
      jnp.zeros_like(node_features),
      edge_features,
      neighbor_indices,
    )  # [0, e_ij, h_j]

    node_edge_features = concatenate_neighbor_nodes(
      node_features,
      temp_node_edge,
      neighbor_indices,
    )  # [h_i, [0, e_ij, h_j]]

    sequence_edge_features = concatenate_neighbor_nodes(
      embedded_sequence,
      edge_features,
      neighbor_indices,
    )  # [e_ij, s_j]

    attention_mask = jnp.take_along_axis(ar_mask, neighbor_indices, axis=1)
    mask_bw = mask[:, None] * attention_mask
    mask_fw = mask[:, None] * (1 - attention_mask)
    masked_node_edge_features = mask_fw[..., None] * node_edge_features

    loop_node_features = node_features
    for i, layer in enumerate(self.layers):
      # Construct the decoder context for this layer by gathering neighbor features
      # and concatenating with sequence edge features
      current_features = concatenate_neighbor_nodes(
        loop_node_features,
        sequence_edge_features,
        neighbor_indices,
      )

      layer_edge_features = (mask_bw[..., None] * current_features) + masked_node_edge_features

      # Run the layer (masking already applied to layer_edge_features)
      loop_node_features = layer(
        loop_node_features,
        layer_edge_features,
        mask,
        inference=inference,
        key=keys[i],
      )

    return loop_node_features
