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
  dense: eqx.nn.MLP
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
      width_size=mlp_input_dim,
      depth=1,
      activation=_gelu,
      key=keys[3],
    )
    self.norm2 = LayerNorm(node_features)

  def __call__(
    self,
    node_features: NodeFeatures,
    layer_edge_features: EdgeFeatures,
    mask: AlphaCarbonMask,
    scale: float = 30.0,
    attention_mask: Array | None = None,
    *,
    key: PRNGKeyArray | None = None,
  ) -> NodeFeatures:
    """Forward pass with Mixed Precision Stability (FP32 Accumulation)."""
    keys = jax.random.split(key, 2) if key is not None else (None, None)

    compute_dtype = node_features.dtype

    # FIX: Ensure mask matches compute_dtype to prevent upcasting output
    mask = mask.astype(compute_dtype)

    node_features_expand = jnp.tile(
      jnp.expand_dims(node_features, -2),
      [1, layer_edge_features.shape[1], 1],
    )

    mlp_input = jnp.concatenate([node_features_expand, layer_edge_features], -1)
    message = jax.vmap(jax.vmap(self.message_mlp))(mlp_input)

    if attention_mask is not None:
      # FIX: Cast attention_mask to avoid upcasting
      mask_cast = attention_mask.astype(compute_dtype)
      message = jnp.expand_dims(mask_cast, -1) * message

    # --- STABILITY FIX: Accumulate in Float32 ---
    message_f32 = message.astype(jnp.float32)
    aggregated_message_f32 = jnp.sum(message_f32, -2) / scale
    aggregated_message = aggregated_message_f32.astype(compute_dtype)
    # --------------------------------------------

    aggregated_message = self.dropout1(aggregated_message, key=keys[0])
    node_features = node_features + aggregated_message

    node_features_norm1 = jax.vmap(self.norm1)(node_features)
    dense_output = jax.vmap(self.dense)(node_features_norm1)
    dense_output = self.dropout2(dense_output, key=keys[1])

    node_features = node_features_norm1 + dense_output
    node_features_norm2 = jax.vmap(self.norm2)(node_features)

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
    edge_features: int,
    hidden_features: int,
    num_layers: int = 3,
    dropout_rate: float = 0.1,
    *,
    key: PRNGKeyArray,
  ) -> None:
    self.node_features_dim = node_features
    self.edge_features_dim = edge_features

    keys = jax.random.split(key, num_layers)

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
    edge_features: EdgeFeatures,
    neighbor_indices: NeighborIndices,
    mask: AlphaCarbonMask,
    *,
    key: PRNGKeyArray | None = None,
  ) -> NodeFeatures:
    """Forward pass for UNCONDITIONAL decoding."""
    keys = jax.random.split(key, len(self.layers)) if key is not None else [None] * len(self.layers)

    # FIX: Ensure mask is cast to prevent upcasting at output
    mask = mask.astype(node_features.dtype)

    zeros_with_edges = concatenate_neighbor_nodes(
      jnp.zeros_like(node_features),
      edge_features,
      neighbor_indices,
    )

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
    node_features: NodeFeatures,
    edge_features: EdgeFeatures,
    neighbor_indices: NeighborIndices,
    mask: AlphaCarbonMask,
    ar_mask: AutoRegressiveMask,
    one_hot_sequence: OneHotProteinSequence,
    w_s_weight: Array,
    *,
    key: PRNGKeyArray | None = None,
  ) -> NodeFeatures:
    """Forward pass for CONDITIONAL decoding (scoring)."""
    keys = jax.random.split(key, len(self.layers)) if key is not None else [None] * len(self.layers)

    # FIX: Explicitly cast masks to compute_dtype (bfloat16)
    # This prevents 'float32 * bfloat16 -> float32' promotion
    compute_dtype = node_features.dtype
    mask = mask.astype(compute_dtype)
    ar_mask = ar_mask.astype(compute_dtype)

    embedded_sequence = jnp.atleast_2d(one_hot_sequence) @ w_s_weight

    temp_node_edge = concatenate_neighbor_nodes(
      jnp.zeros_like(node_features),
      edge_features,
      neighbor_indices,
    )

    node_edge_features = concatenate_neighbor_nodes(
      node_features,
      temp_node_edge,
      neighbor_indices,
    )

    sequence_edge_features = concatenate_neighbor_nodes(
      embedded_sequence,
      edge_features,
      neighbor_indices,
    )

    # Masking logic
    attention_mask = jnp.take_along_axis(ar_mask, neighbor_indices, axis=1)

    # Now these operations are (bf16 * bf16), staying in bf16
    mask_bw = mask[:, None] * attention_mask
    mask_fw = mask[:, None] * (1 - attention_mask)

    masked_node_edge_features = mask_fw[..., None] * node_edge_features

    loop_node_features = node_features
    for i, layer in enumerate(self.layers):
      current_features = concatenate_neighbor_nodes(
        loop_node_features,
        sequence_edge_features,
        neighbor_indices,
      )

      layer_edge_features = (mask_bw[..., None] * current_features) + masked_node_edge_features

      loop_node_features = layer(
        loop_node_features,
        layer_edge_features,
        mask,
        key=keys[i],
      )

    return loop_node_features
