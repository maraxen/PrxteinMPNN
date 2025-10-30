"""Equinox-based neural network modules for PrxteinMPNN.

This module contains Equinox implementations of the core neural network
components used in ProteinMPNN, enabling a more modular and composable
architecture.

prxteinmpnn.eqx
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax

if TYPE_CHECKING:
  from jaxtyping import Array

LayerNorm = eqx.nn.LayerNorm


STANDARD_EPSILON = 1e-5


class DenseLayer(eqx.Module):
  """Two-layer feedforward network with GeLU activation.

  This implements a standard feedforward block used in transformers:
    output = Linear_out(GeLU(Linear_in(x)))

  Attributes:
    linear_in: First linear transformation.
    linear_out: Second linear transformation.

  """

  linear_in: eqx.nn.Linear
  linear_out: eqx.nn.Linear

  def __init__(
    self,
    in_features: int,
    hidden_features: int,
    out_features: int,
    *,
    key: jax.Array,
  ) -> None:
    """Initialize DenseLayer module.

    Args:
      in_features: Input feature dimension.
      hidden_features: Hidden layer dimension.
      out_features: Output feature dimension.
      key: PRNG key for initialization.

    Example:
      >>> import jax
      >>> key = jax.random.PRNGKey(0)
      >>> dense = DenseLayer(128, 512, 128, key=key)
      >>> x = jax.random.normal(jax.random.PRNGKey(1), (10, 128))
      >>> y = dense(x)

    """
    key_in, key_out = jax.random.split(key)
    self.linear_in = eqx.nn.Linear(in_features, hidden_features, key=key_in)
    self.linear_out = eqx.nn.Linear(hidden_features, out_features, key=key_out)

  def __call__(self, x: Array) -> Array:
    """Apply dense layer to input.

    Args:
      x: Input tensor of shape (..., in_features).

    Returns:
      Output tensor of shape (..., out_features).

    Example:
      >>> import jax
      >>> key = jax.random.PRNGKey(0)
      >>> dense = DenseLayer(128, 512, 128, key=key)
      >>> x = jax.random.normal(jax.random.PRNGKey(1), (10, 128))
      >>> y = dense(x)
      >>> y.shape
      (10, 128)

    """
    x = self.linear_in(x)
    x = jax.nn.gelu(x, approximate=False)  # Match functional API
    return self.linear_out(x)


class EncoderLayer(eqx.Module):
  """Single encoder layer for ProteinMPNN.

  An encoder layer consists of:
  1. Edge message computation (3-layer MLP on concatenated node/edge features)
  2. Node feature update with normalization and dense layer
  3. Edge feature update with normalization

  Attributes:
    w1, w2, w3: Edge message computation weights (MLP layers).
    norm1: First layer normalization (applied to node features after aggregation).
    dense: Dense feedforward layer for node features.
    norm2: Second layer normalization (applied after dense layer).
    w11, w12, w13: Edge update weights (MLP layers).
    norm3: Third layer normalization (applied to edge features).

  """

  # Edge message computation (3-layer MLP)
  w1: eqx.nn.Linear
  w2: eqx.nn.Linear
  w3: eqx.nn.Linear

  # Node feature update
  norm1: LayerNorm
  dense: DenseLayer
  norm2: LayerNorm

  # Edge feature update (3-layer MLP)
  w11: eqx.nn.Linear
  w12: eqx.nn.Linear
  w13: eqx.nn.Linear
  norm3: LayerNorm

  def __init__(
    self,
    node_features: int,
    edge_features: int,
    hidden_features: int,
    *,
    key: jax.Array,
  ) -> None:
    """Initialize EncoderLayer.

    Args:
      node_features: Node feature dimension.
      edge_features: Edge feature dimension (after concatenation with neighbor nodes).
      hidden_features: Hidden dimension for dense layer.
      key: PRNG key for initialization.

    Example:
      >>> import jax
      >>> key = jax.random.PRNGKey(0)
      >>> layer = EncoderLayer(128, 256, 512, key=key)

    """
    keys = jax.random.split(key, 7)

    # Edge message computation
    self.w1 = eqx.nn.Linear(edge_features, hidden_features, key=keys[0])
    self.w2 = eqx.nn.Linear(hidden_features, hidden_features, key=keys[1])
    self.w3 = eqx.nn.Linear(hidden_features, node_features, key=keys[2])

    # Node feature normalization and dense layer
    self.norm1 = LayerNorm(node_features)
    self.dense = DenseLayer(node_features, hidden_features, node_features, key=keys[3])
    self.norm2 = LayerNorm(node_features)

    # Edge feature update
    self.w11 = eqx.nn.Linear(edge_features, hidden_features, key=keys[4])
    self.w12 = eqx.nn.Linear(hidden_features, hidden_features, key=keys[5])
    self.w13 = eqx.nn.Linear(hidden_features, node_features, key=keys[6])
    self.norm3 = LayerNorm(node_features)


class DecoderLayer(eqx.Module):
  """Single decoder layer for ProteinMPNN.

  A decoder layer consists of:
  1. Edge message computation (3-layer MLP on concatenated sequence/edge features)
  2. Sequence feature update with normalization and dense layer

  Attributes:
    w1, w2, w3: Edge message computation weights (MLP layers).
    norm1: First layer normalization (applied to sequence features after aggregation).
    dense: Dense feedforward layer for sequence features.
    norm2: Second layer normalization (applied after dense layer).

  """

  # Edge message computation (3-layer MLP)
  w1: eqx.nn.Linear
  w2: eqx.nn.Linear
  w3: eqx.nn.Linear

  # Sequence feature update
  norm1: LayerNorm
  dense: DenseLayer
  norm2: LayerNorm

  def __init__(
    self,
    sequence_features: int,
    edge_features: int,
    hidden_features: int,
    *,
    key: jax.Array,
  ) -> None:
    """Initialize DecoderLayer.

    Args:
      sequence_features: Sequence feature dimension.
      edge_features: Edge feature dimension (after concatenation).
      hidden_features: Hidden dimension for dense layer.
      key: PRNG key for initialization.

    Example:
      >>> import jax
      >>> key = jax.random.PRNGKey(0)
      >>> layer = DecoderLayer(128, 256, 512, key=key)

    """
    keys = jax.random.split(key, 4)

    # Edge message computation
    self.w1 = eqx.nn.Linear(edge_features, hidden_features, key=keys[0])
    self.w2 = eqx.nn.Linear(hidden_features, hidden_features, key=keys[1])
    self.w3 = eqx.nn.Linear(hidden_features, sequence_features, key=keys[2])

    # Sequence feature normalization and dense layer
    self.norm1 = LayerNorm(sequence_features)
    self.dense = DenseLayer(sequence_features, hidden_features, sequence_features, key=keys[3])
    self.norm2 = LayerNorm(sequence_features)
