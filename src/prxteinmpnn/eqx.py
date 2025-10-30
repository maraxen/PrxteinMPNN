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
import jax.numpy as jnp

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


class Encoder(eqx.Module):
  """Full encoder stack for ProteinMPNN.

  The encoder processes edge features through multiple encoder layers,
  maintaining and updating both node and edge features.

  Attributes:
    layers: Tuple of EncoderLayer modules.
    node_feature_dim: Dimension of node features.
    scale: Scaling factor for message aggregation (default: 30.0).

  """

  layers: tuple[EncoderLayer, ...]
  node_feature_dim: int
  scale: float = 30.0

  def __init__(
    self,
    node_features: int,
    edge_features: int,
    hidden_features: int,
    num_layers: int = 3,
    scale: float = 30.0,
    *,
    key: jax.Array,
  ) -> None:
    """Initialize Encoder.

    Args:
      node_features: Node feature dimension.
      edge_features: Edge feature dimension.
      hidden_features: Hidden dimension for MLPs.
      num_layers: Number of encoder layers.
      scale: Scaling factor for message aggregation.
      key: PRNG key for initialization.

    Example:
      >>> import jax
      >>> key = jax.random.PRNGKey(0)
      >>> encoder = Encoder(128, 384, 512, num_layers=3, key=key)

    """
    self.node_feature_dim = node_features
    self.scale = scale

    # Create encoder layers (use tuple for hashability with JAX JIT)
    keys = jax.random.split(key, num_layers)
    self.layers = tuple(
      EncoderLayer(node_features, edge_features, hidden_features, key=k) for k in keys
    )

  def __call__(
    self,
    edge_features: jax.Array,
    neighbor_indices: jax.Array,
    mask: jax.Array,
  ) -> tuple[jax.Array, jax.Array]:
    """Run encoder forward pass.

    Args:
      edge_features: Edge features of shape (num_atoms, num_neighbors, edge_dim).
      neighbor_indices: Neighbor indices of shape (num_atoms, num_neighbors).
      mask: Atom mask of shape (num_atoms,).

    Returns:
      tuple: Updated (node_features, edge_features).

    Example:
      >>> import jax
      >>> import jax.numpy as jnp
      >>> key = jax.random.PRNGKey(0)
      >>> encoder = Encoder(128, 384, 512, num_layers=3, key=key)
      >>> edge_features = jax.random.normal(key, (100, 30, 384))
      >>> neighbor_indices = jnp.arange(100)[:, None].repeat(30, axis=1)
      >>> mask = jnp.ones(100)
      >>> node_features, updated_edges = encoder(edge_features, neighbor_indices, mask)

    """
    from prxteinmpnn.utils.concatenate import concatenate_neighbor_nodes  # noqa: PLC0415
    from prxteinmpnn.utils.gelu import GeLU  # noqa: PLC0415

    # Initialize node features
    node_features = jnp.zeros((edge_features.shape[0], self.node_feature_dim))

    # Process through encoder layers
    for layer in self.layers:
      # Encode: concatenate neighbor nodes and compute edge messages
      edge_features_cat = concatenate_neighbor_nodes(
        node_features,
        edge_features,
        neighbor_indices,
      )
      node_features_expand = jnp.tile(
        jnp.expand_dims(node_features, -2),
        [1, edge_features_cat.shape[-2], 1],
      )
      edge_input = jnp.concatenate([node_features_expand, edge_features_cat], -1)

      # Edge message computation (w1 -> w2 -> w3)
      # Use vmap to apply linear layers across the neighbor dimension
      message = jax.vmap(jax.vmap(lambda x: GeLU(layer.w1(x))))(edge_input)
      message = jax.vmap(jax.vmap(lambda x: GeLU(layer.w2(x))))(message)
      message = jax.vmap(jax.vmap(layer.w3))(message)

      # Update node features with aggregated messages
      node_features = node_features + (jnp.sum(message, -2) / self.scale)
      node_features = jax.vmap(layer.norm1)(node_features)
      node_features = node_features + jax.vmap(layer.dense)(node_features)
      node_features = jax.vmap(layer.norm2)(node_features)
      node_features = mask[:, None] * node_features

      # Update edge features
      edge_features_cat = concatenate_neighbor_nodes(
        node_features,
        edge_features,
        neighbor_indices,
      )
      node_features_expand = jnp.tile(
        jnp.expand_dims(node_features, -2),
        [1, edge_features_cat.shape[-2], 1],
      )
      mlp_input = jnp.concatenate([node_features_expand, edge_features_cat], -1)

      # Edge update (w11 -> w12 -> w13)
      # Use vmap to apply linear layers across the neighbor dimension
      edge_message = jax.vmap(jax.vmap(lambda x: GeLU(layer.w11(x))))(mlp_input)
      edge_message = jax.vmap(jax.vmap(lambda x: GeLU(layer.w12(x))))(edge_message)
      edge_message = jax.vmap(jax.vmap(layer.w13))(edge_message)

      # Apply vmap twice for edges (atoms, neighbors)
      edge_features = jax.vmap(jax.vmap(layer.norm3))(edge_features + edge_message)

    return node_features, edge_features


class Decoder(eqx.Module):
  """Full decoder stack for ProteinMPNN.

  The decoder processes sequence and edge features through multiple decoder layers,
  updating sequence representations for amino acid prediction.

  Attributes:
    layers: Tuple of DecoderLayer modules.
    scale: Scaling factor for message aggregation (default: 30.0).

  """

  layers: tuple[DecoderLayer, ...]
  scale: float = 30.0

  def __init__(
    self,
    sequence_features: int,
    edge_features: int,
    hidden_features: int,
    num_layers: int = 3,
    scale: float = 30.0,
    *,
    key: jax.Array,
  ) -> None:
    """Initialize Decoder.

    Args:
      sequence_features: Sequence feature dimension.
      edge_features: Edge feature dimension.
      hidden_features: Hidden dimension for MLPs.
      num_layers: Number of decoder layers.
      scale: Scaling factor for message aggregation.
      key: PRNG key for initialization.

    Example:
      >>> import jax
      >>> key = jax.random.PRNGKey(0)
      >>> decoder = Decoder(128, 384, 512, num_layers=3, key=key)

    """
    self.scale = scale

    # Create decoder layers (use tuple for hashability with JAX JIT)
    keys = jax.random.split(key, num_layers)
    self.layers = tuple(
      DecoderLayer(sequence_features, edge_features, hidden_features, key=k) for k in keys
    )

  def __call__(
    self,
    node_features: jax.Array,
    edge_features: jax.Array,
    mask: jax.Array,
  ) -> jax.Array:
    """Run decoder forward pass (unconditional mode).

    Args:
      node_features: Node features of shape (num_atoms, sequence_dim).
      edge_features: Raw edge features of shape (num_atoms, num_neighbors, edge_dim).
        These are typically 128-dimensional features from encoder output.
      mask: Atom mask of shape (num_atoms,).

    Returns:
      Updated node features of shape (num_atoms, sequence_dim).

    Example:
      >>> import jax
      >>> import jax.numpy as jnp
      >>> key = jax.random.PRNGKey(0)
      >>> decoder = Decoder(128, 512, 512, num_layers=3, key=key)
      >>> node_features = jax.random.normal(key, (100, 128))
      >>> edge_features = jax.random.normal(key, (100, 30, 128))
      >>> mask = jnp.ones(100)
      >>> updated_features = decoder(node_features, edge_features, mask)

    """
    from prxteinmpnn.utils.gelu import GeLU  # noqa: PLC0415

    # Prepare decoder input features once (unconditional mode)
    # Format: [initial_nodes, zeros, edge_features] -> 128 + 128 + 128 = 384 dims
    nodes_expanded = jnp.tile(
      jnp.expand_dims(node_features, -2),
      [1, edge_features.shape[1], 1],
    )
    zeros_expanded = jnp.tile(
      jnp.expand_dims(jnp.zeros_like(node_features), -2),
      [1, edge_features.shape[1], 1],
    )
    decoder_input_features = jnp.concatenate(
      [nodes_expanded, zeros_expanded, edge_features],
      -1,
    )

    # Process through decoder layers
    for layer in self.layers:
      # Decode message: concatenate current nodes with decoder input
      # Format: [current_nodes, decoder_input] -> 128 + 384 = 512 dims
      node_features_expand = jnp.tile(
        jnp.expand_dims(node_features, -2),
        [1, decoder_input_features.shape[1], 1],
      )
      node_edge_features = jnp.concatenate([node_features_expand, decoder_input_features], -1)

      # Decode message computation (w1 -> w2 -> w3)
      # Use vmap to apply linear layers across the neighbor dimension
      message = jax.vmap(jax.vmap(lambda x: GeLU(layer.w1(x))))(node_edge_features)
      message = jax.vmap(jax.vmap(lambda x: GeLU(layer.w2(x))))(message)
      message = jax.vmap(jax.vmap(layer.w3))(message)

      # Update node features with aggregated messages
      node_features = node_features + (jnp.sum(message, -2) / self.scale)
      node_features = jax.vmap(layer.norm1)(node_features)
      node_features = node_features + jax.vmap(layer.dense)(node_features)
      node_features = jax.vmap(layer.norm2)(node_features)
      node_features = mask[:, None] * node_features

    return node_features


class PrxteinMPNN(eqx.Module):
  """Full ProteinMPNN model combining feature extraction, encoding, decoding, and projection.

  This is the top-level model class that orchestrates the complete forward pass through
  the ProteinMPNN architecture, from structural features to amino acid logits.

  Attributes:
    encoder: The encoder module for processing structural features.
    decoder: The decoder module for sequence generation.
    w_out: Linear projection layer (128 -> 21 amino acids, includes bias).
    b_out: Bias for output projection (stored separately for inspection; already
      included in w_out).
    w_e: Edge embedding matrix (transforms positional encodings).
    w_pos: Positional encoding embedding matrix.
    b_pos: Bias for positional encodings.

  """

  encoder: Encoder
  decoder: Decoder
  w_out: eqx.nn.Linear
  b_out: jnp.ndarray
  w_e: jnp.ndarray
  w_pos: jnp.ndarray
  b_pos: jnp.ndarray

  def __call__(
    self,
    edge_features: jnp.ndarray,
    neighbor_indices: jnp.ndarray,
    mask: jnp.ndarray,
  ) -> jnp.ndarray:
    """Forward pass through the full ProteinMPNN model.

    Args:
      edge_features: Edge features with shape (num_atoms, num_neighbors, edge_dim).
      neighbor_indices: Indices of neighbors with shape (num_atoms, num_neighbors).
      mask: Binary mask for valid atoms with shape (num_atoms,).

    Returns:
      Logits for amino acid predictions with shape (num_atoms, 21).

    """
    # Encode: process structural features through encoder
    node_features, edge_features = self.encoder(edge_features, neighbor_indices, mask)

    # Decode: generate sequence representation
    node_features = self.decoder(node_features, edge_features, mask)

    # Project to amino acid logits (w_out already includes bias)
    return jax.vmap(self.w_out)(node_features)


def load_prxteinmpnn(
  model_path: str,
  num_encoder_layers: int = 3,
  num_decoder_layers: int = 3,
  scale: float = 30.0,
) -> PrxteinMPNN:
  """Load a PrxteinMPNN model from an .eqx file.

  Args:
    model_path: Path to the .eqx file.
    num_encoder_layers: Number of encoder layers.
    num_decoder_layers: Number of decoder layers.
    scale: Scaling factor for message aggregation.

  Returns:
    Loaded PrxteinMPNN model.

  Example:
    >>> model = load_prxteinmpnn("model.eqx")
    >>> # Use model for inference
    >>> logits = model(edge_features, neighbor_indices, mask)

  """
  # Create a template model to get the structure
  from prxteinmpnn.conversion import create_prxteinmpnn  # noqa: PLC0415
  from prxteinmpnn.functional import get_functional_model  # noqa: PLC0415

  model_params = get_functional_model()
  key = jax.random.PRNGKey(0)
  template = create_prxteinmpnn(
    model_params,
    num_encoder_layers=num_encoder_layers,
    num_decoder_layers=num_decoder_layers,
    scale=scale,
    key=key,
  )

  # Load the saved weights into the template
  return eqx.tree_deserialise_leaves(model_path, template)
