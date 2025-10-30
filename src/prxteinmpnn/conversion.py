"""Weight conversion helpers for migrating from functional to Equinox models.

This module provides utilities to convert PyTree parameter dictionaries from
the functional API to Equinox module instances.

prxteinmpnn.conversion
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax

from prxteinmpnn.eqx import DecoderLayer, DenseLayer, EncoderLayer, LayerNorm

if TYPE_CHECKING:
  from jaxtyping import Array, PRNGKeyArray

  from prxteinmpnn.utils.types import ModelParameters


def create_linear(w: Array, b: Array) -> eqx.nn.Linear:
  """Create an Equinox Linear layer from weight and bias arrays.

  Args:
    w: Weight matrix of shape (in_features, out_features).
    b: Bias vector of shape (out_features,).

  Returns:
    An eqx.nn.Linear module with the given weights and biases.

  Example:
    >>> import jax.numpy as jnp
    >>> w = jnp.ones((128, 256))
    >>> b = jnp.zeros((256,))
    >>> linear = create_linear(w, b)
    >>> x = jnp.ones((10, 128))
    >>> y = linear(x)
    >>> y.shape
    (10, 256)

  """
  in_features, out_features = w.shape
  # Create a dummy Linear layer to get the structure
  dummy_key = jax.random.PRNGKey(0)
  linear = eqx.nn.Linear(in_features, out_features, key=dummy_key)

  # Replace weights and bias
  linear = eqx.tree_at(lambda m: m.weight, linear, w.T)  # Transpose for Equinox convention
  return eqx.tree_at(lambda m: m.bias, linear, b)


def create_layernorm(
  scale: Array,
  offset: Array,
) -> LayerNorm:
  """Create an Equinox LayerNorm from scale and offset arrays.

  Args:
    scale: Scale parameter (gamma) of shape matching the normalized dimensions.
    offset: Offset parameter (beta) of shape matching the normalized dimensions.

  Returns:
    An eqx.nn.LayerNorm module with the given scale and offset.

  Example:
    >>> import jax.numpy as jnp
    >>> scale = jnp.ones((128,))
    >>> offset = jnp.zeros((128,))
    >>> layer_norm = create_layernorm(scale, offset)
    >>> x = jax.random.normal(jax.random.PRNGKey(0), (10, 128))
    >>> y = layer_norm(x)

  """
  # Create a LayerNorm with the right shape (inferred from scale/offset)
  shape = scale.shape
  layer_norm = eqx.nn.LayerNorm(shape)

  # Replace scale and offset with pretrained weights
  layer_norm = eqx.tree_at(lambda m: m.weight, layer_norm, scale)
  return eqx.tree_at(lambda m: m.bias, layer_norm, offset)


def create_dense(p_dict: ModelParameters, *, key: PRNGKeyArray) -> DenseLayer:
  """Create an Equinox DenseLayer from parameter dictionary.

  Args:
    p_dict: Parameter dictionary containing dense layer weights.
    key: PRNG key for initialization (used for structure, weights replaced).

  Returns:
    A DenseLayer module with weights from the parameter dictionary.

  Example:
    >>> import jax
    >>> import jax.numpy as jnp
    >>> params = {
    ...   "dense_W_in": {"w": jnp.ones((128, 512)), "b": jnp.zeros((512,))},
    ...   "dense_W_out": {"w": jnp.ones((512, 128)), "b": jnp.zeros((128,))},
    ... }
    >>> key = jax.random.PRNGKey(0)
    >>> dense = create_dense(params, "test", key=key)

  """
  w_in = p_dict["dense_W_in"]["w"]
  b_in = p_dict["dense_W_in"]["b"]
  w_out = p_dict["dense_W_out"]["w"]
  b_out = p_dict["dense_W_out"]["b"]

  in_features = w_in.shape[0]
  hidden_features = w_in.shape[1]
  out_features = w_out.shape[1]

  # Create a DenseLayer with the right dimensions
  dense = DenseLayer(in_features, hidden_features, out_features, key=key)

  # Replace the weights
  linear_in = create_linear(w_in, b_in)
  linear_out = create_linear(w_out, b_out)

  dense = eqx.tree_at(lambda m: m.linear_in, dense, linear_in)
  return eqx.tree_at(lambda m: m.linear_out, dense, linear_out)


def create_encoder_layer(p_dict: ModelParameters, *, key: PRNGKeyArray) -> EncoderLayer:
  """Create an Equinox EncoderLayer from parameter dictionary.

  Args:
    p_dict: Parameter dictionary containing encoder layer weights.
      Expected keys: W1, W2, W3, norm1, dense_W_in, dense_W_out, norm2, W11, W12, W13, norm3.
    key: PRNG key for initialization (used for structure, weights replaced).

  Returns:
    An EncoderLayer module with weights from the parameter dictionary.

  Example:
    >>> import jax
    >>> import jax.numpy as jnp
    >>> # Assume p_dict contains all necessary weights
    >>> key = jax.random.PRNGKey(0)
    >>> encoder = create_encoder_layer(p_dict, key=key)

  """
  # Extract weight dimensions from the parameter dictionary
  node_features = p_dict["W3"]["w"].shape[1]  # Output dimension of W3
  edge_features = p_dict["W1"]["w"].shape[0]  # Input dimension of W1
  hidden_features = p_dict["W1"]["w"].shape[1]  # Hidden dimension

  # Create encoder layer with correct dimensions
  encoder = EncoderLayer(node_features, edge_features, hidden_features, key=key)

  # Replace edge message computation weights (W1, W2, W3)
  encoder = eqx.tree_at(
    lambda m: m.w1,
    encoder,
    create_linear(p_dict["W1"]["w"], p_dict["W1"]["b"]),
  )
  encoder = eqx.tree_at(
    lambda m: m.w2,
    encoder,
    create_linear(p_dict["W2"]["w"], p_dict["W2"]["b"]),
  )
  encoder = eqx.tree_at(
    lambda m: m.w3,
    encoder,
    create_linear(p_dict["W3"]["w"], p_dict["W3"]["b"]),
  )

  # Replace normalization layers
  encoder = eqx.tree_at(
    lambda m: m.norm1,
    encoder,
    create_layernorm(p_dict["norm1"]["scale"], p_dict["norm1"]["offset"]),
  )
  encoder = eqx.tree_at(
    lambda m: m.norm2,
    encoder,
    create_layernorm(p_dict["norm2"]["scale"], p_dict["norm2"]["offset"]),
  )
  encoder = eqx.tree_at(
    lambda m: m.norm3,
    encoder,
    create_layernorm(p_dict["norm3"]["scale"], p_dict["norm3"]["offset"]),
  )

  # Replace dense layer
  dense_params = {
    "dense_W_in": p_dict["dense_W_in"],
    "dense_W_out": p_dict["dense_W_out"],
  }
  encoder = eqx.tree_at(lambda m: m.dense, encoder, create_dense(dense_params, key=key))

  # Replace edge update weights (W11, W12, W13)
  encoder = eqx.tree_at(
    lambda m: m.w11,
    encoder,
    create_linear(p_dict["W11"]["w"], p_dict["W11"]["b"]),
  )
  encoder = eqx.tree_at(
    lambda m: m.w12,
    encoder,
    create_linear(p_dict["W12"]["w"], p_dict["W12"]["b"]),
  )
  return eqx.tree_at(
    lambda m: m.w13,
    encoder,
    create_linear(p_dict["W13"]["w"], p_dict["W13"]["b"]),
  )


def create_decoder_layer(p_dict: ModelParameters, *, key: PRNGKeyArray) -> DecoderLayer:
  """Create an Equinox DecoderLayer from parameter dictionary.

  Args:
    p_dict: Parameter dictionary containing decoder layer weights.
      Expected keys: W1, W2, W3, norm1, dense_W_in, dense_W_out, norm2.
    key: PRNG key for initialization (used for structure, weights replaced).

  Returns:
    A DecoderLayer module with weights from the parameter dictionary.

  Example:
    >>> import jax
    >>> import jax.numpy as jnp
    >>> # Assume p_dict contains all necessary weights
    >>> key = jax.random.PRNGKey(0)
    >>> decoder = create_decoder_layer(p_dict, key=key)

  """
  # Extract weight dimensions from the parameter dictionary
  sequence_features = p_dict["W3"]["w"].shape[1]  # Output dimension of W3
  edge_features = p_dict["W1"]["w"].shape[0]  # Input dimension of W1
  hidden_features = p_dict["W1"]["w"].shape[1]  # Hidden dimension

  # Create decoder layer with correct dimensions
  decoder = DecoderLayer(sequence_features, edge_features, hidden_features, key=key)

  # Replace edge message computation weights (W1, W2, W3)
  decoder = eqx.tree_at(
    lambda m: m.w1,
    decoder,
    create_linear(p_dict["W1"]["w"], p_dict["W1"]["b"]),
  )
  decoder = eqx.tree_at(
    lambda m: m.w2,
    decoder,
    create_linear(p_dict["W2"]["w"], p_dict["W2"]["b"]),
  )
  decoder = eqx.tree_at(
    lambda m: m.w3,
    decoder,
    create_linear(p_dict["W3"]["w"], p_dict["W3"]["b"]),
  )

  # Replace normalization layers
  decoder = eqx.tree_at(
    lambda m: m.norm1,
    decoder,
    create_layernorm(p_dict["norm1"]["scale"], p_dict["norm1"]["offset"]),
  )
  decoder = eqx.tree_at(
    lambda m: m.norm2,
    decoder,
    create_layernorm(p_dict["norm2"]["scale"], p_dict["norm2"]["offset"]),
  )

  # Replace dense layer
  dense_params = {
    "dense_W_in": p_dict["dense_W_in"],
    "dense_W_out": p_dict["dense_W_out"],
  }
  return eqx.tree_at(lambda m: m.dense, decoder, create_dense(dense_params, key=key))
