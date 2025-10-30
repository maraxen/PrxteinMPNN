"""Weight conversion helpers for migrating from functional to Equinox models.

This module provides utilities to convert PyTree parameter dictionaries from
the functional API to Equinox module instances.

prxteinmpnn.conversion
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax

from prxteinmpnn.eqx import DenseLayer, LayerNorm

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
