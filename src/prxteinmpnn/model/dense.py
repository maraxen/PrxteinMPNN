"""Dense layer implementation for ProteinMPNN."""

import jax
import jax.numpy as jnp

from prxteinmpnn.utils.gelu import GeLU
from prxteinmpnn.utils.types import (
  ModelParameters,
  NodeFeatures,
)


@jax.jit
def dense_layer(layer_parameters: ModelParameters, node_features: NodeFeatures) -> NodeFeatures:
  """Apply a dense layer to node features."""
  ff_in_params = layer_parameters["dense_W_in"]
  ff_out_params = layer_parameters["dense_W_out"]
  w_in, b_in = ff_in_params["w"], ff_in_params["b"]
  w_out, b_out = ff_out_params["w"], ff_out_params["b"]
  return (
    jnp.dot(
      GeLU(
        jnp.dot(
          node_features,
          w_in,
        )
        + b_in,
      ),
      w_out,
    )
    + b_out
  )
