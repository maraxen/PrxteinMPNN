"""Final projection layer for the PrxteinMPNN model."""

import jax
import jax.numpy as jnp

from prxteinmpnn.utils.types import Logits, ModelParameters, NodeFeatures


@jax.jit
def final_projection(
  model_parameters: ModelParameters,
  node_features: NodeFeatures,
) -> Logits:
  """Convert node features to logits.

  Args:
    model_parameters: Model parameters for the final projection.
    node_features: Node features after the last MPNN layer.

  Returns:
    Logits: The final logits for the model.

  """
  w_out, b_out = (
    model_parameters["protein_mpnn/~/W_out"]["w"],
    model_parameters["protein_mpnn/~/W_out"]["b"],
  )
  return jnp.dot(node_features, w_out) + b_out
