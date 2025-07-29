"""Final projection layer for the PrxteinMPNN model."""

import jax
import jax.numpy as jnp

from prxteinmpnn.utils.types import Logits, ModelParameters, NodeFeatures


@jax.jit
def final_projection(
  mpnn_parameters: ModelParameters,
  final_node_features: NodeFeatures,
) -> Logits:
  """Convert node features to logits.

  Args:
    mpnn_parameters: Model parameters for the final projection.
    final_node_features: Node features after the last MPNN layer.

  Returns:
    Logits: The final logits for the model.

  """
  w_out, b_out = (
    mpnn_parameters["protein_mpnn/~/W_out"]["w"],
    mpnn_parameters["protein_mpnn/~/W_out"]["b"],
  )
  return jnp.dot(final_node_features, w_out) + b_out  # + bias
