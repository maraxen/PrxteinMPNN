"""Encoder module (functional legacy API).

prxteinmpnn.functional.encoder
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from prxteinmpnn.model.masked_attention import MaskedAttentionType, mask_attention
from prxteinmpnn.utils.concatenate import concatenate_neighbor_nodes
from prxteinmpnn.utils.gelu import GeLU

from .dense import dense_layer
from .normalize import layer_normalization

if TYPE_CHECKING:
  from collections.abc import Callable

  from jaxtyping import Int

  from prxteinmpnn.utils.types import (
    AlphaCarbonMask,
    AttentionMask,
    EdgeFeatures,
    Message,
    ModelParameters,
    NeighborIndices,
    NodeFeatures,
  )


def encoder_parameter_pytree(
  model_parameters: ModelParameters,
  num_encoder_layers: int = 3,
) -> ModelParameters:
  """Make the model weights accessible as a PyTree.

  Args:
    model_parameters: Model parameters for the encoder.
    edge_features: Edge features to initialize the node features.
    num_encoder_layers: Number of encoder layers to set up.

  Returns:
    tuple: A tuple containing the encoder parameters as a PyTree and the initial node features.

  """
  all_encoder_layer_params_list = []
  for i in range(num_encoder_layers):
    prefix = "protein_mpnn/~/enc_layer"
    if i > 0:
      prefix += f"_{i}"
    layer_name_suffix = f"enc{i}"
    layer_params_dict = {
      "W1": model_parameters[f"{prefix}/~/{layer_name_suffix}_W1"],
      "W2": model_parameters[f"{prefix}/~/{layer_name_suffix}_W2"],
      "W3": model_parameters[f"{prefix}/~/{layer_name_suffix}_W3"],
      "norm1": model_parameters[f"{prefix}/~/{layer_name_suffix}_norm1"],
      "dense_W_in": model_parameters[
        f"{prefix}/~/position_wise_feed_forward/~/{layer_name_suffix}_dense_W_in"
      ],
      "dense_W_out": model_parameters[
        f"{prefix}/~/position_wise_feed_forward/~/{layer_name_suffix}_dense_W_out"
      ],
      "norm2": model_parameters[f"{prefix}/~/{layer_name_suffix}_norm2"],
      "W11": model_parameters[f"{prefix}/~/{layer_name_suffix}_W11"],
      "W12": model_parameters[f"{prefix}/~/{layer_name_suffix}_W12"],
      "W13": model_parameters[f"{prefix}/~/{layer_name_suffix}_W13"],
      "norm3": model_parameters[f"{prefix}/~/{layer_name_suffix}_norm3"],
    }
    all_encoder_layer_params_list.append(layer_params_dict)
  return jax.tree_util.tree_map(
    lambda *args: jnp.stack(args),
    *all_encoder_layer_params_list,
  )


@jax.jit
def encode(
  node_features: NodeFeatures,
  edge_features: EdgeFeatures,
  neighbor_indices: NeighborIndices,
  layer_params: ModelParameters,
) -> Message:
  """Encode node and edge features into messages.

  Args:
    node_features: Node features of shape (num_atoms, num_features).
    edge_features: Edge features of shape (num_atoms, num_neighbors, num_features).
    neighbor_indices: Indices of neighboring nodes of shape (num_atoms, num_neighbors).
    layer_params: Model parameters for the encoding layer.

  Returns:
    Message: Encoded messages of shape (num_atoms, num_neighbors, num_features).

  """
  edge_features = concatenate_neighbor_nodes(node_features, edge_features, neighbor_indices)
  node_features_expand = jnp.tile(
    jnp.expand_dims(node_features, -2),
    [1, edge_features.shape[-2], 1],
  )
  edge_features = jnp.concatenate([node_features_expand, edge_features], -1)

  w1, b1, w2, b2, w3, b3 = (
    layer_params["W1"]["w"],
    layer_params["W1"]["b"],
    layer_params["W2"]["w"],
    layer_params["W2"]["b"],
    layer_params["W3"]["w"],
    layer_params["W3"]["b"],
  )

  message = GeLU(jnp.dot(GeLU(jnp.dot(edge_features, w1) + b1), w2) + b2)
  return jnp.dot(message, w3) + b3


@jax.jit
def encoder_normalize(
  message: Message,
  node_features: NodeFeatures,
  edge_features: EdgeFeatures,
  neighbor_indices: NeighborIndices,
  mask: AlphaCarbonMask,
  layer_params: ModelParameters,
  scale: float = 30.0,
) -> tuple[NodeFeatures, EdgeFeatures]:
  """Normalize the encoded messages and update node features.

  Args:
    message: Encoded messages of shape (num_atoms, num_neighbors, num_features).
    node_features: Node features of shape (num_atoms, num_features).
    edge_features: Edge features of shape (num_atoms, num_neighbors, num_features).
    neighbor_indices: Indices of neighboring nodes of shape (num_atoms, num_neighbors).
    mask: Atom mask indicating valid atoms.
    layer_params: Model parameters for the normalization layer.
    scale: Scaling factor for normalization.

  Returns:
    tuple: Updated node features and edge features after normalization.

  """
  node_features = node_features + (jnp.sum(message, -2) / scale)
  norm1_params = layer_params["norm1"]
  node_features = layer_normalization(node_features, norm1_params)
  node_features = node_features + dense_layer(layer_params, node_features)
  norm2_params = layer_params["norm2"]
  node_features = layer_normalization(node_features, norm2_params)
  node_features = mask[:, None] * node_features
  edge_features_cat = concatenate_neighbor_nodes(node_features, edge_features, neighbor_indices)
  node_features_expand = jnp.tile(
    jnp.expand_dims(node_features, -2),
    [1, edge_features_cat.shape[-2], 1],
  )
  mlp_input = jnp.concatenate([node_features_expand, edge_features_cat], -1)

  w11, b11 = layer_params["W11"]["w"], layer_params["W11"]["b"]
  w12, b12 = layer_params["W12"]["w"], layer_params["W12"]["b"]
  w13, b13 = layer_params["W13"]["w"], layer_params["W13"]["b"]

  edge_message = GeLU(jnp.dot(GeLU(jnp.dot(mlp_input, w11) + b11), w12) + b12)
  edge_message = jnp.dot(edge_message, w13) + b13

  norm3_params = layer_params["norm3"]
  updated_edge_features = layer_normalization(edge_features + edge_message, norm3_params)

  return node_features, updated_edge_features


def make_encode_layer(
  attention_mask_type: MaskedAttentionType | None = None,
) -> Callable[..., Message]:
  """Create a function to run the encoder with given model parameters."""
  if attention_mask_type is not None:

    @jax.jit
    def masked_attn_encoder_fn(
      node_features: NodeFeatures,
      edge_features: EdgeFeatures,
      neighbor_indices: NeighborIndices,
      mask: AlphaCarbonMask,
      attention_mask: AttentionMask,
      layer_params: ModelParameters,
      scale: float = 30.0,
    ) -> Message:
      """Run the encoder with the provided edge features and neighbor indices."""
      message = encode(node_features, edge_features, neighbor_indices, layer_params)
      message = mask_attention(message, attention_mask)
      return encoder_normalize(
        message,
        node_features,
        edge_features,
        neighbor_indices,
        mask,
        layer_params,
        scale,
      )

    return masked_attn_encoder_fn

  @jax.jit
  def encoder_fn(
    node_features: NodeFeatures,
    edge_features: EdgeFeatures,
    neighbor_indices: NeighborIndices,
    mask: AlphaCarbonMask,
    layer_params: ModelParameters,
    scale: float = 30.0,
  ) -> Message:
    """Run the encoder with the provided edge features and neighbor indices."""
    message = encode(node_features, edge_features, neighbor_indices, layer_params)
    return encoder_normalize(
      message,
      node_features,
      edge_features,
      neighbor_indices,
      mask,
      layer_params,
      scale,
    )

  return encoder_fn


@jax.jit
def initialize_node_features(
  model_parameters: ModelParameters,
  edge_features: EdgeFeatures,
) -> NodeFeatures:
  """Initialize node features based on model parameters."""
  return jnp.zeros(
    (edge_features.shape[0], model_parameters["protein_mpnn/~/W_e"]["b"].shape[0]),
  )


def setup_encoder(
  model_parameters: ModelParameters,
  attention_mask_type: MaskedAttentionType | None = None,
  num_encoder_layers: int = 3,
) -> tuple[ModelParameters, Callable[..., Message]]:
  """Set up the encoder parameters and initial node features."""
  all_encoder_layer_params = encoder_parameter_pytree(model_parameters, num_encoder_layers)
  encode_layer_fn = make_encode_layer(attention_mask_type=attention_mask_type)
  return all_encoder_layer_params, encode_layer_fn


def make_encoder(
  model_parameters: ModelParameters,
  attention_mask_type: MaskedAttentionType | None = None,
  num_encoder_layers: int = 3,
  scale: float = 30.0,
) -> Callable[..., tuple[NodeFeatures, EdgeFeatures]]:
  """Create a function to run the encoder with given model parameters."""
  all_encoder_layer_params, encode_layer_fn = setup_encoder(
    model_parameters,
    attention_mask_type,
    num_encoder_layers,
  )

  if attention_mask_type is None:

    @jax.jit
    def run_encoder(
      edge_features: EdgeFeatures,
      neighbor_indices: NeighborIndices,
      mask: AlphaCarbonMask,
    ) -> tuple[NodeFeatures, EdgeFeatures]:
      """Run the encoder with the provided edge features and neighbor indices."""
      node_features_encoder = initialize_node_features(model_parameters, edge_features)

      def encoder_loop_body(
        i: Int,
        carry: tuple[NodeFeatures, EdgeFeatures],
      ) -> tuple[NodeFeatures, EdgeFeatures]:
        node_features, edge_features = carry
        current_layer_params = jax.tree_util.tree_map(lambda x: x[i], all_encoder_layer_params)
        node_features, edge_features = encode_layer_fn(
          node_features,
          edge_features,
          neighbor_indices,
          mask,
          current_layer_params,
          scale,
        )
        return (node_features, edge_features)

      return jax.lax.fori_loop(
        0,
        num_encoder_layers,
        encoder_loop_body,
        (node_features_encoder, edge_features),
      )

    return run_encoder

  @jax.jit
  def run_masked_attention_encoder(
    edge_features: EdgeFeatures,
    neighbor_indices: NeighborIndices,
    mask: AlphaCarbonMask,
    attention_mask: AttentionMask,
  ) -> tuple[NodeFeatures, EdgeFeatures]:
    """Run the encoder with the provided edge features and neighbor indices."""
    node_features_encoder = initialize_node_features(model_parameters, edge_features)

    def encoder_loop_body(
      i: Int,
      carry: tuple[NodeFeatures, EdgeFeatures],
    ) -> tuple[NodeFeatures, EdgeFeatures]:
      node_features, edge_features = carry
      current_layer_params = jax.tree_util.tree_map(lambda x: x[i], all_encoder_layer_params)
      node_features, edge_features = encode_layer_fn(
        node_features,
        edge_features,
        neighbor_indices,
        mask,
        attention_mask,
        current_layer_params,
        scale,
      )
      return (node_features, edge_features)

    return jax.lax.fori_loop(
      0,
      num_encoder_layers,
      encoder_loop_body,
      (node_features_encoder, edge_features),
    )

  return run_masked_attention_encoder
