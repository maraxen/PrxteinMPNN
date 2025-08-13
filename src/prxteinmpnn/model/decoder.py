"""Decoder module for the PrxteinMPNN model."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

if TYPE_CHECKING:
  from jaxtyping import Array

from prxteinmpnn.utils.gelu import GeLU
from prxteinmpnn.utils.normalize import layer_normalization
from prxteinmpnn.utils.types import (
  AtomMask,
  AttentionMask,
  AutoRegressiveMask,
  EdgeFeatures,
  Logits,
  Message,
  ModelParameters,
  NeighborIndices,
  NodeEdgeFeatures,
  NodeFeatures,
  OneHotProteinSequence,
  ProteinSequence,
  SequenceEdgeFeatures,
)

from .projection import final_projection

if TYPE_CHECKING:
  from jaxtyping import Int


import enum

from prxteinmpnn.utils.concatenate import concatenate_neighbor_nodes

from .dense import dense_layer
from .masked_attention import MaskedAttentionEnum, mask_attention


class DecodingEnum(enum.Enum):
  """Enum for different types of decoders."""

  CONDITIONAL = "conditional"
  UNCONDITIONAL = "unconditional"
  AUTOREGRESSIVE = "autoregressive"


DecodeMessageInputs = tuple[
  NodeFeatures,
  EdgeFeatures,
  ModelParameters,
]
DecodeMessageFn = Callable[[*DecodeMessageInputs], Message]

DecoderNormalizeInputs = tuple[
  Message,
  NodeFeatures,
  AtomMask,
  ModelParameters,
  float,
]
DecoderNormalizeFn = Callable[[*DecoderNormalizeInputs], NodeFeatures]

MaskedAttentionDecoderInputs = tuple[
  NodeFeatures,
  EdgeFeatures,
  AtomMask,
  AttentionMask,
  ModelParameters,
  float,
]
MaskedAttentionDecoderFn = Callable[[*MaskedAttentionDecoderInputs], NodeFeatures]
DecoderInputs = tuple[
  NodeFeatures,
  EdgeFeatures,
  AtomMask,
  ModelParameters,
  float,
]
DecoderFn = Callable[[*DecoderInputs], NodeFeatures]

RunDecoderInputs = tuple[NodeFeatures, EdgeFeatures, AtomMask]
RunDecoderFn = Callable[[*RunDecoderInputs], NodeFeatures]
RunMaskedAttentionDecoderInputs = tuple[
  NodeFeatures,
  EdgeFeatures,
  AtomMask,
  AttentionMask,
]
RunMaskedAttentionDecoderFn = Callable[[*RunMaskedAttentionDecoderInputs], NodeFeatures]
RunConditionalDecoderInputs = tuple[
  NodeFeatures,
  EdgeFeatures,
  NeighborIndices,
  AtomMask,
  AutoRegressiveMask,
  ProteinSequence,
]
RunConditionalDecoderFn = Callable[[*RunConditionalDecoderInputs], NodeFeatures]
RunAutoregressiveDecoderInputs = tuple[
  PRNGKeyArray,
  NodeFeatures,
  EdgeFeatures,
  NeighborIndices,
  AtomMask,
  AutoRegressiveMask,
  float,
]
RunAutoregressiveDecoderFn = Callable[
  [*RunAutoregressiveDecoderInputs],
  tuple[OneHotProteinSequence, Logits],
]


def decoder_parameter_pytree(
  model_parameters: ModelParameters,
  num_decoder_layers: int = 3,
) -> ModelParameters:
  """Make the model weights accessible as a PyTree.

  Args:
    model_parameters: Model parameters for the decoder.
    num_decoder_layers: Number of decoder layers to set up.

  Returns:
    Decoder parameters as a PyTree.

  """
  all_decoder_layer_params_list = []
  for i in range(num_decoder_layers):
    prefix = "protein_mpnn/~/dec_layer"
    if i > 0:
      prefix += f"_{i}"
    layer_name_suffix = f"dec{i}"
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
    }
    all_decoder_layer_params_list.append(layer_params_dict)
  return jax.tree_util.tree_map(lambda *args: jnp.stack(args), *all_decoder_layer_params_list)


def embed_sequence(
  model_parameters: ModelParameters,
  one_hot_sequence: OneHotProteinSequence,
) -> NodeFeatures:
  """Embeds a one-hot encoded sequence."""
  w_s = model_parameters["protein_mpnn/~/embed_positionoken"]["W_s"]
  return one_hot_sequence @ w_s


@jax.jit
def initialize_conditional_decoder(
  one_hot_sequence: OneHotProteinSequence,
  node_features: NodeFeatures,
  edge_features: EdgeFeatures,
  neighbor_indices: NeighborIndices,
  layer_params: ModelParameters,
) -> tuple[NodeEdgeFeatures, SequenceEdgeFeatures]:
  """Initialize the decoder with node and edge features.

  Args:
    one_hot_sequence: One-hot encoded sequence of shape (num_residues, num_classes).
    node_features: Node features of shape (num_atoms, num_features).
    edge_features: EdgeFeatures of shape (num_atoms, num_neighbors, num_features).
    neighbor_indices: Indices of neighboring nodes of shape (num_atoms, num_neighbors).
    layer_params: ModelParameters for the embedding layer.

  Returns:
    A tuple of node-edge features and sequence-edge features.

  """
  embedded_sequence = embed_sequence(layer_params, one_hot_sequence)

  node_edge_features = concatenate_neighbor_nodes(
    jnp.zeros_like(node_features),
    edge_features,
    neighbor_indices,
  )
  node_edge_features = concatenate_neighbor_nodes(
    node_features,
    node_edge_features,
    neighbor_indices,
  )
  sequence_edge_features = concatenate_neighbor_nodes(
    embedded_sequence,
    edge_features,
    neighbor_indices,
  )
  return node_edge_features, sequence_edge_features


@jax.jit
def decode_message(
  node_features: NodeFeatures,
  edge_features: EdgeFeatures,
  layer_params: ModelParameters,
) -> Message:
  """Decode node and edge features into messages.

  Args:
    node_features: Node features of shape (num_atoms, num_features).
    edge_features: Edge features of shape (num_atoms, num_neighbors, num_features).
    layer_params: Model parameters for the encoding layer.

  Returns:
    Message: decoded messages of shape (num_atoms, num_neighbors, num_features).

  """
  node_features_expand = jnp.tile(
    jnp.expand_dims(node_features, -2),
    [1, edge_features.shape[-2], 1],
  )
  node_edge_features = jnp.concatenate([node_features_expand, edge_features], -1)

  w1, b1, w2, b2, w3, b3 = (
    layer_params["W1"]["w"],
    layer_params["W1"]["b"],
    layer_params["W2"]["w"],
    layer_params["W2"]["b"],
    layer_params["W3"]["w"],
    layer_params["W3"]["b"],
  )
  message = GeLU(jnp.dot(GeLU(jnp.dot(node_edge_features, w1) + b1), w2) + b2)
  return jnp.dot(message, w3) + b3


@partial(jax.jit, static_argnames=("scale",))
def decoder_normalize(
  message: Message,
  node_features: NodeFeatures,
  mask: AtomMask,
  layer_params: ModelParameters,
  scale: float = 30.0,
) -> NodeFeatures:
  """Normalize the decoded messages and update node features.

  Args:
    message: decoded messages of shape (num_atoms, num_neighbors, num_features).
    node_features: Node features of shape (num_atoms, num_features).
    mask: Atom mask indicating valid atoms.
    layer_params: Model parameters for the normalization layer.
    scale: Scaling factor for normalization.

  Returns:
    Updated node features after normalization.

  """
  node_features = node_features + (jnp.sum(message, -2) / scale)
  norm1_params = layer_params["norm1"]
  node_features = layer_normalization(node_features, norm1_params)
  node_features = node_features + dense_layer(layer_params, node_features)
  norm2_params = layer_params["norm2"]
  node_features = layer_normalization(node_features, norm2_params)
  return mask[:, None] * node_features


def make_decode_layer(
  attention_mask_enum: MaskedAttentionEnum,
) -> MaskedAttentionDecoderFn | DecoderFn:
  """Create a function to run the decoder with given model parameters."""
  if (
    attention_mask_enum is MaskedAttentionEnum.NONE
    or attention_mask_enum is MaskedAttentionEnum.CROSS
  ):

    @partial(jax.jit, static_argnames=("scale",))
    def decoder_fn(
      node_features: NodeFeatures,
      edge_features: EdgeFeatures,
      mask: AtomMask,
      layer_params: ModelParameters,
      scale: float = 30.0,
    ) -> Message:
      """Run the decoder with the provided edge features and neighbor indices."""
      message = decode_message(node_features, edge_features, layer_params)
      return decoder_normalize(
        message,
        node_features,
        mask,
        layer_params,
        scale,
      )

    return decoder_fn

  @partial(jax.jit, static_argnames=("scale",))
  def masked_attn_decoder_fn(
    node_features: NodeFeatures,
    edge_features: EdgeFeatures,
    mask: AtomMask,
    attention_mask: AttentionMask,
    layer_params: ModelParameters,
    scale: float = 30.0,
  ) -> Message:
    """Run the decoder with the provided edge features and neighbor indices."""
    message = decode_message(node_features, edge_features, layer_params)
    message = mask_attention(message, attention_mask)
    return decoder_normalize(
      message,
      node_features,
      mask,
      layer_params,
      scale,
    )

  return masked_attn_decoder_fn


def setup_decoder(
  model_parameters: ModelParameters,
  attention_mask_enum: MaskedAttentionEnum,
  decoding_enum: DecodingEnum,
  num_decoder_layers: int = 3,
) -> tuple[ModelParameters, Callable[..., Message]]:
  """Set up the decoder parameters and initial node features."""
  all_decoder_layer_params = decoder_parameter_pytree(model_parameters, num_decoder_layers)
  if decoding_enum is DecodingEnum.CONDITIONAL:
    decode_layer_fn = make_decode_layer(attention_mask_enum=MaskedAttentionEnum.CONDITIONAL)
  else:
    decode_layer_fn = make_decode_layer(attention_mask_enum=attention_mask_enum)
  return all_decoder_layer_params, decode_layer_fn


def _check_enums(
  attention_mask_enum: MaskedAttentionEnum,
  decoding_enum: DecodingEnum,
) -> None:
  """Check if the provided enums are valid."""
  if not isinstance(attention_mask_enum, MaskedAttentionEnum):
    msg = f"Unknown attention mask enum: {attention_mask_enum}"
    raise TypeError(msg)
  if not isinstance(decoding_enum, DecodingEnum):
    msg = f"Unknown decoding enum: {decoding_enum}"
    raise TypeError(msg)


def make_decoder(
  model_parameters: ModelParameters,
  attention_mask_enum: MaskedAttentionEnum,
  decoding_enum: DecodingEnum = DecodingEnum.UNCONDITIONAL,
  num_decoder_layers: int = 3,
  scale: float = 30.0,
) -> (
  RunDecoderFn | RunMaskedAttentionDecoderFn | RunAutoregressiveDecoderFn | RunConditionalDecoderFn
):
  """Create a function to run the decoder with given model parameters."""
  _check_enums(
    attention_mask_enum,
    decoding_enum,
  )
  all_decoder_layer_params, decode_layer_fn = setup_decoder(
    model_parameters,
    attention_mask_enum,
    decoding_enum,
    num_decoder_layers,
  )
  if decoding_enum is DecodingEnum.AUTOREGRESSIVE:

    @jax.jit
    def run_autoregressive_decoder(
      prng_key: PRNGKeyArray,
      node_features: NodeFeatures,
      edge_features: EdgeFeatures,
      neighbor_indices: NeighborIndices,
      mask: AtomMask,
      ar_mask: AutoRegressiveMask,
      temperature: float = 1.0,
    ) -> tuple[OneHotProteinSequence, Logits]:
      """Run a full, efficient, local-update autoregressive sampling process."""
      attention_mask = jnp.take_along_axis(ar_mask, neighbor_indices, axis=1)
      mask_1d = mask[:, None]
      mask_bw = mask_1d * attention_mask
      mask_fw = mask_1d * (1 - attention_mask)
      decoding_order = jnp.argsort(jnp.sum(ar_mask, axis=1))

      context_edge_features = concatenate_neighbor_nodes(
        jnp.zeros_like(node_features),
        edge_features,
        neighbor_indices,
      )
      fixed_context_features = concatenate_neighbor_nodes(
        node_features,
        context_edge_features,
        neighbor_indices,
      )
      fixed_context_features = mask_fw[..., None] * fixed_context_features

      def autoregressive_step(
        carry: tuple[NodeFeatures, NodeFeatures, Logits],
        scan_inputs: tuple[Array, PRNGKeyArray],
      ) -> tuple[
        tuple[NodeFeatures, NodeFeatures, Logits],
        tuple[OneHotProteinSequence, Logits],
      ]:
        all_layers_node_features, embedded_sequence_state, all_logits = carry
        position, key = scan_inputs
        fixed_context_features_position = fixed_context_features[position]
        position_neighborhood_indices = neighbor_indices[position]
        mask_position = mask[position]
        mask_bw_position = mask_bw[position]
        edge_features_position = edge_features[position]
        neighbor_sequence_features = embedded_sequence_state[position_neighborhood_indices]
        sequence_edge_features_position = concatenate_neighbor_nodes(
          neighbor_sequence_features,
          edge_features_position,
          position_neighborhood_indices,
        )

        def decoder_layer_loop(
          layer_num: int,
          loop_all_layers_features: NodeFeatures,
        ) -> NodeFeatures:
          layer_input_node_features = loop_all_layers_features[layer_num]

          decoder_input_features_position = concatenate_neighbor_nodes(
            layer_input_node_features,
            sequence_edge_features_position,
            position_neighborhood_indices,
          )
          total_input_features_position = (
            mask_bw_position[..., None] * decoder_input_features_position
          ) + fixed_context_features_position

          layer_params = jax.tree_util.tree_map(lambda x: x[layer_num], all_decoder_layer_params)

          updated_node_features_position = decode_layer_fn(
            layer_input_node_features[position],
            total_input_features_position,
            mask=mask_position,
            layer_params=layer_params,
            attention_mask=mask_bw_position,
            scale=scale,
          )
          return loop_all_layers_features.at[layer_num + 1, position].set(
            updated_node_features_position,
          )

        final_all_layers_node_features = jax.lax.fori_loop(
          0,
          num_decoder_layers,
          decoder_layer_loop,
          all_layers_node_features,
        )
        final_node_features_position = final_all_layers_node_features[-1, position]

        logits_position = final_projection(model_parameters, final_node_features_position)
        sampled_logits = logits_position / temperature + jax.random.gumbel(
          key,
          logits_position.shape,
        )
        sequence_position = jax.nn.one_hot(sampled_logits.argmax(-1), num_classes=21)

        embedded_sequence_position = embed_sequence(model_parameters, sequence_position)
        next_embedded_sequence_state = embedded_sequence_state.at[position].set(
          embedded_sequence_position,
        )
        all_logits = all_logits.at[position].set(sampled_logits)

        next_carry = (final_all_layers_node_features, next_embedded_sequence_state, all_logits)
        outputs_position_collect = (sequence_position, all_logits)
        return next_carry, outputs_position_collect

      num_residues = node_features.shape[0]
      initial_all_layers_node_features = jnp.array(
        [node_features] + [jnp.zeros_like(node_features)] * num_decoder_layers,
      )
      initial_embedded_sequence_state = jnp.zeros_like(
        embed_sequence(
          model_parameters,
          jax.nn.one_hot(jnp.zeros(num_residues, dtype=jnp.int32), num_classes=21),
        ),
      )
      all_logits = jnp.zeros((num_residues, 21), dtype=jnp.float32)
      initial_carry = (
        initial_all_layers_node_features,
        initial_embedded_sequence_state,
        all_logits,
      )

      scan_inputs = (decoding_order, jax.random.split(prng_key, num_residues))

      _, collected_outputs = jax.lax.scan(
        autoregressive_step,
        initial_carry,
        scan_inputs,
      )

      sequence_all, final_logits = collected_outputs
      final_sequence = jnp.zeros_like(sequence_all).at[decoding_order].set(sequence_all)

      return final_sequence, all_logits

    return run_autoregressive_decoder
  if decoding_enum is DecodingEnum.CONDITIONAL:

    @jax.jit
    def run_conditional_decoder(
      node_features: NodeFeatures,
      edge_features: EdgeFeatures,
      neighbor_indices: NeighborIndices,
      mask: AtomMask,
      ar_mask: AutoRegressiveMask,
      one_hot_sequence: OneHotProteinSequence,
    ) -> NodeFeatures:
      """Run the decoder with the provided edge features and neighbor indices."""
      node_edge_features, sequence_edge_features = initialize_conditional_decoder(
        one_hot_sequence,
        node_features,
        edge_features,
        neighbor_indices,
        model_parameters,
      )
      attention_mask = jnp.take_along_axis(
        ar_mask,
        neighbor_indices,
        axis=1,
      )
      mask_bw = mask[:, None] * attention_mask
      mask_fw = mask[:, None] * (1 - attention_mask)
      masked_node_edge_features = mask_fw[..., None] * node_edge_features

      def decoder_loop_body(
        i: Int,
        carry: NodeFeatures,
      ) -> NodeFeatures:
        loop_node_features = carry
        current_layer_params = jax.tree_util.tree_map(lambda x: x[i], all_decoder_layer_params)
        current_features = concatenate_neighbor_nodes(
          loop_node_features,
          sequence_edge_features,
          neighbor_indices,
        )
        loop_edge_features = (mask_bw[..., None] * current_features) + masked_node_edge_features
        return decode_layer_fn(
          loop_node_features,
          loop_edge_features,
          mask,
          attention_mask,
          current_layer_params,
          scale,
        )

      return jax.lax.fori_loop(
        0,
        num_decoder_layers,
        decoder_loop_body,
        node_features,
      )

    return run_conditional_decoder
  if decoding_enum is DecodingEnum.UNCONDITIONAL:
    if attention_mask_enum is MaskedAttentionEnum.NONE:

      @jax.jit
      def run_decoder(
        node_features: NodeFeatures,
        edge_features: EdgeFeatures,
        mask: AtomMask,
      ) -> NodeFeatures:
        """Run the decoder with the provided edge features and neighbor indices."""
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

        def decoder_loop_body(
          i: Int,
          carry: NodeFeatures,
        ) -> NodeFeatures:
          loop_node_features = carry
          current_layer_params = jax.tree_util.tree_map(lambda x: x[i], all_decoder_layer_params)
          return decode_layer_fn(
            loop_node_features,
            decoder_input_features,
            mask,
            current_layer_params,
            scale,
          )

        return jax.lax.fori_loop(
          0,
          num_decoder_layers,
          decoder_loop_body,
          node_features,
        )

      return run_decoder

    @jax.jit
    def run_masked_attention_decoder(
      node_features: NodeFeatures,
      edge_features: EdgeFeatures,
      mask: AtomMask,
      attention_mask: AttentionMask,
    ) -> NodeFeatures:
      """Run the decoder with the provided edge features and neighbor indices."""

      def decoder_loop_body(
        i: Int,
        carry: NodeFeatures,
      ) -> NodeFeatures:
        loop_node_features = carry
        current_layer_params = jax.tree_util.tree_map(lambda x: x[i], all_decoder_layer_params)
        return decode_layer_fn(
          loop_node_features,
          edge_features,
          mask,
          attention_mask,
          current_layer_params,
          scale,
        )

      return jax.lax.fori_loop(
        0,
        num_decoder_layers,
        decoder_loop_body,
        node_features,
      )

    return run_masked_attention_decoder

  msg = f"Unknown decoding enum: {decoding_enum}"
  raise ValueError(msg)
