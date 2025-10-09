"""Sample the logits conditioned on sequence from ProteinMPNN."""

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from prxteinmpnn.model.decoder import make_decoder
from prxteinmpnn.model.encoder import make_encoder
from prxteinmpnn.model.projection import final_projection
from prxteinmpnn.sampling.initialize import sampling_encode
from prxteinmpnn.utils.decoding_order import DecodingOrderFn
from prxteinmpnn.utils.types import (
  AlphaCarbonMask,
  AutoRegressiveMask,
  BackboneNoise,
  ChainIndex,
  EdgeFeatures,
  InputBias,
  Logits,
  ModelParameters,
  NeighborIndices,
  NodeFeatures,
  OneHotProteinSequence,
  ProteinSequence,
  ResidueIndex,
  StructureAtomicCoordinates,
)

if TYPE_CHECKING:
  from prxteinmpnn.model.decoding_signatures import RunConditionalDecoderFn

ConditionalLogitsFn = Callable[
  [
    PRNGKeyArray,
    StructureAtomicCoordinates,
    ProteinSequence,
    AlphaCarbonMask,
    ResidueIndex,
    ChainIndex,
    InputBias | None,
    int,
    BackboneNoise | None,
  ],
  tuple[Logits, NodeFeatures, EdgeFeatures],
]


def make_conditional_logits_fn(
  model_parameters: ModelParameters,
  decoding_order_fn: DecodingOrderFn | None = None,
  num_encoder_layers: int = 3,
  num_decoder_layers: int = 3,
) -> ConditionalLogitsFn:
  """Perform one conditional pass on the model to get the logits for a given sequence.

  This function sets up the ProteinMPNN model, runs a single encoder pass to extract features,
  and then runs a single conditional decoder pass to compute the final logits for a given
  input sequence and structure.

  Args:
    model_parameters: A dictionary of the pre-trained ProteinMPNN model parameters.
    decoding_order_fn: A function that generates the decoding order.
    num_encoder_layers: The number of encoder layers to use. Defaults to 3.
    num_decoder_layers: The number of decoder layers to use. Defaults to 3.

  Returns:
    A function that computes conditional logits for a given sequence and structure.

  """
  encoder = make_encoder(
    model_parameters=model_parameters,
    attention_mask_type="cross",
    num_encoder_layers=num_encoder_layers,
  )

  decoder = make_decoder(
    model_parameters=model_parameters,
    attention_mask_type="conditional",
    decoding_approach="conditional",
    num_decoder_layers=num_decoder_layers,
  )
  decoder = cast("RunConditionalDecoderFn", decoder)

  sample_model_pass = sampling_encode(
    encoder=encoder,
    decoding_order_fn=decoding_order_fn,
  )

  @partial(jax.jit, static_argnames=("k_neighbors",))
  def condition_logits(
    prng_key: PRNGKeyArray,
    structure_coordinates: StructureAtomicCoordinates,
    sequence: ProteinSequence | OneHotProteinSequence,
    mask: AlphaCarbonMask,
    residue_index: ResidueIndex,
    chain_index: ChainIndex,
    bias: InputBias | None = None,
    k_neighbors: int = 48,
    backbone_noise: BackboneNoise | None = None,
  ) -> tuple[Logits, NodeFeatures, EdgeFeatures]:
    if bias is None:
      bias = jnp.zeros((structure_coordinates.shape[0], 21), dtype=jnp.float32)

    if backbone_noise is None:
      backbone_noise = jnp.array(0.0, dtype=jnp.float32)

    autoregressive_mask = (
      1 - jnp.eye(structure_coordinates.shape[0]) if decoding_order_fn is None else None
    )

    (
      node_features,
      edge_features,
      neighbor_indices,
      _,  # decoding_order
      autoregressive_mask,  # autoregressive_mask
      _,  # next_rng_key (not needed for this function)
    ) = sample_model_pass(
      prng_key,
      model_parameters,
      structure_coordinates,
      mask,
      residue_index,
      chain_index,
      autoregressive_mask,
      k_neighbors,
      backbone_noise,
    )
    if sequence.ndim == 1:
      sequence = jax.nn.one_hot(sequence, 21, dtype=jnp.float32)

    decoded_node_features = decoder(
      node_features,
      edge_features,
      neighbor_indices,
      mask,
      autoregressive_mask,
      sequence,
    )

    logits = final_projection(model_parameters, decoded_node_features)

    return logits + bias, decoded_node_features, edge_features

  return condition_logits


def make_encoding_conditional_logits_split_fn(
  model_parameters: ModelParameters,
  decoding_order_fn: DecodingOrderFn | None = None,
  num_encoder_layers: int = 3,
  num_decoder_layers: int = 3,
) -> tuple[Callable, Callable]:
  """Create functions for encoding and decoding, intended for averaging encodings.

  This function returns two functions: `encode` and `condition_logits`.
  `encode` runs the encoder part of the model to get structural features.
  `condition_logits` runs the decoder part on provided features to get logits.
  This separation allows for averaging the results of `encode` from multiple runs
  (e.g., with different noise) before a single `condition_logits` call.

  Args:
    model_parameters: A dictionary of the pre-trained ProteinMPNN model parameters.
    decoding_order_fn: A function that generates the decoding order.
    num_encoder_layers: The number of encoder layers to use. Defaults to 3.
    num_decoder_layers: The number of decoder layers to use. Defaults to 3.

  Returns:
      A tuple containing two functions: (`encode`, `condition_logits`).

  """
  encoder = make_encoder(
    model_parameters=model_parameters,
    attention_mask_type="cross",
    num_encoder_layers=num_encoder_layers,
  )

  decoder = make_decoder(
    model_parameters=model_parameters,
    attention_mask_type="conditional",
    decoding_approach="conditional",
    num_decoder_layers=num_decoder_layers,
  )
  decoder = cast("RunConditionalDecoderFn", decoder)

  sample_model_pass = sampling_encode(
    encoder=encoder,
    decoding_order_fn=decoding_order_fn,
  )

  @partial(jax.jit, static_argnames=("k_neighbors",))
  def encode(
    prng_key: PRNGKeyArray,
    structure_coordinates: StructureAtomicCoordinates,
    mask: AlphaCarbonMask,
    residue_index: ResidueIndex,
    chain_index: ChainIndex,
    k_neighbors: int = 48,
    backbone_noise: BackboneNoise | None = None,
  ) -> tuple[
    NodeFeatures,
    EdgeFeatures,
    NeighborIndices,
    AlphaCarbonMask,
    AutoRegressiveMask | None,
  ]:
    """Encode the structure to get node and edge features."""
    if backbone_noise is None:
      backbone_noise = jnp.array(0.0, dtype=jnp.float32)

    autoregressive_mask = (
      1 - jnp.eye(structure_coordinates.shape[0]) if decoding_order_fn is None else None
    )

    (
      node_features,
      edge_features,
      neighbor_indices,
      _,
      autoregressive_mask,
      _,  # next_rng_key (not needed for this function)
    ) = sample_model_pass(
      prng_key,
      model_parameters,
      structure_coordinates,
      mask,
      residue_index,
      chain_index,
      autoregressive_mask,
      k_neighbors,
      backbone_noise,
    )

    return node_features, edge_features, neighbor_indices, mask, autoregressive_mask

  @jax.jit
  def condition_logits(
    node_features: NodeFeatures,
    edge_features: EdgeFeatures,
    neighbor_indices: NeighborIndices,
    mask: AlphaCarbonMask,
    autoregressive_mask: AutoRegressiveMask,
    sequence: ProteinSequence | OneHotProteinSequence,
    bias: InputBias | None = None,
  ) -> tuple[Logits, NodeFeatures, EdgeFeatures]:
    """Get conditional logits given encoded features and a sequence."""
    if bias is None:
      bias = jnp.zeros((node_features.shape[0], 21), dtype=jnp.float32)

    if sequence.ndim == 1:
      sequence = jax.nn.one_hot(sequence, 21, dtype=jnp.float32)

    decoded_node_features = decoder(
      node_features,
      edge_features,
      neighbor_indices,
      mask,
      autoregressive_mask,
      sequence,
    )

    logits = final_projection(model_parameters, decoded_node_features)

    return logits + bias, decoded_node_features, edge_features

  return encode, condition_logits
