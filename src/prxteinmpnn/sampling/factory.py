"""Factory for creating sequence sampling functions in ProteinMPNN.

prxteinmpnn.sampling.factory
"""

from collections.abc import Callable
from functools import partial
from typing import cast

import jax
import jax.numpy as jnp
from jaxtyping import Float, PRNGKeyArray

from prxteinmpnn.model.decoder import DecodingEnum, RunConditionalDecoderFn, make_decoder
from prxteinmpnn.model.encoder import make_encoder
from prxteinmpnn.model.final_projection import final_projection
from prxteinmpnn.model.masked_attention import MaskedAttentionEnum
from prxteinmpnn.utils.data_structures import SamplingEnum
from prxteinmpnn.utils.decoding_order import DecodingOrderFn
from prxteinmpnn.utils.types import (
  AtomChainIndex,
  AtomMask,
  AtomResidueIndex,
  DecodingOrder,
  InputBias,
  Logits,
  ModelParameters,
  ProteinSequence,
  StructureAtomicCoordinates,
)

from .initialize import sampling_encode
from .sampling import preload_sampling_step_decoder


def make_sample_sequences(
  model_parameters: ModelParameters,
  decoding_order_fn: DecodingOrderFn,
  sampling_strategy: SamplingEnum = SamplingEnum.TEMPERATURE,
  num_encoder_layers: int = 3,
  num_decoder_layers: int = 3,
) -> Callable[
  [
    PRNGKeyArray,
    ProteinSequence,
    StructureAtomicCoordinates,
    AtomMask,
    AtomResidueIndex,
    AtomChainIndex,
    InputBias | None,
    int,  # k_neighbors
    float,  # augment_eps
    float,  # temperature
    int,  # iterations
  ],
  tuple[ProteinSequence, Float],
]:
  """Create a function to sample sequences from a structure using ProteinMPNN.

  Args:
    model_parameters: Pre-trained ProteinMPNN model parameters.
    decoding_order_fn: Function to generate decoding order.
    sampling_strategy: Strategy for sampling from logits.
    num_encoder_layers: Number of encoder layers.
    num_decoder_layers: Number of decoder layers.

  Returns:
    A function that samples sequences given structural inputs.

  """
  encoder = make_encoder(
    model_parameters=model_parameters,
    attention_mask_enum=MaskedAttentionEnum.CROSS,
    num_encoder_layers=num_encoder_layers,
  )

  decoder = make_decoder(
    model_parameters=model_parameters,
    attention_mask_enum=MaskedAttentionEnum.CONDITIONAL,
    decoding_enum=DecodingEnum.CONDITIONAL,
    num_decoder_layers=num_decoder_layers,
  )
  decoder = cast("RunConditionalDecoderFn", decoder)

  sample_model_pass = sampling_encode(
    encoder=encoder,
    decoding_order_fn=decoding_order_fn,
  )
  make_sampling_step_fn = preload_sampling_step_decoder(
    decoder=decoder,
    sampling_strategy=sampling_strategy,
  )

  @partial(jax.jit, static_argnames=("k_neighbors", "iterations"))
  def sample_sequences(  # noqa: PLR0913
    prng_key: PRNGKeyArray,
    initial_sequence: ProteinSequence,
    structure_coordinates: StructureAtomicCoordinates,
    mask: AtomMask,
    residue_indices: AtomResidueIndex,
    chain_indices: AtomChainIndex,
    bias: InputBias | None = None,
    k_neighbors: int = 48,
    augment_eps: float = 0.0,
    temperature: float = 1.0,
    iterations: int = 1,
  ) -> tuple[
    ProteinSequence,
    Logits,
    DecodingOrder,
  ]:
    """Sample sequences from a structure using autoregressive decoding."""
    if bias is None:
      bias = jnp.zeros((structure_coordinates.shape[0], 21), dtype=jnp.float32)

    (
      node_features,
      edge_features,
      neighbor_indices,
      decoding_order,
      autoregressive_mask,
      next_rng_key,
    ) = sample_model_pass(
      prng_key,
      structure_coordinates,
      mask,
      residue_indices,
      chain_indices,
      model_parameters,
      k_neighbors,
      augment_eps,
    )

    decoded_node_features = decoder(
      node_features,
      edge_features,
      neighbor_indices,
      mask,
      autoregressive_mask,
      initial_sequence,
    )

    logits = final_projection(model_parameters, decoded_node_features) + bias
    sample_step = make_sampling_step_fn(
      neighbor_indices,
      mask,
      autoregressive_mask,
      model_parameters,
      temperature,
    )

    initial_carry = (
      next_rng_key,
      edge_features,
      node_features,
      initial_sequence,
      logits,
    )

    final_carry = jax.lax.fori_loop(
      0,
      iterations,
      sample_step,
      initial_carry,
    )

    return final_carry[3], final_carry[4], decoding_order

  return sample_sequences
