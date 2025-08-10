"""Factory for creating sequence sampling functions in ProteinMPNN.

prxteinmpnn.sampling.factory
"""

import warnings
from collections.abc import Callable
from dataclasses import asdict
from functools import partial
from typing import cast

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from prxteinmpnn.model.decoder import DecodingEnum, RunConditionalDecoderFn, make_decoder
from prxteinmpnn.model.encoder import make_encoder
from prxteinmpnn.model.masked_attention import MaskedAttentionEnum
from prxteinmpnn.model.projection import final_projection
from prxteinmpnn.utils.data_structures import ModelInputs, SamplingInputs
from prxteinmpnn.utils.decoding_order import DecodingOrderFn
from prxteinmpnn.utils.types import (
  AtomMask,
  ChainIndex,
  DecodingOrder,
  InputBias,
  Logits,
  ModelParameters,
  ProteinSequence,
  ResidueIndex,
  SamplingHyperparameters,
  StructureAtomicCoordinates,
)

from .initialize import sampling_encode
from .sampling_step import SamplingEnum, preload_sampling_step_decoder

SamplerFnBase = Callable[
  [
    PRNGKeyArray,
    ProteinSequence,
    StructureAtomicCoordinates,
    AtomMask,
    ResidueIndex,
    ChainIndex,
    InputBias | None,
    int,
    float,
    SamplingHyperparameters,
    int,
  ],
  tuple[ProteinSequence, Logits, DecodingOrder],
]

SamplerFnFromModelInputs = Callable[
  [
    PRNGKeyArray,
    SamplingHyperparameters,
    int,
  ],
  tuple[ProteinSequence, Logits, DecodingOrder],
]

SamplerFnFromSamplingInputs = Callable[
  [],
  tuple[ProteinSequence, Logits, DecodingOrder],
]


SamplerFn = SamplerFnBase | SamplerFnFromModelInputs | SamplerFnFromSamplingInputs


def make_sample_sequences(
  model_parameters: ModelParameters,
  decoding_order_fn: DecodingOrderFn,
  sampling_strategy: SamplingEnum = SamplingEnum.TEMPERATURE,
  num_encoder_layers: int = 3,
  num_decoder_layers: int = 3,
  model_inputs: ModelInputs | None = None,
  sampling_inputs: SamplingInputs | None = None,
) -> SamplerFn:
  """Create a function to sample sequences from a structure using ProteinMPNN.

  Args:
    model_parameters (ModelParameters): Pre-trained ProteinMPNN model parameters.
    decoding_order_fn (DecodingOrderFn): Function to generate decoding order.
    sampling_strategy (SamplingEnum): Strategy for sampling from logits. Default is temperature
      sampling.
    num_encoder_layers (int): Number of encoder layers. Default is 3.
    num_decoder_layers (int): Number of decoder layers. Default is 3.
    model_inputs (ModelInputs | None): Optional model inputs for sampling. Output function signature
      requires `prng_key`, `bias`, `k_neighbors`, `augment_eps`, `hyperparameters`, and
      `iterations`.
    sampling_inputs (SamplingInputs | None): Optional sampling inputs for sequence sampling. Output
      function signature does not require any arguments, as it uses the attributes of
      `sampling_inputs`.

    If both `model_inputs` and `sampling_inputs` are provided, `sampling_inputs` will be used.

  Returns:
    A function that samples sequences given structural inputs.

  """
  if model_inputs and sampling_inputs:
    warnings.warn(
      "Both model_inputs and sampling_inputs are provided. Using sampling_inputs for sampling.",
      UserWarning,
      stacklevel=2,
    )

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

  @partial(jax.jit, static_argnames=("k_neighbors", "augment_eps"))
  def sample_sequences(  # noqa: PLR0913
    prng_key: PRNGKeyArray,
    sequence: ProteinSequence,
    structure_coordinates: StructureAtomicCoordinates,
    mask: AtomMask,
    residue_index: ResidueIndex,
    chain_index: ChainIndex,
    bias: InputBias | None = None,
    k_neighbors: int = 48,
    augment_eps: float = 0.0,
    hyperparameters: SamplingHyperparameters = (0.0,),
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
      model_parameters,
      structure_coordinates,
      mask,
      residue_index,
      chain_index,
      k_neighbors,
      augment_eps,
    )

    decoded_node_features = decoder(
      node_features,
      edge_features,
      neighbor_indices,
      mask,
      autoregressive_mask,
      sequence,
    )

    logits = final_projection(model_parameters, decoded_node_features) + bias
    sample_step = make_sampling_step_fn(
      model_parameters,
      neighbor_indices,
      mask,
      autoregressive_mask,
      hyperparameters,
    )

    initial_carry = (
      next_rng_key,
      edge_features,
      node_features,
      sequence,
      logits,
    )

    final_carry = jax.lax.fori_loop(
      0,
      iterations,
      sample_step,
      initial_carry,
    )

    return final_carry[3], final_carry[4] + bias, decoding_order

  if sampling_inputs:
    return partial(
      sample_sequences,
      **asdict(sampling_inputs),
    )

  if model_inputs:
    return partial(
      sample_sequences,
      sequence=model_inputs.sequence,
      structure_coordinates=model_inputs.structure_coordinates,
      mask=model_inputs.mask,
      residue_index=model_inputs.residue_index,
      chain_index=model_inputs.chain_index,
      bias=model_inputs.bias,
      k_neighbors=model_inputs.k_neighbors,
      augment_eps=model_inputs.augment_eps,
    )

  return sample_sequences
