"""Factory for creating sequence sampling and optimization functions."""

from dataclasses import asdict
from functools import partial

import jax
from jaxtyping import PRNGKeyArray

from prxteinmpnn.model.decoder import (
  DecodingEnum,
  make_decoder,
)
from prxteinmpnn.model.encoder import make_encoder
from prxteinmpnn.model.masked_attention import MaskedAttentionEnum
from prxteinmpnn.utils.data_structures import ModelInputs
from prxteinmpnn.utils.decoding_order import DecodingOrderFn
from prxteinmpnn.utils.types import (
  AtomMask,
  ChainIndex,
  DecodingOrder,
  Logits,
  ModelParameters,
  ProteinSequence,
  ResidueIndex,
  StructureAtomicCoordinates,
)

from .config import SamplingConfig, SamplingEnum
from .initialize import sampling_encode
from .sampling_step import preload_sampling_step_decoder
from .ste_optimize import optimize_sequence

# Simplified type hints
SamplerFn = partial[tuple[ProteinSequence, Logits, DecodingOrder]]


def make_sample_sequences(
  model_parameters: ModelParameters,
  decoding_order_fn: DecodingOrderFn,
  config: SamplingConfig,
  num_encoder_layers: int = 3,
  num_decoder_layers: int = 3,
  model_inputs: ModelInputs | None = None,
) -> SamplerFn:
  """Create a function to sample or optimize sequences from a structure."""
  encoder = make_encoder(
    model_parameters=model_parameters,
    attention_mask_enum=MaskedAttentionEnum.CROSS,
    num_encoder_layers=num_encoder_layers,
  )

  is_optimizing = config.sampling_strategy == SamplingEnum.STRAIGHT_THROUGH

  # Select the appropriate decoder for the task
  decoding_enum = DecodingEnum.CONDITIONAL if is_optimizing else DecodingEnum.AUTOREGRESSIVE
  decoder = make_decoder(
    model_parameters=model_parameters,
    attention_mask_enum=MaskedAttentionEnum.CONDITIONAL
    if is_optimizing
    else MaskedAttentionEnum.NONE,
    decoding_enum=decoding_enum,
    num_decoder_layers=num_decoder_layers,
  )

  sample_model_pass = sampling_encode(encoder=encoder, decoding_order_fn=decoding_order_fn)

  @partial(jax.jit, static_argnames=("k_neighbors", "augment_eps"))
  def sample_or_optimize_fn(
    prng_key: PRNGKeyArray,
    structure_coordinates: StructureAtomicCoordinates,
    mask: AtomMask,
    residue_index: ResidueIndex,
    chain_index: ChainIndex,
    k_neighbors: int = 48,
    augment_eps: float = 0.0,
  ) -> tuple[ProteinSequence, Logits, DecodingOrder]:
    """Dispatches to either the optimization or sampling function."""
    (
      node_features,
      edge_features,
      neighbor_indices,
      decoding_order,
      _,
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

    if is_optimizing:
      output_sequence, output_logits = optimize_sequence(
        prng_key=next_rng_key,
        conditional_decoder=decoder,
        model_parameters=model_parameters,
        node_features=node_features,
        edge_features=edge_features,
        neighbor_indices=neighbor_indices,
        mask=mask,
        num_steps=config.iterations,
        learning_rate=config.learning_rate,
      )
      return output_sequence, output_logits, decoding_order
    # Temperature sampling
    sample_model_pass_fn = partial(
      sample_model_pass,
      model_parameters=model_parameters,
      structure_coordinates=structure_coordinates,
      mask=mask,
      residue_index=residue_index,
      chain_index=chain_index,
      k_neighbors=k_neighbors,
      augment_eps=augment_eps,
    )
    sample_step = preload_sampling_step_decoder(
      decoder,
      sample_model_pass_fn,
      config,
    )
    _, output_sequence, output_logits = sample_step(prng_key=next_rng_key)
    return output_sequence, output_logits, decoding_order

  if model_inputs:
    return partial(sample_or_optimize_fn, **asdict(model_inputs))

  return sample_or_optimize_fn
