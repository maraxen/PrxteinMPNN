"""Factory for creating sequence sampling and optimization functions."""

from functools import partial
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from prxteinmpnn.model.decoder import (
  make_decoder,
)

if TYPE_CHECKING:
  from prxteinmpnn.model.decoding_signatures import (
    RunAutoregressiveDecoderFn,
    RunConditionalDecoderFn,
    RunSTEAutoregressiveDecoderFn,
  )
from prxteinmpnn.model.encoder import make_encoder
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

from .config import SamplingConfig
from .initialize import sampling_encode
from .sampling_step import preload_sampling_step_decoder
from .ste_optimize import make_optimize_sequence_fn

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
    attention_mask_type="cross",
    num_encoder_layers=num_encoder_layers,
  )

  conditional_decoder: RunConditionalDecoderFn = cast(
    "RunConditionalDecoderFn",
    make_decoder(
      model_parameters=model_parameters,
      attention_mask_type="conditional",
      decoding_approach="conditional",
      num_decoder_layers=num_decoder_layers,
    ),
  )
  ste_autoregressive_decoder: RunSTEAutoregressiveDecoderFn = cast(
    "RunSTEAutoregressiveDecoderFn",
    make_decoder(
      model_parameters=model_parameters,
      attention_mask_type=None,
      decoding_approach="ste_autoregressive",
      num_decoder_layers=num_decoder_layers,
    ),
  )
  autoregressive_decoder = cast(
    "RunAutoregressiveDecoderFn",
    make_decoder(
      model_parameters=model_parameters,
      attention_mask_type=None,
      decoding_approach="autoregressive",
      num_decoder_layers=num_decoder_layers,
    ),
  )

  sample_model_pass = sampling_encode(encoder=encoder, decoding_order_fn=decoding_order_fn)

  optimize_seq_fn = make_optimize_sequence_fn(
    ste_autoregressive_decoder=ste_autoregressive_decoder,
    conditional_decoder=conditional_decoder,
    decoding_order_fn=decoding_order_fn,
    model_parameters=model_parameters,
  )

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

    if config.sampling_strategy == "straight_through":
      output_sequence, output_logits = optimize_seq_fn(
        next_rng_key,
        node_features,
        edge_features,
        neighbor_indices,
        mask,
        config.iterations,
        config.learning_rate,
      )
      output_sequence = output_sequence.argmax(axis=-1).astype(jnp.int8)
      return output_sequence, output_logits, decoding_order
    sample_model_pass_fn = partial(
      sample_model_pass,
      model_parameters=model_parameters,  # type: ignore[arg-type]
      structure_coordinates=structure_coordinates,  # type: ignore[arg-type]
      mask=mask,  # type: ignore[arg-type]
      residue_index=residue_index,  # type: ignore[arg-type]
      chain_index=chain_index,  # type: ignore[arg-type]
      k_neighbors=k_neighbors,  # type: ignore[arg-type]
      augment_eps=augment_eps,  # type: ignore[arg-type]
    )
    sample_step = preload_sampling_step_decoder(
      autoregressive_decoder,
      sample_model_pass_fn,
      config,
    )
    _, output_sequence, output_logits = sample_step(prng_key=next_rng_key)
    return output_sequence, output_logits, decoding_order

  if model_inputs:
    structure_coordinates = model_inputs.structure_coordinates
    mask = model_inputs.mask
    residue_index = model_inputs.residue_index
    chain_index = model_inputs.chain_index
    return partial(
      sample_or_optimize_fn,
      structure_coordinates=structure_coordinates,
      mask=mask,
      residue_index=residue_index,
      chain_index=chain_index,
    )

  return sample_or_optimize_fn  # type: ignore[return-value]
