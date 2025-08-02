"""Score a given sequence on a structure using the ProteinMPNN model."""

from collections.abc import Callable
from functools import partial
from typing import cast

import jax
import jax.numpy as jnp
from jaxtyping import Float, PRNGKeyArray

from prxteinmpnn.model.decoder import DecodingEnum, RunConditionalDecoderFn, make_decoder
from prxteinmpnn.model.encoder import make_encoder
from prxteinmpnn.model.features import extract_features, project_features
from prxteinmpnn.model.final_projection import final_projection
from prxteinmpnn.model.masked_attention import MaskedAttentionEnum
from prxteinmpnn.utils.autoregression import generate_ar_mask
from prxteinmpnn.utils.data_structures import ModelInputs
from prxteinmpnn.utils.decoding_order import DecodingOrderFn
from prxteinmpnn.utils.types import (
  AtomMask,
  ChainIndex,
  DecodingOrder,
  ModelParameters,
  ProteinSequence,
  ResidueIndex,
  StructureAtomicCoordinates,
)

ScoringFnBase = Callable[
  [
    PRNGKeyArray,
    ProteinSequence,
    DecodingOrder,
    ModelParameters,
    StructureAtomicCoordinates,
    AtomMask,
    ResidueIndex,
    ChainIndex,
    int,
    float,
  ],
  Float,
]

ScoringFnFromModelInputs = Callable[
  [
    PRNGKeyArray,
    ProteinSequence,
  ],
  Float,
]

ScoringFn = ScoringFnBase | ScoringFnFromModelInputs


def make_score_sequence(
  model_parameters: ModelParameters,
  decoding_order_fn: DecodingOrderFn,
  num_encoder_layers: int = 3,
  num_decoder_layers: int = 3,
  model_inputs: ModelInputs | None = None,
) -> ScoringFn:
  """Create a function to score a sequence on a structure."""
  encoder = make_encoder(
    model_parameters=model_parameters,
    attention_mask_enum=MaskedAttentionEnum.CROSS,
    num_encoder_layers=num_encoder_layers,
  )

  decoder: RunConditionalDecoderFn = cast(
    "RunConditionalDecoderFn",
    make_decoder(
      model_parameters=model_parameters,
      attention_mask_enum=MaskedAttentionEnum.NONE,
      decoding_enum=DecodingEnum.CONDITIONAL,
      num_decoder_layers=num_decoder_layers,
    ),
  )

  @partial(jax.jit, static_argnames=("k_neighbors", "augment_eps"))
  def score_sequence(
    prng_key: PRNGKeyArray,
    sequence: ProteinSequence,
    structure_coordinates: StructureAtomicCoordinates,
    mask: AtomMask,
    residue_index: ResidueIndex,
    chain_index: ChainIndex,
    k_neighbors: int = 48,
    augment_eps: float = 0.0,  # TODO(mar): maybe move k_neighbors and augment_eps to factory args # noqa: TD003, FIX002, E501
  ) -> Float:
    """Score a sequence on a structure using the ProteinMPNN model."""
    decoding_order, _ = decoding_order_fn(prng_key, sequence.shape[0])
    autoregressive_mask = generate_ar_mask(decoding_order)

    edge_features, neighbor_indices = extract_features(
      model_parameters,
      structure_coordinates,
      mask,
      residue_index,
      chain_index,
      prng_key,
      k_neighbors=k_neighbors,
      augment_eps=augment_eps,
    )

    edge_features = project_features(
      model_parameters,
      edge_features,
    )

    attention_mask = jnp.take_along_axis(
      mask[:, None] * mask[None, :],
      neighbor_indices,
      axis=1,
    )

    node_features, edge_features = encoder(
      edge_features,
      neighbor_indices,
      mask,
      attention_mask,
    )

    node_features = decoder(
      node_features,
      edge_features,
      neighbor_indices,
      mask,
      autoregressive_mask,
      sequence,
    )
    logits = final_projection(
      node_features=node_features,
      sequence=sequence,
      model_parameters=model_parameters,
    )
    return jnp.sum(logits, axis=-1) / logits.shape[-1]  # FIGURE OUT WHAT SHOULD ACTUALLY GO HERE

  if model_inputs:
    return partial(
      score_sequence,
      structure_coordinates=model_inputs.structure_coordinates,
      mask=model_inputs.mask,
      residue_indices=model_inputs.residue_index,
      chain_indices=model_inputs.chain_index,
      k_neighbors=model_inputs.k_neighbors,
      augment_eps=model_inputs.augment_eps,
    )
  return score_sequence
