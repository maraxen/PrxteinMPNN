"""Score a given sequence on a structure using the ProteinMPNN model."""

from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp
from jaxtyping import Float, PRNGKeyArray

if TYPE_CHECKING:
  from prxteinmpnn.model.decoding_signatures import RunConditionalDecoderFn

from prxteinmpnn.model.decoder import make_decoder
from prxteinmpnn.model.encoder import make_encoder
from prxteinmpnn.model.features import extract_features, project_features
from prxteinmpnn.model.projection import final_projection
from prxteinmpnn.utils.autoregression import generate_ar_mask
from prxteinmpnn.utils.decoding_order import DecodingOrderFn, random_decoding_order
from prxteinmpnn.utils.types import (
  AlphaCarbonMask,
  AutoRegressiveMask,
  BackboneNoise,
  ChainIndex,
  DecodingOrder,
  Logits,
  ModelParameters,
  OneHotProteinSequence,
  ProteinSequence,
  ResidueIndex,
  StructureAtomicCoordinates,
)

ScoringFn = Callable[
  [
    PRNGKeyArray,
    ProteinSequence,
    StructureAtomicCoordinates,
    AlphaCarbonMask,
    ResidueIndex,
    ChainIndex,
    int,
    BackboneNoise | None,
    AutoRegressiveMask | None,
  ],
  tuple[Float, Logits, DecodingOrder],
]


SCORE_EPS = 1e-8


def make_score_sequence(
  model_parameters: ModelParameters,
  decoding_order_fn: DecodingOrderFn = random_decoding_order,
  num_encoder_layers: int = 3,
  num_decoder_layers: int = 3,
) -> ScoringFn:
  """Create a function to score a sequence on a structure."""
  encoder = make_encoder(
    model_parameters=model_parameters,
    attention_mask_type="cross",
    num_encoder_layers=num_encoder_layers,
  )

  decoder: RunConditionalDecoderFn = cast(
    "RunConditionalDecoderFn",
    make_decoder(
      model_parameters=model_parameters,
      attention_mask_type=None,
      decoding_approach="conditional",
      num_decoder_layers=num_decoder_layers,
    ),
  )

  @partial(jax.jit, static_argnames=("k_neighbors",))
  def score_sequence(
    prng_key: PRNGKeyArray,
    sequence: ProteinSequence | OneHotProteinSequence,
    structure_coordinates: StructureAtomicCoordinates,
    mask: AlphaCarbonMask,
    residue_index: ResidueIndex,
    chain_index: ChainIndex,
    k_neighbors: int = 48,
    backbone_noise: BackboneNoise | None = None,
    ar_mask: AutoRegressiveMask | None = None,
  ) -> tuple[Float, Logits, DecodingOrder]:
    """Score a sequence on a structure using the ProteinMPNN model."""
    decoding_order, prng_key = decoding_order_fn(prng_key, sequence.shape[0])
    autoregressive_mask = generate_ar_mask(decoding_order) if ar_mask is None else ar_mask

    if sequence.ndim == 1:
      sequence = jax.nn.one_hot(sequence, num_classes=21)

    edge_features, neighbor_indices, prng_key = extract_features(
      prng_key,
      model_parameters,
      structure_coordinates,
      mask,
      residue_index,
      chain_index,
      k_neighbors=k_neighbors,
      backbone_noise=backbone_noise,
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
      model_parameters,
      node_features,
    )

    log_probability = jax.nn.log_softmax(logits, axis=-1)[..., :20]

    if sequence.ndim == 1:
      sequence = jax.nn.one_hot(sequence, num_classes=21)

    score = -(sequence[..., :20] * log_probability).sum(-1)
    masked_score_sum = (score * mask).sum(-1)
    mask_sum = mask.sum() + SCORE_EPS

    return masked_score_sum / mask_sum, logits, decoding_order

  return score_sequence
