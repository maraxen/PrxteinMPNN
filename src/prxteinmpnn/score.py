"""Score a given sequence on a structure using the ProteinMPNN model."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Float, PRNGKeyArray

from prxteinmpnn.model.decoder import DecodingEnum, make_decoder
from prxteinmpnn.model.encoder import make_encoder
from prxteinmpnn.model.features import extract_features, project_features
from prxteinmpnn.model.final_projection import final_projection
from prxteinmpnn.model.masked_attention import MaskedAttentionEnum
from prxteinmpnn.utils.autoregression import generate_ar_mask
from prxteinmpnn.utils.types import (
  AtomChainIndex,
  AtomMask,
  AtomResidueIndex,
  DecodingOrder,
  ModelParameters,
  Sequence,
  StructureAtomicCoordinates,
)


def make_score_sequence(
  model_parameters: ModelParameters,
) -> Callable[
  [
    Sequence,
    DecodingOrder,
    ModelParameters,
    StructureAtomicCoordinates,
    AtomMask,
    AtomResidueIndex,
    AtomChainIndex,
    PRNGKeyArray,
    int,
    float,
  ],
  Float,
]:
  """Create a function to score a sequence on a structure."""
  encoder = make_encoder(
    model_parameters=model_parameters,
    attention_mask_enum=MaskedAttentionEnum.CROSS,
    num_encoder_layers=3,
  )

  decoder = make_decoder(
    model_parameters=model_parameters,
    attention_mask_enum=MaskedAttentionEnum.NONE,
    decoding_enum=DecodingEnum.CONDITIONAL,
    num_decoder_layers=3,
  )

  @jax.jit
  def score_sequence(
    sequence: Sequence,
    decoding_order: DecodingOrder,
    mpnn_parameters: ModelParameters,
    structure_coordinates: StructureAtomicCoordinates,
    mask: AtomMask,
    residue_indices: AtomResidueIndex,
    chain_indices: AtomChainIndex,
    prng_key: PRNGKeyArray,
    k_neighbors: int = 48,
    augment_eps: float = 0.0,
  ) -> Float:
    autoregressive_mask = generate_ar_mask(decoding_order)

    edge_features, neighbor_indices = extract_features(
      structure_coordinates,
      autoregressive_mask,
      residue_indices,
      chain_indices,
      mask,
      mpnn_parameters,
      prng_key,
      k_neighbors=k_neighbors,
      augment_eps=augment_eps,
    )

    edge_features = project_features(
      edge_features,
      mpnn_parameters,
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
      node_features=node_features,
      edge_features=edge_features,
      neighbor_indices=neighbor_indices,
      mask=mask,
      ar_mask=autoregressive_mask,
      sequence=sequence,
    )
    logits = final_projection(
      node_features=node_features,
      sequence=sequence,
      model_parameters=mpnn_parameters,
    )
    return jnp.sum(logits, axis=-1) / logits.shape[-1]  # FIGURE OUT WHAT SHOULD ACTUALLY GO HERE

  return score_sequence
