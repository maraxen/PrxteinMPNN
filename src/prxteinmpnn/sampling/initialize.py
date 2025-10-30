"""Sample sequences from a structure using the ProteinMPNN model.

prxteinmpnn.sampling.initialize
"""

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from prxteinmpnn.model.features import extract_features, project_features
from prxteinmpnn.utils.autoregression import generate_ar_mask
from prxteinmpnn.utils.decoding_order import DecodingOrderFn
from prxteinmpnn.utils.types import (
  AlphaCarbonMask,
  AutoRegressiveMask,
  BackboneDihedrals,
  BackboneNoise,
  ChainIndex,
  EdgeFeatures,
  ModelParameters,
  NeighborIndices,
  NodeFeatures,
  ResidueIndex,
  StructureAtomicCoordinates,
)

SamplingModelPassInput = tuple[
  PRNGKeyArray,
  ModelParameters,
  StructureAtomicCoordinates,
  AlphaCarbonMask,
  ResidueIndex,
  ChainIndex,
  BackboneDihedrals | None,
  AutoRegressiveMask | None,
  int,  # k_neighbors
  BackboneNoise | None,
]

SamplingModelPassOutput = tuple[
  NodeFeatures,
  EdgeFeatures,
  NeighborIndices,
  AlphaCarbonMask,
  AutoRegressiveMask,
  PRNGKeyArray,
]

SamplingModelPassFn = Callable[
  [*SamplingModelPassInput],
  SamplingModelPassOutput,
]


def sampling_encode(
  encoder: Callable[..., tuple[NodeFeatures, EdgeFeatures]],
  decoding_order_fn: DecodingOrderFn | None,
) -> SamplingModelPassFn:
  """Create a function to run a single pass through the encoder."""

  @partial(jax.jit, static_argnames=("k_neighbors",))
  def sample_model_pass(
    prng_key: PRNGKeyArray,
    model_parameters: ModelParameters,
    structure_coordinates: StructureAtomicCoordinates,
    mask: AlphaCarbonMask,
    residue_index: ResidueIndex,
    chain_index: ChainIndex,
    dihedrals: BackboneDihedrals | None = None,
    autoregressive_mask: AutoRegressiveMask | None = None,
    k_neighbors: int = 48,
    backbone_noise: BackboneNoise | None = None,
  ) -> SamplingModelPassOutput:
    """Run a single pass through the encoder and decoder to prepare for sampling."""
    if decoding_order_fn is not None:
      decoding_order, next_rng_key = decoding_order_fn(prng_key, structure_coordinates.shape[0])
    else:
      decoding_order, next_rng_key = jnp.arange(structure_coordinates.shape[0]), prng_key

    ar_mask = (
      generate_ar_mask(decoding_order) if autoregressive_mask is None else autoregressive_mask
    )

    edge_features, neighbor_indices, next_rng_key, dihedral_features = extract_features(
      next_rng_key,
      model_parameters,
      structure_coordinates,
      mask,
      residue_index,
      chain_index,
      dihedrals=dihedrals,
      k_neighbors=k_neighbors,
      backbone_noise=backbone_noise,
    )

    edge_features = project_features(model_parameters, edge_features)

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
      dihedral_features,
    )
    return (
      node_features,
      edge_features,
      neighbor_indices,
      mask,
      ar_mask,
      next_rng_key,
    )

  return sample_model_pass
