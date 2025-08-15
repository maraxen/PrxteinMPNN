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
  AtomMask,
  AutoRegressiveMask,
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
  StructureAtomicCoordinates,
  AtomMask,
  ResidueIndex,
  ChainIndex,
  ModelParameters,
  int,  # k_neighbors
  float,  # augment_eps
]

SamplingModelPassOutput = tuple[
  NodeFeatures,
  EdgeFeatures,
  NeighborIndices,
  AtomMask,
  AutoRegressiveMask,
  PRNGKeyArray,
]

SamplingModelPassFn = Callable[
  [*SamplingModelPassInput],
  SamplingModelPassOutput,
]


def sampling_encode(
  encoder: Callable[..., tuple[NodeFeatures, EdgeFeatures]],
  decoding_order_fn: DecodingOrderFn,
) -> SamplingModelPassFn:
  """Create a function to run a single pass through the encoder."""

  @partial(jax.jit, static_argnames=("k_neighbors", "augment_eps"))
  def sample_model_pass(
    prng_key: PRNGKeyArray,
    model_parameters: ModelParameters,
    structure_coordinates: StructureAtomicCoordinates,
    mask: AtomMask,
    residue_index: ResidueIndex,
    chain_index: ChainIndex,
    k_neighbors: int = 48,
    augment_eps: float = 0.0,
  ) -> SamplingModelPassOutput:
    """Run a single pass through the encoder and decoder to prepare for sampling."""
    decoding_order, next_rng_key = decoding_order_fn(prng_key, structure_coordinates.shape[0])

    autoregressive_mask = generate_ar_mask(decoding_order)

    edge_features, neighbor_indices, next_rng_key = extract_features(
      next_rng_key,
      model_parameters,
      structure_coordinates,
      mask,
      residue_index,
      chain_index,
      k_neighbors=k_neighbors,
      augment_eps=augment_eps,
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
    )
    return (
      node_features,
      edge_features,
      neighbor_indices,
      mask,
      autoregressive_mask,
      next_rng_key,
    )

  return sample_model_pass
