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
  AtomChainIndex,
  AtomMask,
  AtomResidueIndex,
  AutoRegressiveMask,
  DecodingOrder,
  EdgeFeatures,
  ModelParameters,
  NeighborIndices,
  NodeFeatures,
  StructureAtomicCoordinates,
)

SamplingModelPassInput = tuple[
  PRNGKeyArray,
  StructureAtomicCoordinates,
  AtomMask,
  AtomResidueIndex,
  AtomChainIndex,
  ModelParameters,
  int,  # k_neighbors
  float,  # augment_eps
]

SamplingModelPassOutput = tuple[
  NodeFeatures,
  EdgeFeatures,
  NeighborIndices,
  DecodingOrder,
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
    structure_coordinates: StructureAtomicCoordinates,
    mask: AtomMask,
    residue_indices: AtomResidueIndex,
    chain_indices: AtomChainIndex,
    model_parameters: ModelParameters,
    k_neighbors: int = 48,
    augment_eps: float = 0.0,
  ) -> SamplingModelPassOutput:
    """Run a single pass through the encoder and decoder to prepare for sampling."""
    decoding_order, next_rng_key = decoding_order_fn(prng_key, structure_coordinates.shape[0])

    autoregressive_mask = generate_ar_mask(decoding_order)

    edge_features, neighbor_indices = extract_features(
      structure_coordinates,
      mask,
      residue_indices,
      chain_indices,
      model_parameters,
      prng_key,
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
      decoding_order,
      autoregressive_mask,
      next_rng_key,
    )

  return sample_model_pass
