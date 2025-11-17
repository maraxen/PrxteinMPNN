"""Factory for creating conditional logits functions.

Conditional logits are computed given a specific sequence input,
allowing the model to evaluate how well a sequence fits a structure.

This is used for:
- Jacobian computation (sensitivity analysis)
- Sequence scoring and validation
- Conformational inference
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import equinox as eqx
import jax


def _eqx_module_hash(self: object) -> int:  # pragma: no cover - safe shim
  return id(self)


eqx.Module.__hash__ = _eqx_module_hash
if TYPE_CHECKING:
  from collections.abc import Callable

  from jaxtyping import PRNGKeyArray

  from prxteinmpnn.model import PrxteinMPNN
  from prxteinmpnn.utils.types import (
    AlphaCarbonMask,
    AutoRegressiveMask,
    BackboneNoise,
    ChainIndex,
    Logits,
    ProteinSequence,
    ResidueIndex,
    StructureAtomicCoordinates,
  )

  ConditionalLogitsFn = Callable[
    [
      PRNGKeyArray,
      StructureAtomicCoordinates,
      AlphaCarbonMask,
      ResidueIndex,
      ChainIndex,
      ProteinSequence,
      AutoRegressiveMask | None,
      BackboneNoise | None,
      jax.Array | None,
    ],
    Logits,
  ]
else:
  from collections.abc import Callable
  from typing import Any

  ConditionalLogitsFn = Callable[..., Any]


def make_conditional_logits_fn(
  model: PrxteinMPNN,
) -> ConditionalLogitsFn:
  """Create a function to compute conditional logits for a given sequence.

  Conditional logits evaluate how well a sequence fits a structure by
  running the model with the sequence as input.

  Args:
    model: A PrxteinMPNN Equinox model instance.

  Returns:
    A function that computes conditional logits for sequence-structure pairs.

  Example:
    >>> from prxteinmpnn.io.weights import load_model
    >>> model = load_model()
    >>> logits_fn = make_conditional_logits_fn(model)
    >>> logits = logits_fn(key, coords, mask, res_idx, chain_idx, sequence)

  """

  @partial(jax.jit)
  def conditional_logits(
    prng_key: PRNGKeyArray,
    structure_coordinates: StructureAtomicCoordinates,
    mask: AlphaCarbonMask,
    residue_index: ResidueIndex,
    chain_index: ChainIndex,
    sequence: ProteinSequence,
    ar_mask: AutoRegressiveMask | None = None,
    backbone_noise: BackboneNoise | None = None,
    structure_mapping: jax.Array | None = None,
  ) -> Logits:
    """Compute conditional logits for a sequence-structure pair.

    Args:
      prng_key: JAX random key (unused but kept for API consistency).
      structure_coordinates: Atomic coordinates (N, 4, 3).
      mask: Alpha carbon mask indicating valid residues.
      residue_index: Residue indices.
      chain_index: Chain indices.
      sequence: Protein sequence as integer array (N,) or one-hot (N, 21).
      ar_mask: Optional autoregressive mask (N, N).
      backbone_noise: Optional noise for backbone coordinates.
      structure_mapping: Optional (N,) array mapping each residue to a structure ID.
                        When provided (multi-state mode), prevents cross-structure
                        neighbors to avoid information leakage between conformational states.

    Returns:
      Logits of shape (N, 21) for each residue position.

    Example:
      >>> logits = conditional_logits(
      ...     key, coords, mask, res_idx, chain_idx, sequence
      ... )

    """
    _, logits = model(
      structure_coordinates=structure_coordinates,
      mask=mask,
      residue_index=residue_index,
      chain_index=chain_index,
      decoding_approach="conditional",
      prng_key=prng_key,
      one_hot_sequence=sequence,
      ar_mask=ar_mask,
      backbone_noise=backbone_noise,
      structure_mapping=structure_mapping,
    )

    return logits

  return conditional_logits


def make_encoding_conditional_logits_split_fn(
  model: PrxteinMPNN,
) -> tuple[Callable, Callable]:
  """Create separate encoding and decoding functions for averaged encodings.

  This splits the model into two parts:
  1. Encoding: Structure -> Encoder features (node_features, edge_features, neighbor_indices)
  2. Decoding: (Encoder features, Sequence) -> Logits

  This separation allows:
  - Averaging encoder features across multiple noise levels
  - Efficient jacobian computation by caching encoder output
  - Reusing encoder output for multiple sequence evaluations

  Args:
    model: A PrxteinMPNN Equinox model instance.

  Returns:
    Tuple of (encode_fn, decode_fn) where:
      - encode_fn: Computes encoder features from structure
      - decode_fn: Computes logits from cached features and sequence

  Example:
    >>> encode_fn, decode_fn = make_encoding_conditional_logits_split_fn(model)
    >>> # Encode once
    >>> key = jax.random.key(0)
    >>> encoding = encode_fn(key, coords, mask, res_idx, chain_idx, noise=0.1)
    >>> # Decode multiple sequences using same encoding
    >>> logits1 = decode_fn(encoding, sequence1)
    >>> logits2 = decode_fn(encoding, sequence2)

  """

  def encode_fn(
    structure_coordinates: StructureAtomicCoordinates,
    mask: AlphaCarbonMask,
    residue_index: ResidueIndex,
    chain_index: ChainIndex,
    backbone_noise: BackboneNoise | None = None,
    prng_key: PRNGKeyArray | None = None,
    structure_mapping: jax.Array | None = None,
  ) -> tuple:
    """Encode structure to get encoder features.

    Args:
      structure_coordinates: Atomic coordinates (N, 4, 3).
      mask: Alpha carbon mask indicating valid residues.
      residue_index: Residue indices.
      chain_index: Chain indices.
      backbone_noise: Optional noise for backbone coordinates.
      prng_key: JAX random key for feature extraction.
      structure_mapping: Optional (N,) array mapping each residue to a structure ID.
                        When provided (multi-state mode), prevents cross-structure
                        neighbors to avoid information leakage between conformational states.

    Returns:
      Tuple of (node_features, edge_features, neighbor_indices, mask, ar_mask_placeholder)
      where ar_mask_placeholder is zeros to maintain consistent shape.

    """
    if backbone_noise is None:
      backbone_noise = jax.numpy.array(0.0, dtype=jax.numpy.float32)

    if prng_key is None:
      prng_key = jax.random.PRNGKey(0)

    edge_features, neighbor_indices, initial_node_features, _ = model.features(
      prng_key,
      structure_coordinates,
      mask,
      residue_index,
      chain_index,
      backbone_noise,
      structure_mapping=structure_mapping,
    )

    node_features, processed_edge_features = model.encoder(
      edge_features,
      neighbor_indices,
      mask,
      node_features=initial_node_features,
    )

    ar_mask_placeholder = jax.numpy.zeros((mask.shape[0], mask.shape[0]), dtype=jax.numpy.int32)

    return (node_features, processed_edge_features, neighbor_indices, mask, ar_mask_placeholder)

  def decode_fn(
    encoding: tuple,
    sequence: ProteinSequence,
    ar_mask: AutoRegressiveMask | None = None,
  ) -> Logits:
    """Decode encoder features to logits for a given sequence.

    Args:
      encoding: Tuple of (node_features, edge_features, neighbor_indices, mask, _)
                from encode_fn.
      sequence: Protein sequence as integer array (N,) or one-hot (N, 21).
      ar_mask: Optional autoregressive mask (N, N). If None, uses zeros.

    Returns:
      Logits of shape (N, 21) for each residue position.

    """
    node_features, processed_edge_features, neighbor_indices, mask, _ = encoding

    if ar_mask is None:
      ar_mask = jax.numpy.zeros((mask.shape[0], mask.shape[0]), dtype=jax.numpy.int32)

    if sequence.ndim == 1:
      one_hot_sequence = jax.nn.one_hot(sequence, model.w_s_embed.num_embeddings)
    else:
      one_hot_sequence = sequence

    decoded_node_features = model.decoder.call_conditional(
      node_features,
      processed_edge_features,
      neighbor_indices,
      mask,
      ar_mask,
      one_hot_sequence,
      model.w_s_embed.weight,
    )

    return jax.vmap(model.w_out)(decoded_node_features)

  return encode_fn, decode_fn
