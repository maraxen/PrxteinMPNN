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

import jax

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
    ],
    Logits,
  ]


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

    Returns:
      Logits of shape (N, 21) for each residue position.

    Example:
      >>> logits = conditional_logits(
      ...     key, coords, mask, res_idx, chain_idx, sequence
      ... )

    """
    del prng_key  # Not used in conditional mode

    # Run model in conditional mode
    _, logits = model(
      structure_coordinates,
      mask,
      residue_index,
      chain_index,
      decoding_approach="conditional",
      one_hot_sequence=sequence,
      ar_mask=ar_mask,
      backbone_noise=backbone_noise,
    )

    return logits

  return conditional_logits


def make_encoding_conditional_logits_split_fn(
  model: PrxteinMPNN,
) -> tuple[Callable, Callable]:
  """Create separate encoding and decoding functions for jacobian computation.

  This splits the model into two parts:
  1. Encoding: Structure -> Encoder features
  2. Decoding: (Encoder features, Sequence) -> Logits

  This separation allows efficient jacobian computation by caching
  the encoder output and only computing gradients through the decoder.

  Args:
    model: A PrxteinMPNN Equinox model instance.

  Returns:
    Tuple of (encode_fn, decode_fn) where:
      - encode_fn: Computes encoder features from structure
      - decode_fn: Computes logits from features and sequence

  Example:
    >>> encode_fn, decode_fn = make_encoding_conditional_logits_split_fn(model)
    >>> features = encode_fn(coords, mask, res_idx, chain_idx)
    >>> logits = decode_fn(features, sequence)

  Note:
    This requires accessing internal model components (encoder/decoder).
    The current Equinox model structure may need modifications to cleanly
    expose these intermediate representations.

  """
  # NOTE: Split encoding/decoding not yet implemented
  # This requires modifications to the PrxteinMPNN model to expose
  # intermediate encoder outputs. For now, return conditional logits
  # function twice as a placeholder.

  conditional_fn = make_conditional_logits_fn(model)

  def encode_fn(
    structure_coordinates: StructureAtomicCoordinates,
    mask: AlphaCarbonMask,
    residue_index: ResidueIndex,
    chain_index: ChainIndex,
    backbone_noise: BackboneNoise | None = None,
  ) -> tuple:
    """Encode structure to intermediate representation (placeholder).
    
    Returns:
      Tuple of inputs needed for decoding.
      
    """
    # For now, just return the inputs that will be needed
    return (structure_coordinates, mask, residue_index, chain_index, backbone_noise)

  def decode_fn(
    encoding: tuple,
    sequence: ProteinSequence,
    ar_mask: AutoRegressiveMask | None = None,
  ) -> Logits:
    """Decode intermediate representation to logits (placeholder).
    
    Returns:
      Logits for the given sequence.
      
    """
    structure_coordinates, mask, residue_index, chain_index, backbone_noise = encoding
    return conditional_fn(
      None,  # type: ignore[arg-type]
      structure_coordinates,
      mask,
      residue_index,
      chain_index,
      sequence,
      ar_mask,
      backbone_noise,
    )

  return encode_fn, decode_fn

