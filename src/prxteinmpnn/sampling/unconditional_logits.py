"""Factory for creating unconditional logits functions.

Unconditional logits are computed without providing a sequence input,
allowing the model to predict the most likely amino acids at each position
based solely on the structure.

This is used for:
- Straight-through optimization (as the target distribution)
- Baseline sequence scoring
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
    ResidueIndex,
    StructureAtomicCoordinates,
  )

  UnconditionalLogitsFn = Callable[
    [
      PRNGKeyArray,
      StructureAtomicCoordinates,
      AlphaCarbonMask,
      ResidueIndex,
      ChainIndex,
      AutoRegressiveMask | None,
      BackboneNoise | None,
    ],
    Logits,
  ]


def make_unconditional_logits_fn(
  model: PrxteinMPNN,
) -> UnconditionalLogitsFn:
  """Create a function to compute unconditional logits from a structure.

  Unconditional logits are computed without sequence input, predicting
  the most likely amino acids at each position based purely on structure.

  Args:
    model: A PrxteinMPNN Equinox model instance.

  Returns:
    A function that computes unconditional logits from structures.

  Example:
    >>> from prxteinmpnn.io.weights import load_model
    >>> model = load_model()
    >>> logits_fn = make_unconditional_logits_fn(model)
    >>> logits = logits_fn(key, coords, mask, res_idx, chain_idx)

  """

  @partial(jax.jit)
  def unconditional_logits(
    prng_key: PRNGKeyArray,
    structure_coordinates: StructureAtomicCoordinates,
    mask: AlphaCarbonMask,
    residue_index: ResidueIndex,
    chain_index: ChainIndex,
    ar_mask: AutoRegressiveMask | None = None,
    backbone_noise: BackboneNoise | None = None,
  ) -> Logits:
    """Compute unconditional logits for a structure.

    Args:
      prng_key: JAX random key (unused but kept for API consistency).
      structure_coordinates: Atomic coordinates (N, 4, 3).
      mask: Alpha carbon mask indicating valid residues.
      residue_index: Residue indices.
      chain_index: Chain indices.
      ar_mask: Optional autoregressive mask (N, N).
      backbone_noise: Optional noise for backbone coordinates.

    Returns:
      Logits of shape (N, 21) for each residue position.

    Example:
      >>> logits = unconditional_logits(
      ...     key, coords, mask, res_idx, chain_idx
      ... )

    """
    del prng_key  # Not used in unconditional mode

    # Run model in unconditional mode
    _, logits = model(
      structure_coordinates,
      mask,
      residue_index,
      chain_index,
      decoding_approach="unconditional",
      ar_mask=ar_mask,
      backbone_noise=backbone_noise,
    )

    return logits

  return unconditional_logits

