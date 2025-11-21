"""Sampling utilities for PrXteinMPNN."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from prxteinmpnn.sampling.conditional_logits import (
  make_conditional_logits_fn,
  make_encoding_conditional_logits_split_fn,
)
from prxteinmpnn.sampling.sample import make_sample_sequences
from prxteinmpnn.sampling.unconditional_logits import make_unconditional_logits_fn
from prxteinmpnn.utils import ste

if TYPE_CHECKING:
  import jax

  from prxteinmpnn.model import PrxteinMPNN

__all__ = [
  "make_conditional_logits_fn",
  "make_encoding_conditional_logits_split_fn",
  "make_sample_sequences",
  "make_unconditional_logits_fn",
  "sample",
  "ste",
]


def sample(
  prng_key: jax.Array,
  model: PrxteinMPNN,
  structure_coordinates: jax.Array,
  mask: jax.Array,
  residue_index: jax.Array,
  chain_index: jax.Array,
  **kwargs: Any,  # noqa: ANN401
) -> tuple[jax.Array, jax.Array, jax.Array]:
  """Sample sequences from a structure using the default temperature sampler.

  This is a convenience wrapper around `make_sample_sequences`.

  Args:
    prng_key: JAX random key.
    model: A PrxteinMPNN Equinox model instance.
    structure_coordinates: Atomic coordinates (N, 4, 3).
    mask: Alpha carbon mask indicating valid residues.
    residue_index: Residue indices.
    chain_index: Chain indices.
    **kwargs: Additional keyword arguments for the sampler.

  Returns:
    Tuple of (sampled sequence, logits, decoding order).

  """
  sampler = make_sample_sequences(model, sampling_strategy="temperature")
  return sampler(
    prng_key,
    structure_coordinates,
    mask,
    residue_index,
    chain_index,
    **kwargs,
  )
