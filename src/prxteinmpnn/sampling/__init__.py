"""Sampling utilities for PrXteinMPNN."""

from prxteinmpnn.sampling.conditional_logits import (
    make_conditional_logits_fn,
    make_encoding_conditional_logits_split_fn,
)
from prxteinmpnn.sampling.sample import make_encoding_sampling_split_fn, make_sample_sequences
from prxteinmpnn.sampling.unconditional_logits import make_unconditional_logits_fn
from prxteinmpnn.utils import ste

__all__ = [
    "make_conditional_logits_fn",
    "make_encoding_conditional_logits_split_fn",
    "make_sample_sequences",
    "make_unconditional_logits_fn",
    "sample",
    "ste",
]

def sample(
    prng_key,
    model,
    structure_coordinates,
    mask,
    residue_index,
    chain_index,
    **kwargs,
):
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
