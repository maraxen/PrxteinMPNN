"""Utility functions for converting between AlphaFold and ProteinMPNN amino acid orders."""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Int

ProteinSequence = Int[Array, "N"]  # Type alias for a protein sequence

MPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
AF_ALPHABET = "ARNDCQEGHILKMFPSTWYVX"


def af_to_mpnn(sequence: ProteinSequence) -> ProteinSequence:
  """Convert a tensor from AlphaFold to ProteinMPNN amino acid orders."""
  sequence = jnp.asarray(sequence)
  perm = tuple(AF_ALPHABET.index(k) for k in MPNN_ALPHABET)
  if jnp.issubdtype(sequence.dtype, jnp.integer):
    sequence = jax.nn.one_hot(sequence, 21)
  if sequence.shape[-1] == len(MPNN_ALPHABET):
    sequence = jnp.pad(sequence, [[0, 0]] * (sequence.ndim - 1) + [[0, 1]])
  return sequence[..., perm]


def mpnn_to_af(sequence: ProteinSequence) -> ProteinSequence:
  """Convert a tensor from ProteinMPNN to AlphaFold amino acid orders."""
  sequence = jnp.asarray(sequence)
  perm = tuple(MPNN_ALPHABET.index(k) for k in AF_ALPHABET)
  return sequence[..., perm]
