"""Utility functions for converting between AlphaFold and ProteinMPNN amino acid orders."""

import jax.numpy as jnp
from jaxtyping import Array, Int

ProteinSequence = Int[Array, "N"]  # Type alias for a protein sequence

MPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
AF_ALPHABET = "ARNDCQEGHILKMFPSTWYVX"


def af_to_mpnn(sequence: ProteinSequence) -> ProteinSequence:
  """Convert a tensor from AlphaFold to ProteinMPNN amino acid orders."""
  sequence = jnp.asarray(sequence)
  perm = tuple(AF_ALPHABET.index(k) for k in MPNN_ALPHABET)
  return sequence[..., perm]


def mpnn_to_af(sequence: ProteinSequence) -> ProteinSequence:
  """Convert a tensor from ProteinMPNN to AlphaFold amino acid orders."""
  sequence = jnp.asarray(sequence)
  perm = tuple(MPNN_ALPHABET.index(k) for k in AF_ALPHABET)
  return sequence[..., perm]
