"""Utility functions for converting between AlphaFold and ProteinMPNN amino acid orders."""

import jax.numpy as jnp

from prxteinmpnn.utils.types import ProteinSequence

MPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
AF_ALPHABET = "ARNDCQEGHILKMFPSTWYVX"

_AF_TO_MPNN_PERM = jnp.array(
  [MPNN_ALPHABET.index(k) for k in AF_ALPHABET],
)

_MPNN_TO_AF_PERM = jnp.array(
  [AF_ALPHABET.index(k) for k in MPNN_ALPHABET],
)


def af_to_mpnn(sequence: ProteinSequence) -> ProteinSequence:
  """Convert a sequence of integer indices from AlphaFold's to ProteinMPNN's alphabet order."""
  return _AF_TO_MPNN_PERM[sequence].astype(jnp.int8)


def mpnn_to_af(sequence: ProteinSequence) -> ProteinSequence:
  """Convert a sequence of integer indices from ProteinMPNN's to AlphaFold's alphabet order."""
  return _MPNN_TO_AF_PERM[sequence].astype(jnp.int8)
