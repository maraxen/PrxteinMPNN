"""Unit tests for amino acid conversion functions in the prxteinmpnn.utils.aa_convert module."""

import pytest
import jax.numpy as jnp

from prxteinmpnn.utils import aa_convert


def test_af_to_mpnn_with_integer_sequence():
  """Test af_to_mpnn with integer-encoded AF sequence.

  Args:
    None
  Returns:
    None
  Raises:
    AssertionError: If the output does not match the expected one-hot permutation.

  Example:
    >>> test_af_to_mpnn_with_integer_sequence()

  """
  seq = jnp.array([0, 1, 2, 3, 4])  # A, R, N, D, C in AF_ALPHABET
  af_seq = aa_convert.af_to_mpnn(seq)
  print(af_seq)
  expected = jnp.array([aa_convert.MPNN_ALPHABET.index(k) for k in "ARNDC"])
  print(expected)
  assert jnp.allclose(af_seq, expected), f"Expected {expected}, got {af_seq}"


def test_mpnn_to_af_and_back_roundtrip():
  """Test roundtrip conversion mpnn_to_af followed by af_to_mpnn returns original.

  Args:
    None
  Returns:
    None
  Raises:
    AssertionError: If the roundtrip conversion does not return the original input.

  Example:
    >>> test_mpnn_to_af_and_back_roundtrip()

  """
  mpnn_seq = jnp.array([0, 1, 2, 3, 4])  # A, R, N, D, C in MPNN_ALPHABET
  af_seq = aa_convert.mpnn_to_af(mpnn_seq)
  mpnn_seq_back = aa_convert.af_to_mpnn(af_seq)
  assert jnp.allclose(mpnn_seq_back, mpnn_seq), (
    f"Expected {mpnn_seq}, got {mpnn_seq_back}"
  )
