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
  one_hot = aa_convert.af_to_mpnn(seq)
  # Should be one-hot, permuted to MPNN order
  perm = [aa_convert.AF_ALPHABET.index(k) for k in aa_convert.MPNN_ALPHABET]
  expected = jnp.eye(21)[seq][..., perm]
  assert jnp.allclose(one_hot, expected), f"Expected {expected}, got {one_hot}"


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
  mpnn_one_hot = jnp.eye(len(aa_convert.MPNN_ALPHABET), dtype=jnp.float32)
  af_one_hot = aa_convert.mpnn_to_af(mpnn_one_hot)
  mpnn_one_hot_back = aa_convert.af_to_mpnn(af_one_hot)
  assert jnp.allclose(mpnn_one_hot_back, mpnn_one_hot), (
    f"Expected {mpnn_one_hot}, got {mpnn_one_hot_back}"
  )


def test_af_to_mpnn_pad_behavior():
  """Test af_to_mpnn pads correctly when input is missing last column.

  Args:
    None
  Returns:
    None
  Raises:
    AssertionError: If the output shape is not as expected after padding.

  Example:
    >>> test_af_to_mpnn_pad_behavior()

  """
  # Simulate input with shape (..., 20) instead of 21
  arr = jnp.ones((5, 20))
  result = aa_convert.af_to_mpnn(arr)
  assert result.shape[-1] == 21, f"Expected last dimension 21, got {result.shape[-1]}"
