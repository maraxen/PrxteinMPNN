"""Tests for masked_attention.py."""

import os
from pathlib import Path

import chex
import jax.numpy as jnp
import numpy as np
import pytest

from prxteinmpnn.model.masked_attention import MaskedAttentionType, mask_attention


@pytest.fixture
def message():
  """Fixture for a sample message tensor.

  Returns:
    jnp.ndarray: A sample message tensor of shape (2, 3, 4).

  """
  return jnp.arange(24).reshape(2, 3, 4).astype(jnp.float32)


@pytest.fixture
def attention_mask():
  """Fixture for a sample attention mask.

  Returns:
    jnp.ndarray: A sample attention mask of shape (2, 3).

  """
  return jnp.array([[1, 0, 1], [0, 1, 1]], dtype=jnp.float32)


def test_mask_attention_shapes(message, attention_mask):
  """Test that mask_attention returns the correct shape.

  Args:
    message: Fixture for the message tensor.
    attention_mask: Fixture for the attention mask.

  Returns:
    None

  Raises:
    AssertionError: If the output shape is incorrect.

  Example:
    >>> test_mask_attention_shapes(message, attention_mask)

  """
  masked = mask_attention(message, attention_mask)
  assert masked.shape == message.shape, f"Expected shape {message.shape}, got {masked.shape}"


def test_mask_attention_values(message, attention_mask):
  """Test that mask_attention correctly applies the mask.

  Args:
    message: Fixture for the message tensor.
    attention_mask: Fixture for the attention mask.

  Returns:
    None

  Raises:
    AssertionError: If the masked values are incorrect.

  Example:
    >>> test_mask_attention_values(message, attention_mask)

  """
  masked = mask_attention(message, attention_mask)
  expected = jnp.expand_dims(attention_mask, -1) * message
  assert jnp.allclose(masked, expected), (
    f"Masked output does not match expected values.\nExpected:\n{expected}\nGot:\n{masked}"
  )


def test_mask_attention_zero_mask(message):
  """Test that a zero mask returns all zeros.

  Args:
    message: Fixture for the message tensor.

  Returns:
    None

  Raises:
    AssertionError: If the output is not all zeros.

  Example:
    >>> test_mask_attention_zero_mask(message)

  """
  zero_mask = jnp.zeros((2, 3), dtype=jnp.float32)
  masked = mask_attention(message, zero_mask)
  assert jnp.all(masked == 0), "Masked output should be all zeros when mask is zero."


def test_mask_attention_one_mask(message):
  """Test that a mask of all ones returns the original message.

  Args:
    message: Fixture for the message tensor.

  Returns:
    None

  Raises:
    AssertionError: If the output does not match the input message.

  Example:
    >>> test_mask_attention_one_mask(message)

  """
  one_mask = jnp.ones((2, 3), dtype=jnp.float32)
  masked = mask_attention(message, one_mask)
  assert jnp.allclose(masked, message), (
    "Masked output should match the original message when mask is all ones."
  )


def test_mask_attention_with_golden(message, attention_mask):
    """
    Test the mask_attention function against a golden file.
    If the golden file doesn't exist, it will be created.
    """
    golden_file = Path(__file__).parent / "golden_files" / "masked_attention_golden.npz"

    # Run the function to get the actual output
    actual_masked_output = mask_attention(message, attention_mask)

    if not golden_file.exists():
        os.makedirs(golden_file.parent, exist_ok=True)
        np.savez(golden_file, masked_output=actual_masked_output)
        pytest.skip(f"Golden file created at {golden_file}. Please re-run the tests.")

    # Load the golden data
    golden_data = np.load(golden_file)
    expected_masked_output = golden_data["masked_output"]

    # Compare the results
    chex.assert_trees_all_close(actual_masked_output, expected_masked_output, atol=1e-6, rtol=1e-6)
