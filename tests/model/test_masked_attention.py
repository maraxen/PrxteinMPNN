"""Tests for masked_attention.py."""

import jax.numpy as jnp
import pytest

from prxteinmpnn.model.masked_attention import MaskedAttentionEnum, mask_attention


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


def test_masked_attention_enum_values():
  """Test the MaskedAttentionEnum values.

  Args:
    None

  Returns:
    None

  Raises:
    AssertionError: If enum values are incorrect.

  Example:
    >>> test_masked_attention_enum_values()

  """
  assert MaskedAttentionEnum.NONE.value == "none"
  assert MaskedAttentionEnum.CROSS.value == "cross"
  assert MaskedAttentionEnum.CONDITIONAL.value == "conditional"
