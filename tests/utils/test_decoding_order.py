"""Tests for decoding order utilities."""

import chex
import jax
import jax.numpy as jnp
import pytest
from prxteinmpnn.utils.decoding_order import random_decoding_order


def test_random_decoding_order_properties():
  """Test properties of the random_decoding_order function.

  Raises:
      AssertionError: If output shapes, types, or values are incorrect.
  """
  key = jax.random.PRNGKey(42)
  num_residues = 10

  decoding_order, next_key = random_decoding_order(key, num_residues)

  # Check output shapes
  chex.assert_shape(decoding_order, (num_residues,))
  chex.assert_shape(next_key, key.shape)

  # Check output dtype
  chex.assert_type(decoding_order, jnp.int32)

  # Check that the PRNG key was consumed and changed
  assert not jnp.all(key == next_key), "PRNGKey was not updated."

  # Check that the output is a permutation of the input range
  # The sorted order should be identical to jnp.arange
  sorted_order = jnp.sort(decoding_order)
  expected_order = jnp.arange(num_residues, dtype=jnp.int32)
  chex.assert_trees_all_equal(sorted_order, expected_order)


def test_random_decoding_order_is_deterministic_with_same_key():
  """Test that the same key produces the same decoding order."""
  key = jax.random.PRNGKey(0)
  num_residues = 50

  order1, _ = random_decoding_order(key, num_residues)
  order2, _ = random_decoding_order(key, num_residues)

  chex.assert_trees_all_equal(order1, order2)


@pytest.mark.parametrize("num_residues", [1, 10, 100])
def test_random_decoding_order_with_various_lengths(num_residues):
  """Test the function with different sequence lengths."""
  key = jax.random.PRNGKey(num_residues)
  decoding_order, _ = random_decoding_order(key, num_residues)

  chex.assert_shape(decoding_order, (num_residues,))
  assert jnp.unique(decoding_order).shape[0] == num_residues
  
def test_random_decoding_order_with_zero_length():
  """Test the function with zero-length sequences."""
  key = jax.random.PRNGKey(0)
  decoding_order, _ = random_decoding_order(key, 0)

  chex.assert_shape(decoding_order, (0,))
  chex.assert_trees_all_equal(decoding_order, jnp.array([], dtype=jnp.int32))
  
def test_random_decoding_order_with_negative_length():
  """Test the function with negative sequence lengths."""
  key = jax.random.PRNGKey(0)
  
  with pytest.raises(TypeError):
    random_decoding_order(key, -5)
  
  with pytest.raises(TypeError):
    random_decoding_order(key, -1)
    
def test_random_decoding_order_from_array_shape():
  """Test the function with an array shape input."""
  key = jax.random.PRNGKey(0)
  arr = jnp.zeros((5, 4, 3))  # Example shape
  num_residues = arr.shape[0]

  decoding_order, _ = random_decoding_order(key, num_residues)

  chex.assert_shape(decoding_order, (num_residues,))
  assert jnp.unique(decoding_order).shape[0] == num_residues