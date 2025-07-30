"""Tests for normalization utilities."""

import chex
import jax
import jax.numpy as jnp
import pytest
from prxteinmpnn.utils.normalize import layer_normalization, normalize


@pytest.fixture
def setup_data():
  """Provide common data for normalization tests."""
  key = jax.random.PRNGKey(0)
  x = jax.random.normal(key, (4, 8, 16))
  scale = jnp.ones((16,))
  offset = jnp.zeros((16,))
  return x, scale, offset


def test_normalize(setup_data):
  """Test the functional layer normalization implementation.

  Raises:
      AssertionError: If mean is not ~0 or variance is not ~1.
  """
  x, scale, offset = setup_data
  normalized_x = normalize(x, scale, offset, axis=-1)

  chex.assert_shape(x, normalized_x.shape)
  # Manually check mean and variance
  mean = jnp.mean(normalized_x, axis=-1)
  variance = jnp.var(normalized_x, axis=-1)
  chex.assert_trees_all_close(mean, jnp.zeros_like(mean), atol=1e-6)
  chex.assert_trees_all_close(variance, jnp.ones_like(variance), atol=1e-4)


def test_normalize_with_scale_offset(setup_data):
  """Test normalization with non-trivial scale and offset.

  Raises:
      AssertionError: If mean or variance are not as expected.
  """
  x, _, _ = setup_data
  scale = jnp.full((16,), 2.0)
  offset = jnp.full((16,), 0.5)
  normalized_x = normalize(x, scale, offset, axis=-1)

  # Manually check mean and variance
  mean = jnp.mean(normalized_x, axis=-1)
  variance = jnp.var(normalized_x, axis=-1)
  chex.assert_trees_all_close(mean, jnp.full_like(mean, 0.5), atol=1e-6)
  chex.assert_trees_all_close(variance, jnp.full_like(variance, 4.0), atol=1e-4, rtol=5e-5)


def test_layer_normalization(setup_data):
  """Test the layer_normalization wrapper function.

  Raises:
      AssertionError: If wrapper output differs from direct call.
  """
  x, scale, offset = setup_data
  layer_params = {"norm": {"scale": scale, "offset": offset}}

  wrapped_result = layer_normalization(x, layer_params, axis=-1)
  direct_result = normalize(x, scale, offset, axis=-1)

  chex.assert_trees_all_close(wrapped_result, direct_result)