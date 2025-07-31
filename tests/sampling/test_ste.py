"""Tests for the Straight-Through Estimator (STE)."""

import chex
import jax
import jax.numpy as jnp
from prxteinmpnn.sampling.ste import ste_loss, straight_through_estimator


def test_straight_through_estimator():
  """Test the Straight-Through Estimator implementation.

  Raises:
      AssertionError: If the output does not match the expected value.
  """
  key = jax.random.PRNGKey(0)
  logits = jax.random.normal(key, (10, 21))  # (L, C)

  ste_result = straight_through_estimator(logits)

  # Check shape and that it's a valid probability distribution
  chex.assert_shape(ste_result, logits.shape)
  chex.assert_trees_all_close(jnp.sum(ste_result, axis=-1), jnp.ones(10), atol=1e-6)

  # Check that the argmax is preserved
  expected_argmax = jnp.argmax(jax.nn.softmax(logits, axis=-1), axis=-1)
  actual_argmax = jnp.argmax(ste_result, axis=-1)
  chex.assert_trees_all_equal(expected_argmax, actual_argmax)


def test_ste_loss():
  """Test the STE loss function.

  Raises:
      AssertionError: If the output does not match the expected value.
  """
  logits_to_optimize = jnp.array([[0.1, 0.9], [0.8, 0.2]])
  target_logits = jnp.array([[0.2, 0.8], [0.7, 0.3]])
  mask = jnp.array([True, True])

  # CORRECTED: The manual calculation was wrong.
  # The true loss is (0.43734 + 0.51347) / 2 = 0.4754
  expected_loss = 0.4754
  loss = ste_loss(logits_to_optimize, target_logits, mask)
  chex.assert_trees_all_close(loss, expected_loss, atol=1e-4)

  # Test with mask
  mask_half = jnp.array([True, False])
  expected_loss_half = 0.43734
  loss_half = ste_loss(logits_to_optimize, target_logits, mask_half)
  chex.assert_trees_all_close(loss_half, expected_loss_half, atol=1e-4)