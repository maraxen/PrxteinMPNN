"""Tests for multi-state sampling strategies with tied positions."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
import chex

from prxteinmpnn.model.multi_state_sampling import (
  max_min_over_group_logits,
  min_over_group_logits,
  product_of_probabilities_logits,
)

class TestMultiStateSampling(chex.TestCase):
    def test_min_over_group_logits(self) -> None:
      """Test that min strategy selects the worst-case logit for each amino acid."""
      # Two states with different preferences
      logits = jnp.array([
        [10.0, -5.0, 0.0],  # State 1: strong preference for AA 0
        [-5.0, 8.0, 0.0],   # State 2: strong preference for AA 1
      ])
      group_mask = jnp.array([True, True])

      min_logits = min_over_group_logits(logits, group_mask)
      chex.assert_tree_all_finite(min_logits)

      # Expected: min across both states for each AA
      # AA 0: min(10.0, -5.0) = -5.0
      # AA 1: min(-5.0, 8.0) = -5.0
      # AA 2: min(0.0, 0.0) = 0.0
      expected = jnp.array([[-5.0, -5.0, 0.0]])

      chex.assert_trees_all_close(min_logits, expected)


    def test_product_of_probabilities_logits(self) -> None:
      """Test that product strategy sums logits (multiplies probabilities)."""
      logits = jnp.array([
        [10.0, -5.0, 0.0],
        [8.0, -3.0, 0.0],
      ])
      group_mask = jnp.array([True, True])

      product_logits = product_of_probabilities_logits(logits, group_mask)
      chex.assert_tree_all_finite(product_logits)

      # Expected: sum of logits for each AA
      # AA 0: 10.0 + 8.0 = 18.0
      # AA 1: -5.0 + -3.0 = -8.0
      # AA 2: 0.0 + 0.0 = 0.0
      expected = jnp.array([[18.0, -8.0, 0.0]])

      chex.assert_trees_all_close(product_logits, expected)


    def test_max_min_pure_mean(self) -> None:
      """Test that alpha=0 gives pure mean."""
      logits = jnp.array([
        [10.0, -5.0],
        [8.0, -3.0],
      ])
      group_mask = jnp.array([True, True])

      result = max_min_over_group_logits(logits, group_mask, alpha=0.0)
      chex.assert_tree_all_finite(result)

      # Expected: mean of logits
      # AA 0: (10.0 + 8.0) / 2 = 9.0
      # AA 1: (-5.0 + -3.0) / 2 = -4.0
      expected = jnp.array([[9.0, -4.0]])

      chex.assert_trees_all_close(result, expected)


    def test_max_min_pure_min(self) -> None:
      """Test that alpha=1 gives pure min."""
      logits = jnp.array([
        [10.0, -5.0],
        [8.0, -3.0],
      ])
      group_mask = jnp.array([True, True])

      result = max_min_over_group_logits(logits, group_mask, alpha=1.0)
      chex.assert_tree_all_finite(result)

      # Expected: min of logits
      # AA 0: min(10.0, 8.0) = 8.0
      # AA 1: min(-5.0, -3.0) = -5.0
      expected = jnp.array([[8.0, -5.0]])

      chex.assert_trees_all_close(result, expected)


    def test_max_min_balanced(self) -> None:
      """Test that alpha=0.5 gives balanced combination."""
      logits = jnp.array([
        [10.0, -5.0],
        [8.0, -3.0],
      ])
      group_mask = jnp.array([True, True])

      result = max_min_over_group_logits(logits, group_mask, alpha=0.5)
      chex.assert_tree_all_finite(result)

      # Expected: 0.5 * min + 0.5 * mean
      # AA 0: 0.5 * 8.0 + 0.5 * 9.0 = 8.5
      # AA 1: 0.5 * -5.0 + 0.5 * -4.0 = -4.5
      expected = jnp.array([[8.5, -4.5]])

      chex.assert_trees_all_close(result, expected)


    def test_partial_group_mask(self) -> None:
      """Test that only masked positions contribute to combination."""
      logits = jnp.array([
        [10.0, -5.0],
        [8.0, -3.0],
        [100.0, 100.0],  # This position is NOT in the group
      ])
      group_mask = jnp.array([True, True, False])

      min_logits = min_over_group_logits(logits, group_mask)
      chex.assert_tree_all_finite(min_logits)

      # Expected: min only over first two positions
      # AA 0: min(10.0, 8.0) = 8.0
      # AA 1: min(-5.0, -3.0) = -5.0
      expected = jnp.array([[8.0, -5.0]])

      chex.assert_trees_all_close(min_logits, expected)


    def test_single_position_in_group(self) -> None:
      """Test that single position returns its own logits."""
      logits = jnp.array([[10.0, -5.0, 3.0]])
      group_mask = jnp.array([True])

      min_logits = min_over_group_logits(logits, group_mask)
      chex.assert_tree_all_finite(min_logits)
      product_logits = product_of_probabilities_logits(logits, group_mask)
      chex.assert_tree_all_finite(product_logits)

      # Single position: all strategies should return the same logits
      expected = jnp.array([[10.0, -5.0, 3.0]])

      chex.assert_trees_all_close(min_logits, expected)
      chex.assert_trees_all_close(product_logits, expected)

    @chex.variants(with_jit=True, without_jit=True, with_device=True)
    def test_jit_compatibility(self) -> None:
      """Test that all strategies are JIT-compatible."""
      logits = jnp.array([
        [10.0, -5.0],
        [8.0, -3.0],
      ])
      group_mask = jnp.array([True, True])

      # Compile functions
      min_fn = self.variant(min_over_group_logits)
      product_fn = self.variant(product_of_probabilities_logits)
      max_min_fn = self.variant(lambda l, m: max_min_over_group_logits(l, m, alpha=0.5))

      # Execute JIT-compiled functions
      min_result = min_fn(logits, group_mask)
      chex.assert_tree_all_finite(min_result)
      product_result = product_fn(logits, group_mask)
      chex.assert_tree_all_finite(product_result)
      max_min_result = max_min_fn(logits, group_mask)
      chex.assert_tree_all_finite(max_min_result)

      # Verify results have expected shapes
      chex.assert_shape(min_result, (1, 2))
      chex.assert_shape(product_result, (1, 2))
      chex.assert_shape(max_min_result, (1, 2))


if __name__ == "__main__":
  pytest.main([__file__, "-v"])
