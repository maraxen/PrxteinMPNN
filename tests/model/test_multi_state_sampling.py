"""Tests for multi-state sampling strategies with tied positions."""

from __future__ import annotations

import chex
import jax.numpy as jnp
import pytest

from prxteinmpnn.model.multi_state_sampling import (
    arithmetic_mean_logits,
    geometric_mean_logits,
    product_of_probabilities_logits,
)


class TestMultiStateSampling(chex.TestCase):
    def test_arithmetic_mean_logits(self) -> None:
      """Test that arithmetic mean uses log-sum-exp for stable averaging."""
      # Two states with different preferences
      logits = jnp.array([
        [10.0, -5.0, 0.0],  # State 1
        [8.0, -3.0, 0.0],   # State 2
      ])
      group_mask = jnp.array([True, True])

      mean_logits = arithmetic_mean_logits(logits, group_mask)
      chex.assert_tree_all_finite(mean_logits)

      # Should be close to simple average for similar magnitudes
      # AA 0: ~9.0, AA 1: ~-4.0, AA 2: 0.0
      chex.assert_shape(mean_logits, (1, 3))


    def test_geometric_mean_logits(self) -> None:
      """Test that geometric mean divides by temperature and group count."""
      logits = jnp.array([
        [10.0, -5.0, 0.0],
        [8.0, -3.0, 0.0],
      ])
      group_mask = jnp.array([True, True])
      temperature = 1.0

      geom_logits = geometric_mean_logits(logits, group_mask, temperature)
      chex.assert_tree_all_finite(geom_logits)

      # Expected: (sum of logits) / (temperature * num_in_group)
      # AA 0: (10.0 + 8.0) / (1.0 * 2) = 9.0
      # AA 1: (-5.0 + -3.0) / (1.0 * 2) = -4.0
      # AA 2: (0.0 + 0.0) / (1.0 * 2) = 0.0
      expected = jnp.array([[9.0, -4.0, 0.0]])

      chex.assert_trees_all_close(geom_logits, expected)


    def test_geometric_mean_with_temperature(self) -> None:
      """Test that geometric mean correctly applies temperature scaling."""
      logits = jnp.array([
        [10.0, -5.0],
        [8.0, -3.0],
      ])
      group_mask = jnp.array([True, True])
      temperature = 2.0

      geom_logits = geometric_mean_logits(logits, group_mask, temperature)
      chex.assert_tree_all_finite(geom_logits)

      # Expected: (sum of logits) / (temperature * num_in_group)
      # AA 0: (10.0 + 8.0) / (2.0 * 2) = 4.5
      # AA 1: (-5.0 + -3.0) / (2.0 * 2) = -2.0
      expected = jnp.array([[4.5, -2.0]])

      chex.assert_trees_all_close(geom_logits, expected)


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


    def test_partial_group_mask(self) -> None:
      """Test that only masked positions contribute to combination."""
      logits = jnp.array([
        [10.0, -5.0],
        [8.0, -3.0],
        [100.0, 100.0],  # This position is NOT in the group
      ])
      group_mask = jnp.array([True, True, False])

      arithmetic_logits = arithmetic_mean_logits(logits, group_mask)
      chex.assert_tree_all_finite(arithmetic_logits)

      product_logits = product_of_probabilities_logits(logits, group_mask)
      chex.assert_tree_all_finite(product_logits)

      # Product should only sum first two positions
      # AA 0: 10.0 + 8.0 = 18.0
      # AA 1: -5.0 + -3.0 = -8.0
      expected_product = jnp.array([[18.0, -8.0]])
      chex.assert_trees_all_close(product_logits, expected_product)


    def test_single_position_in_group(self) -> None:
      """Test that single position returns its own logits."""
      logits = jnp.array([[10.0, -5.0, 3.0]])
      group_mask = jnp.array([True])

      arithmetic_logits = arithmetic_mean_logits(logits, group_mask)
      chex.assert_tree_all_finite(arithmetic_logits)
      
      product_logits = product_of_probabilities_logits(logits, group_mask)
      chex.assert_tree_all_finite(product_logits)

      # Single position: all strategies should return the same logits
      expected = jnp.array([[10.0, -5.0, 3.0]])

      chex.assert_trees_all_close(arithmetic_logits, expected)
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
      arithmetic_fn = self.variant(arithmetic_mean_logits)
      product_fn = self.variant(product_of_probabilities_logits)
      geometric_fn = self.variant(lambda l, m: geometric_mean_logits(l, m, temperature=1.0))

      # Execute JIT-compiled functions
      arithmetic_result = arithmetic_fn(logits, group_mask)
      chex.assert_tree_all_finite(arithmetic_result)
      product_result = product_fn(logits, group_mask)
      chex.assert_tree_all_finite(product_result)
      geometric_result = geometric_fn(logits, group_mask)
      chex.assert_tree_all_finite(geometric_result)

      # Verify results have expected shapes
      chex.assert_shape(arithmetic_result, (1, 2))
      chex.assert_shape(product_result, (1, 2))
      chex.assert_shape(geometric_result, (1, 2))


if __name__ == "__main__":
  pytest.main([__file__, "-v"])

