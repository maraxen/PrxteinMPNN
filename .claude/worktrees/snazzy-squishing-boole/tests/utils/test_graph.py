"""Tests for graph utilities."""

import chex
import jax.numpy as jnp

from prxteinmpnn.utils.graph import compute_neighbor_offsets


def test_compute_neighbor_offsets():
  """Test the computation of offsets between residues.

  Raises:
      AssertionError: If the output does not match the expected value.

  """
  residue_indices = jnp.array([0, 0, 1, 1, 2])  # L=5
  neighbor_indices = jnp.array(
    [[1, 2], [0, 3], [0, 4], [1, 2], [2, 3]],
  )  # (L, K) where K=2

  # Corrected expected output based on the function's logic.
  expected = jnp.array([[0, -1], [0, -1], [1, -1], [1, 0], [1, 1]])

  offsets = compute_neighbor_offsets(residue_indices, neighbor_indices)
  chex.assert_trees_all_equal(offsets, expected)
  chex.assert_shape(offsets, neighbor_indices.shape)
