"""Tests for GeLU activation function."""

import chex
import jax
import jax.numpy as jnp
from prxteinmpnn.utils.gelu import GeLU


def test_gelu():
    """Test the GeLU activation function against the JAX implementation.

    Raises:
        AssertionError: If the output does not match the JAX implementation.
    """
    x = jnp.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    expected_y = jax.nn.gelu(x, approximate=False)
    y = GeLU(x)
    chex.assert_trees_all_close(y, expected_y)