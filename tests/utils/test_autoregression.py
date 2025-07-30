"""Tests for autoregression utilities."""

import chex
import jax.numpy as jnp
import pytest
from prxteinmpnn.utils.autoregression import generate_ar_mask


@pytest.mark.parametrize(
    "decoding_order, expected_mask",
    [
        (
            jnp.array([0, 1, 2]),
            jnp.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]]),
        ),
        (
            jnp.array([2, 0, 1]),
            jnp.array([[1, 1, 1], [0, 1, 0], [0, 1, 1]]),
        ),
        (
            jnp.array([1, 2, 0]),
            jnp.array([[1, 0, 1], [1, 1, 1], [0, 0, 1]]),
        ),
    ],
)
def test_generate_ar_mask(decoding_order, expected_mask):
    """Test the generation of the autoregressive mask.

    Args:
        decoding_order: The order in which atoms are decoded.
        expected_mask: The expected autoregressive mask.

    Raises:
        AssertionError: If the output does not match the expected value.
    """
    mask = generate_ar_mask(decoding_order)
    chex.assert_trees_all_equal(mask, expected_mask)
    chex.assert_shape(mask, (len(decoding_order), len(decoding_order)))
    chex.assert_type(mask, int)