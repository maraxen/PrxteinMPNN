"""Test suite for the weights module."""
import jax.numpy as jnp

from prxteinmpnn.psa.weights import linear_weights


def test_linear_weights():
    coordinates = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    r_inner = 1.0
    r_outer = 2.0
    weights = linear_weights(coordinates, r_inner, r_outer)
    expected_weights = jnp.array(
        [
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
    )
    assert jnp.allclose(weights, expected_weights)
