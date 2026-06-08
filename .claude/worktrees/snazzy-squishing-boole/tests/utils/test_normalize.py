"""Unit tests for the normalization module."""

import chex
import jax.numpy as jnp

from prxteinmpnn.utils import normalize


def test_normalize():
    """Test the normalize function for correctness."""
    x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    scale = jnp.array([1.5, 1.5, 1.5])
    offset = jnp.array([0.5, 0.5, 0.5])

    normalized_x = normalize.normalize(x, scale, offset)

    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.var(x, axis=-1, keepdims=True)
    expected_normalized_x = (x - mean) / jnp.sqrt(
        variance + normalize.STANDARD_EPSILON,
    )
    expected_output = expected_normalized_x * scale + offset

    chex.assert_trees_all_close(normalized_x, expected_output)


def test_layer_normalization():
    """Test the layer_normalization function for correctness."""
    x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    layer_parameters = {
        "scale": jnp.array([1.5, 1.5, 1.5]),
        "offset": jnp.array([0.5, 0.5, 0.5]),
    }

    normalized_x = normalize.layer_normalization(x, layer_parameters)

    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.var(x, axis=-1, keepdims=True)
    expected_normalized_x = (x - mean) / jnp.sqrt(
        variance + normalize.STANDARD_EPSILON,
    )
    expected_output = (
        expected_normalized_x * layer_parameters["scale"] + layer_parameters["offset"]
    )

    chex.assert_trees_all_close(normalized_x, expected_output)


def test_normalize_with_different_axis():
    """Test the normalize function with a different normalization axis."""
    x = jnp.array(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
    )  # Shape (2, 2, 2)
    scale = jnp.array([1.5, 1.5])
    offset = jnp.array([0.5, 0.5])

    # Normalize over axis 1
    normalized_x = normalize.normalize(x, scale, offset, axis=1)

    mean = jnp.mean(x, axis=1, keepdims=True)
    variance = jnp.var(x, axis=1, keepdims=True)
    expected_normalized_x = (x - mean) / jnp.sqrt(
        variance + normalize.STANDARD_EPSILON,
    )
    expected_output = expected_normalized_x * scale + offset

    chex.assert_trees_all_close(normalized_x, expected_output)
