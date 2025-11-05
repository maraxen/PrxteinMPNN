"""Tests for the dense module."""
import chex
import jax
import jax.numpy as jnp
import pytest
from prxteinmpnn.model.dense import dense_layer


@pytest.fixture
def layer_parameters():
    """Create a dummy set of layer parameters for testing."""
    key = jax.random.PRNGKey(0)
    params = {
        "dense_W_in": {
            "w": jax.random.normal(key, (128, 512)),
            "b": jax.random.normal(key, (512,)),
        },
        "dense_W_out": {
            "w": jax.random.normal(key, (512, 128)),
            "b": jax.random.normal(key, (128,)),
        },
    }
    return params


def test_dense_layer(layer_parameters):
    """Test the dense_layer function."""
    node_features = jnp.zeros((10, 128))
    output_features = dense_layer(layer_parameters, node_features)
    chex.assert_shape(output_features, (10, 128))
