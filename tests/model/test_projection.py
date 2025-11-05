"""Tests for the projection module."""
import chex
import jax
import jax.numpy as jnp
import pytest
from prxteinmpnn.model.projection import final_projection


@pytest.fixture
def model_parameters():
    """Create a dummy set of model parameters for testing."""
    key = jax.random.PRNGKey(0)
    params = {
        "protein_mpnn/~/W_out": {
            "w": jax.random.normal(key, (128, 21)),
            "b": jax.random.normal(key, (21,)),
        }
    }
    return params


def test_final_projection(model_parameters):
    """Test the final_projection function."""
    node_features = jnp.zeros((10, 128))
    logits = final_projection(model_parameters, node_features)
    chex.assert_shape(logits, (10, 21))
