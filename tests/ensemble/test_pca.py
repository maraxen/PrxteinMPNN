"""Tests for prxteinmpnn.ensemble.pca."""
import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from prxteinmpnn.ensemble.pca import pca_transform


@pytest.fixture
def pca_data():
    """Returns a simple dataset for PCA."""
    key = jax.random.key(0)
    return jax.random.normal(key, (100, 3))


def test_pca_transform_full(pca_data):
    """Test pca_transform with the 'full' solver."""
    n_components = 2
    transformed_data, pca_state = pca_transform(
        data=pca_data, n_components=n_components, solver="full"
    )

    chex.assert_shape(transformed_data, (100, n_components))
    chex.assert_shape(pca_state.components, (n_components, 3))
    chex.assert_shape(pca_state.means, (1, 3))


def test_pca_transform_invalid_solver(pca_data):
    """Test pca_transform with an invalid solver."""
    with pytest.raises(ValueError):
        pca_transform(data=pca_data, n_components=2, solver="invalid")


def test_pca_transform_randomized(pca_data):
    """Test pca_transform with the 'randomized' solver."""
    n_components = 2
    key = jax.random.key(1)
    transformed_data, pca_state = pca_transform(
        data=pca_data, n_components=n_components, solver="randomized", rng=key
    )

    chex.assert_shape(transformed_data, (100, n_components))
    chex.assert_shape(pca_state.components, (n_components, 3))
    chex.assert_shape(pca_state.means, (1, 3))
