"""Tests for the catjac utility functions."""

import jax
import jax.numpy as jnp
import pytest
from chex import assert_trees_all_close

from prxteinmpnn.utils.catjac import (
    add_jacobians,
    make_combine_jac,
    subtract_jacobians,
)


@pytest.fixture
def sample_jacobians():
    """Fixture for a sample CategoricalJacobian tensor."""
    key = jax.random.PRNGKey(0)
    return jax.random.normal(key, shape=(2, 3, 5, 21, 5, 21))


@pytest.fixture
def sample_mapping():
    """Fixture for a sample InterproteinMapping."""
    mapping = jnp.eye(5, dtype=jnp.int32)
    return jnp.stack([mapping, jnp.roll(mapping, 1, axis=1)])


@pytest.fixture
def sample_sequences():
    """Fixture for sample ProteinSequences."""
    return jnp.array([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]])


def test_add_jacobians(sample_jacobians, sample_mapping):
    """Test the add_jacobians function."""
    combined = add_jacobians(sample_jacobians, sample_mapping)
    assert combined.shape == sample_jacobians.shape
    # Basic check: addition should result in a different tensor
    assert not jnp.allclose(combined, sample_jacobians)


def test_subtract_jacobians(sample_jacobians, sample_mapping):
    """Test the subtract_jacobians function."""
    combined = subtract_jacobians(sample_jacobians, sample_mapping)
    assert combined.shape == sample_jacobians.shape
    # Basic check: subtraction should result in a different tensor
    assert not jnp.allclose(combined, sample_jacobians)


def test_make_combine_jac_add(sample_jacobians, sample_sequences):
    """Test the make_combine_jac factory with 'add' operation."""
    combine_fn = make_combine_jac("add")
    combined = combine_fn(sample_jacobians, sample_sequences, None)
    assert combined.shape == (1, 3, 5, 21, 5, 21)


def test_make_combine_jac_subtract(sample_jacobians, sample_sequences):
    """Test the make_combine_jac factory with 'subtract' operation."""
    combine_fn = make_combine_jac("subtract")
    combined = combine_fn(sample_jacobians, sample_sequences, None)
    assert combined.shape == (1, 3, 5, 21, 5, 21)


def test_make_combine_jac_custom(sample_jacobians, sample_sequences):
    """Test the make_combine_jac factory with a custom function."""

    def custom_combine(jacobians, mapping, weights):
        return jacobians * 2

    combine_fn = make_combine_jac(custom_combine)
    combined = combine_fn(sample_jacobians, sample_sequences, None)
    assert_trees_all_close(combined, sample_jacobians[0] * 2)

