"""Tests for the catjac utility functions."""

import jax
import jax.numpy as jnp
import pytest
from chex import assert_trees_all_close

# The new function signatures require jac_A and jac_B to be passed
from prxteinmpnn.utils.catjac import (
    _add_jacobians_mapped,
    _subtract_jacobians_mapped,
    _gather_mapped_jacobian,
    make_combine_jac,
)
# We assume a mock or real align_sequences is available for the factory tests
from prxteinmpnn.utils.align import align_sequences


@pytest.fixture
def sample_jacobians():
    """Fixture for a sample CategoricalJacobian tensor."""
    key = jax.random.PRNGKey(0)
    # Shape: (N, noise_levels, L, 21, L, 21) -> (2, 3, 5, 21, 5, 21)
    return jax.random.normal(key, shape=(2, 3, 5, 21, 5, 21))


@pytest.fixture
def sample_mapping():
    """Fixture for a single inter-protein index mapping."""
    # This now represents a single (L, 2) index map, like align_sequences produces.
    # It maps all 5 positions identically.
    i_indices = jnp.arange(5)
    k_indices = jnp.arange(5)
    return jnp.stack([i_indices, k_indices], axis=-1)


@pytest.fixture
def sample_sequences():
    """Fixture for sample ProteinSequences."""
    # Shape: (N, L) -> (2, 5)
    return jnp.array([[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]])


@pytest.fixture
def sample_weights():
    """Fixture for sample weights."""
    # Shape: (N,) -> (2,)
    return jnp.array([1.0, 0.5])


@pytest.fixture
def sample_pair_tuple(sample_jacobians, sample_mapping, sample_weights):
    """Fixture for a tuple representing a single pair to be combined."""
    # New signature: (jac_A, jac_B, mapping, weight)
    return (
        sample_jacobians[0],  # jac_A
        sample_jacobians[1],  # jac_B
        sample_mapping,       # The index map
        sample_weights[0],    # A scalar weight for the pair
    )


# --- Tests for internal "mapped" functions ---

def test_add_jacobians(sample_pair_tuple):
    """Test the _add_jacobians_mapped function with the new signature."""
    jac_A, _, _, _ = sample_pair_tuple
    combined = _add_jacobians_mapped(*sample_pair_tuple)
    assert combined.shape == jac_A.shape
    assert not jnp.allclose(combined, jac_A)


def test_subtract_jacobians(sample_pair_tuple):
    """Test the _subtract_jacobians_mapped function with the new signature."""
    jac_A, _, _, _ = sample_pair_tuple
    combined = _subtract_jacobians_mapped(*sample_pair_tuple)
    assert combined.shape == jac_A.shape
    assert not jnp.allclose(combined, jac_A)


# --- Tests for the public-facing factory ---

def test_make_combine_jac_add(sample_jacobians, sample_sequences):
    """Test the make_combine_jac factory with 'add' operation."""
    combine_fn = make_combine_jac("add")
    combined = combine_fn(sample_jacobians, sample_sequences, None)
    # For N=2, triu_indices gives 1 pair, so the batch size is 1.
    assert combined.shape == (1, 3, 5, 21, 5, 21)


def test_make_combine_jac_subtract(sample_jacobians, sample_sequences):
    """Test the make_combine_jac factory with 'subtract' operation."""
    combine_fn = make_combine_jac("subtract")
    combined = combine_fn(sample_jacobians, sample_sequences, None)
    # For N=2, triu_indices gives 1 pair, so the batch size is 1.
    assert combined.shape == (1, 3, 5, 21, 5, 21)


def test_make_combine_jac_subtract_identity_is_zero(
    sample_jacobians, sample_sequences
):
    """Test that subtracting an identical Jacobian from itself results in zero."""
    # Create a batch where both proteins are identical
    identical_jacobians = jnp.stack([sample_jacobians[0], sample_jacobians[0]])
    identical_sequences = jnp.stack([sample_sequences[0], sample_sequences[0]])

    combine_fn = make_combine_jac("subtract")

    # The result of J_0 - map(J_0) for the single pair (0, 1) should be zero.
    result = combine_fn(identical_jacobians, identical_sequences, None)

    # Check that the result is close to zero within a small tolerance
    assert_trees_all_close(result, jnp.zeros_like(result), atol=1e-6)
    
def test_make_combine_jac_custom_function(
    sample_jacobians, sample_sequences, sample_pair_tuple
):
    """Test the make_combine_jac factory with a custom combine function."""
    def custom_combine(jac_A, jac_B, mapping, weights):
        # Simple custom function that adds twice the mapped jac_B
        mapped_jac_B = _gather_mapped_jacobian(jac_B, mapping)
        return jac_A + 2.0 * mapped_jac_B * (weights if weights is not None else 1.0)

    combine_fn = make_combine_jac(custom_combine)

    combined = combine_fn(sample_jacobians, sample_sequences, None)
    assert combined.shape == (1, 3, 5, 21, 5, 21)
    # Ensure the result is different from simple addition
    simple_add_fn = make_combine_jac("add")
    simple_added = simple_add_fn(sample_jacobians, sample_sequences, None)
    assert not jnp.allclose(combined, simple_added)