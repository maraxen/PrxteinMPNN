"""Unit tests for Average Product Correction (APC) utilities."""

import chex
import jax
import jax.numpy as jnp
import pytest

from prxteinmpnn.utils.apc import (
    apc_corrected_frobenius_norm,
    apc_correction,
    calculate_frobenius_norm_per_pair,
    mean_center,
    symmetrize,
)


# Fixture to provide a sample Jacobian for testing
@pytest.fixture(name="sample_jacobian")
def sample_jacobian_fixture() -> jax.Array:
    """Provides a sample 4D Jacobian array."""
    key = jax.random.PRNGKey(0)
    # Use a small, manageable size for testing
    L = 5
    return jax.random.uniform(key, (L, L, L, L))


def test_mean_center_correctness(sample_jacobian: jax.Array):
    """Test that mean_center correctly centers the Jacobian."""
    centered_jacobian = mean_center(sample_jacobian)

    # The mean of the centered array should be very close to zero
    assert jnp.allclose(jnp.mean(centered_jacobian), 0.0, atol=1e-6)
    chex.assert_shape(centered_jacobian, sample_jacobian.shape)
    chex.assert_type(centered_jacobian, sample_jacobian.dtype)


def test_symmetrize_correctness(sample_jacobian: jax.Array):
    """Test that symmetrize creates a symmetric array."""
    symmetrized_jacobian = symmetrize(sample_jacobian)

    # A symmetric array is equal to its transpose (with specific axes)
    assert jnp.allclose(symmetrized_jacobian, jnp.transpose(symmetrized_jacobian, (2, 3, 0, 1)))
    chex.assert_shape(symmetrized_jacobian, sample_jacobian.shape)
    chex.assert_type(symmetrized_jacobian, sample_jacobian.dtype)


def test_calculate_frobenius_norm_per_pair_correctness():
    """Test the Frobenius norm calculation with a known input."""
    # Create a small, simple symmetric Jacobian
    L = 2
    # The norm of a 2x2 matrix is sqrt(sum of squares)
    known_jacobian = jnp.array(
        [[[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
         [[[5, 6], [7, 8]], [[9, 10], [11, 12]]]]
    )
    # Make it symmetric for the test
    symmetric_jacobian = 0.5 * (known_jacobian + jnp.transpose(known_jacobian, (2, 3, 0, 1)))

    frobenius_matrix = calculate_frobenius_norm_per_pair(symmetric_jacobian)

    # Manually calculate the norm for element (0, 1)
    # The 2x2 matrix at this position is:
    # [[symmetric_jacobian[0, 0, 1, 0], symmetric_jacobian[0, 0, 1, 1]],
    #  [symmetric_jacobian[0, 1, 1, 0], symmetric_jacobian[0, 1, 1, 1]]]
    # This is also sym_jacobian[0,:,1,:]. The norm should be sqrt(5^2+6^2+7^2+8^2)
    expected_norm_01 = jnp.linalg.norm(symmetric_jacobian[0, :, 1, :], ord="fro")
    assert jnp.allclose(frobenius_matrix[0, 1], expected_norm_01)
    chex.assert_shape(frobenius_matrix, (L, L))
    chex.assert_type(frobenius_matrix, jnp.float32)


def test_apc_correction_correctness():
    """Test APC correction with a simple matrix."""
    frobenius_matrix = jnp.array([[1, 2], [3, 4]], dtype=jnp.float32)
    corrected_matrix = apc_correction(frobenius_matrix)

    # Manual calculation:
    # row_means = [1.5, 3.5]
    # col_means = [2.0, 3.0]
    # total_mean = 2.5
    # apc_matrix = [[(1.5*2.0)/2.5, (1.5*3.0)/2.5], [(3.5*2.0)/2.5, (3.5*3.0)/2.5]]
    # apc_matrix = [[1.2, 1.8], [2.8, 4.2]]
    # expected = frobenius_matrix - apc_matrix
    expected_matrix = jnp.array([[-0.2, 0.2], [0.2, -0.2]])

    assert jnp.allclose(corrected_matrix, expected_matrix)
    chex.assert_shape(corrected_matrix, frobenius_matrix.shape)
    chex.assert_type(corrected_matrix, jnp.float32)


def test_apc_corrected_frobenius_norm_pipeline(sample_jacobian: jax.Array):
    """Test the full APC correction pipeline."""
    corrected_matrix = apc_corrected_frobenius_norm(sample_jacobian)

    # The resulting matrix should have the same dimensions as the pairwise matrix
    L = sample_jacobian.shape[0]
    chex.assert_shape(corrected_matrix, (L, L))
    chex.assert_type(corrected_matrix, jnp.float32)

    # The sum of each row and column of the corrected matrix should be close to zero
    assert jnp.allclose(jnp.sum(corrected_matrix, axis=0), 0.0, atol=1e-6)
    assert jnp.allclose(jnp.sum(corrected_matrix, axis=1), 0.0, atol=1e-6)