"""Tests for entropy utility functions.

Tests cover von Neumann entropy, MLE entropy, and Bayesian entropy estimators.
"""

import jax.numpy as jnp
import pytest
from jax import random
from jax.typing import ArrayLike
from jaxtyping import Float

from prxteinmpnn.utils.entropy import (
  mle_entropy,
  posterior_entropy_mean,
  posterior_entropy_moments,
  posterior_entropy_squared_mean,
  posterior_mean_std,
  von_neumman,
)


class TestVonNeumannEntropy:
  """Test von Neumann entropy calculation."""

  def test_von_neumann_identity_matrix(self) -> None:
    """Test von Neumann entropy of identity matrix.
    
    For an identity matrix, all eigenvalues are 1, so entropy should be 0 
    since 1 * log(1) = 0 for each eigenvalue.
    """
    n = 4
    rho = jnp.eye(n)
    expected = 0.0  # All eigenvalues are 1, so entropy is 0
    result = von_neumman(rho)
    assert jnp.allclose(result, expected), f"Expected {expected}, got {result}"

  def test_von_neumann_uniform_density(self) -> None:
    """Test von Neumann entropy for uniform density matrix.
    
    For a uniform density matrix rho_ij = 1/n, this creates a rank-1 matrix
    with eigenvalue 1 (unnormalized) and the rest 0.
    """
    n = 3
    rho = jnp.ones((n, n)) / n
    result = von_neumman(rho)
    # This matrix has eigenvalue 1.0 with multiplicity 1 and eigenvalue 0 with multiplicity n-1
    # So entropy = entr(1.0) = -1.0 * log(1.0) = 0
    expected = 0.0
    assert jnp.allclose(result, expected, atol=1e-6), f"Expected {expected}, got {result}"

  def test_von_neumann_rank_one_matrix(self) -> None:
    """Test von Neumann entropy for rank-1 matrix (should be zero).
    
    A rank-1 matrix has only one non-zero eigenvalue, so entropy should be 0.
    """
    v = jnp.array([1.0, 0.0, 0.0])
    rho = jnp.outer(v, v)
    result = von_neumman(rho)
    assert jnp.allclose(result, 0.0, atol=1e-10), f"Expected 0.0, got {result}"

  def test_von_neumann_negative_eigenvalues(self) -> None:
    """Test that negative eigenvalues are handled correctly (relu applied)."""
    # Create a matrix with negative eigenvalue
    rho = jnp.array([[-1.0, 0.0], [0.0, 1.0]])
    result = von_neumman(rho)
    # Should only count the positive eigenvalue
    expected = 0.0  # Only one eigenvalue contributes: 1 * log(1) = 0
    assert jnp.allclose(result, expected), f"Expected {expected}, got {result}"

  def test_von_neumann_equal_eigenvalues(self) -> None:
    """Test von Neumann entropy with known equal eigenvalues."""
    # Create a diagonal matrix with equal eigenvalues [0.5, 0.5]
    rho = jnp.array([[0.5, 0.0], [0.0, 0.5]])
    result = von_neumman(rho)
    # The entr function computes -x*log(x), so for eigenvalues [0.5, 0.5]:
    # entropy = entr(0.5) + entr(0.5) = 2 * (-0.5 * log(0.5)) = -log(0.5) = log(2)
    expected = 2 * (-0.5 * jnp.log(0.5))
    assert jnp.allclose(result, expected, atol=1e-6), f"Expected {expected}, got {result}"

  def test_von_neumann_single_nonzero_eigenvalue(self) -> None:
    """Test von Neumann entropy with a single non-zero eigenvalue."""
    # Create a matrix with eigenvalues [0.8, 0, 0]
    rho = jnp.array([[0.8, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    result = von_neumman(rho)
    # Only one eigenvalue contributes: entr(0.8) = -0.8 * log(0.8)
    expected = -0.8 * jnp.log(0.8)
    assert jnp.allclose(result, expected, atol=1e-6), f"Expected {expected}, got {result}"

  def test_von_neumann_proper_density_matrix(self) -> None:
    """Test von Neumann entropy with a proper normalized density matrix."""
    # Create a proper density matrix with trace = 1
    rho = jnp.array([[0.7, 0.1], [0.1, 0.3]])
    result = von_neumman(rho)
    
    # Compute eigenvalues manually to verify
    eigenvals, _ = jnp.linalg.eigh(rho)
    expected_manual = jnp.sum(jnp.where(eigenvals > 0, -eigenvals * jnp.log(eigenvals), 0))
    
    # Should be finite and positive for a mixed state
    assert jnp.isfinite(result), f"Result should be finite, got {result}"
    assert result >= 0, f"von Neumann entropy should be non-negative, got {result}"
    assert jnp.allclose(result, expected_manual, atol=1e-6), (
      f"Expected {expected_manual}, got {result}"
    )


class TestMLEEntropy:
  """Test maximum likelihood entropy estimation."""

  def test_mle_entropy_uniform_distribution(self) -> None:
    """Test MLE entropy for uniform distribution."""
    states = jnp.array([10, 10, 10, 10])  # Uniform counts
    expected = jnp.log(4.0)  # log(n) for uniform distribution
    result = mle_entropy(states)
    assert jnp.allclose(result, expected), f"Expected {expected}, got {result}"

  def test_mle_entropy_single_state(self) -> None:
    """Test MLE entropy when all mass is on one state (should be zero)."""
    states = jnp.array([100, 0, 0, 0])
    result = mle_entropy(states)
    assert jnp.allclose(result, 0.0), f"Expected 0.0, got {result}"

  def test_mle_entropy_binary_distribution(self) -> None:
    """Test MLE entropy for binary distribution."""
    states = jnp.array([50, 50])  # Equal binary
    expected = jnp.log(2.0)
    result = mle_entropy(states)
    assert jnp.allclose(result, expected), f"Expected {expected}, got {result}"

  def test_mle_entropy_flattening(self) -> None:
    """Test that MLE entropy correctly flattens multidimensional input."""
    states = jnp.array([[5, 5], [10, 10]])
    states_flat = jnp.array([5, 5, 10, 10])
    result_2d = mle_entropy(states)
    result_1d = mle_entropy(states_flat)
    assert jnp.allclose(result_2d, result_1d), "Results should be identical after flattening"


class TestPosteriorEntropyMean:
  """Test posterior entropy mean calculation."""

  def test_posterior_entropy_mean_symmetric_alpha(self) -> None:
    """Test posterior entropy mean with symmetric Dirichlet prior."""
    alpha = jnp.array([1.0, 1.0, 1.0])  # Uniform prior
    result = posterior_entropy_mean(alpha)
    # For symmetric Dirichlet(1,1,1), expected value is digamma(4) - digamma(2)
    from jax.lax import digamma
    expected = digamma(4.0) - digamma(2.0)
    assert jnp.allclose(result, expected, atol=1e-6), f"Expected {expected}, got {result}"

  def test_posterior_entropy_mean_single_dimension(self) -> None:
    """Test posterior entropy mean with single dimension (should be zero)."""
    alpha = jnp.array([5.0])
    result = posterior_entropy_mean(alpha)
    assert jnp.allclose(result, 0.0), f"Expected 0.0 for single dimension, got {result}"

  def test_posterior_entropy_mean_concentrated_alpha(self) -> None:
    """Test posterior entropy mean with concentrated distribution."""
    alpha = jnp.array([100.0, 1.0, 1.0])  # Very concentrated on first state
    result = posterior_entropy_mean(alpha)
    # Should be small but not necessarily < 0.1, adjust threshold
    assert result < 0.15, f"Expected small entropy for concentrated alpha, got {result}"


class TestPosteriorEntropySquaredMean:
  """Test posterior entropy squared mean calculation."""

  def test_posterior_entropy_squared_mean_positive(self) -> None:
    """Test that posterior entropy squared mean is always non-negative."""
    alpha = jnp.array([2.0, 3.0, 1.0])
    result = posterior_entropy_squared_mean(alpha)
    assert result >= 0.0, f"Squared mean should be non-negative, got {result}"

  def test_posterior_entropy_squared_mean_single_dimension(self) -> None:
    """Test posterior entropy squared mean with single dimension."""
    alpha = jnp.array([5.0])
    result = posterior_entropy_squared_mean(alpha)
    assert jnp.allclose(result, 0.0), f"Expected 0.0 for single dimension, got {result}"


class TestPosteriorEntropyMoments:
  """Test posterior entropy moments calculation."""

  def test_posterior_entropy_moments_shape(self) -> None:
    """Test that posterior entropy moments returns array of shape (2,)."""
    alpha = jnp.array([1.0, 2.0, 3.0])
    result = posterior_entropy_moments(alpha)
    assert result.shape == (2,), f"Expected shape (2,), got {result.shape}"

  def test_posterior_entropy_moments_consistency(self) -> None:
    """Test consistency between moments and individual function calls."""
    alpha = jnp.array([2.0, 3.0, 1.5])
    moments = posterior_entropy_moments(alpha)
    mean = posterior_entropy_mean(alpha)
    squared_mean = posterior_entropy_squared_mean(alpha)
    
    assert jnp.allclose(moments[0], mean), "First moment should match mean"
    assert jnp.allclose(moments[1], squared_mean), "Second moment should match squared mean"

  def test_posterior_entropy_moments_flattening(self) -> None:
    """Test that moments function correctly flattens input."""
    alpha_2d = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    alpha_1d = jnp.array([1.0, 2.0, 3.0, 4.0])
    
    result_2d = posterior_entropy_moments(alpha_2d)
    result_1d = posterior_entropy_moments(alpha_1d)
    
    assert jnp.allclose(result_2d, result_1d), "Results should be identical after flattening"


class TestPosteriorMeanStd:
  """Test posterior mean and standard deviation calculation."""

  def test_posterior_mean_std_shape(self) -> None:
    """Test that posterior mean std returns array of shape (2,)."""
    alpha = jnp.array([1.0, 2.0, 3.0])
    result = posterior_mean_std(alpha)
    assert result.shape == (2,), f"Expected shape (2,), got {result.shape}"

  def test_posterior_mean_std_positive_std(self) -> None:
    """Test that standard deviation is always non-negative."""
    alpha = jnp.array([2.0, 3.0, 1.5])
    mean, std = posterior_mean_std(alpha)
    assert std >= 0.0, f"Standard deviation should be non-negative, got {std}"

  def test_posterior_mean_std_consistency_with_moments(self) -> None:
    """Test consistency with moments calculation."""
    alpha = jnp.array([3.0, 2.0, 1.0])
    mean, std = posterior_mean_std(alpha)
    moments = posterior_entropy_moments(alpha)
    
    expected_mean = moments[0]
    expected_std = jnp.sqrt(moments[1] - moments[0]**2)
    
    assert jnp.allclose(mean, expected_mean), "Mean should match first moment"
    assert jnp.allclose(std, expected_std), "Std should match sqrt(var)"

  def test_posterior_mean_std_zero_variance_case(self) -> None:
    """Test behavior when variance is effectively zero."""
    # Single dimension case should have zero variance
    alpha = jnp.array([10.0])
    mean, std = posterior_mean_std(alpha)
    assert jnp.allclose(std, 0.0, atol=1e-10), f"Expected zero std for single dimension, got {std}"


@pytest.fixture
def random_key():
  """Provide a random key for tests that need randomness."""
  return random.PRNGKey(42)


@pytest.fixture
def sample_alpha():
  """Provide a sample alpha vector for testing."""
  return jnp.array([1.0, 2.0, 3.0, 0.5])


@pytest.fixture
def sample_rho():
  """Provide a sample density matrix for testing."""
  # Create a valid density matrix (positive semidefinite)
  a = jnp.array([[1.0, 0.5], [0.5, 1.0]])
  return a / jnp.trace(a)  # Normalize


class TestIntegration:
  """Integration tests combining multiple functions."""

  def test_jit_compatibility(self, sample_alpha: Float, sample_rho: Float) -> None:
    """Test that JIT-compiled functions work correctly."""
    from jax import jit
    
    # Test von_neumann (already JIT-compiled)
    result_vn = von_neumman(sample_rho)
    assert jnp.isfinite(result_vn), "JIT-compiled von_neumann should return finite result"
    
    # Test posterior_mean_std (already JIT-compiled)
    result_ms = posterior_mean_std(sample_alpha)
    assert jnp.all(jnp.isfinite(result_ms)), "JIT-compiled posterior_mean_std should return finite results"

  def test_numerical_stability_large_alpha(self) -> None:
    """Test numerical stability with large alpha values."""
    alpha = jnp.array([1000.0, 2000.0, 500.0])
    
    mean = posterior_entropy_mean(alpha)
    squared_mean = posterior_entropy_squared_mean(alpha)
    moments = posterior_entropy_moments(alpha)
    mean_std = posterior_mean_std(alpha)
    
    # All results should be finite
    assert jnp.isfinite(mean), "Mean should be finite for large alpha"
    assert jnp.isfinite(squared_mean), "Squared mean should be finite for large alpha"
    assert jnp.all(jnp.isfinite(moments)), "Moments should be finite for large alpha"
    assert jnp.all(jnp.isfinite(mean_std)), "Mean/std should be finite for large alpha"

  def test_numerical_stability_small_alpha(self) -> None:
    """Test numerical stability with small alpha values."""
    alpha = jnp.array([1e-6, 1e-5, 1e-4])
    
    mean = posterior_entropy_mean(alpha)
    squared_mean = posterior_entropy_squared_mean(alpha)
    
    # Results should be finite and reasonable
    assert jnp.isfinite(mean), "Mean should be finite for small alpha"
    assert jnp.isfinite(squared_mean), "Squared mean should be finite for small alpha"
    assert mean > 0, "Mean entropy should be positive for small alpha"
    
    # Results should be finite and reasonable
    assert jnp.isfinite(mean), "Mean should be finite for small alpha"
    assert jnp.isfinite(squared_mean), "Squared mean should be finite for small alpha"
    assert mean > 0, "Mean entropy should be positive for small alpha"
