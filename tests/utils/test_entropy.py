"""Tests for entropy utility functions.

Tests cover von Neumann entropy, MLE entropy, and Bayesian entropy estimators.
"""

import chex
import jax.numpy as jnp
from jax.lax import digamma

from prxteinmpnn.utils.entropy import (
    mle_entropy,
    posterior_entropy_mean,
    posterior_entropy_moments,
    posterior_entropy_squared_mean,
    posterior_mean_std,
    von_neumman,
)


class TestVonNeumannEntropy(chex.TestCase):
    """Test von Neumann entropy calculation."""

    @chex.variants(with_jit=True, without_jit=True)
    def test_von_neumann_identity_matrix(self):
        """Test von Neumann entropy of identity matrix."""
        n = 4
        rho = jnp.eye(n)
        expected = 0.0
        von_neumman_fn = self.variant(von_neumman)
        result = von_neumman_fn(rho)
        chex.assert_trees_all_close(result, expected)
        chex.assert_tree_all_finite(result)

    @chex.variants(with_jit=True, without_jit=True)
    def test_von_neumann_uniform_density(self):
        """Test von Neumann entropy for uniform density matrix."""
        n = 3
        rho = jnp.ones((n, n)) / n
        expected = 0.0
        von_neumman_fn = self.variant(von_neumman)
        result = von_neumman_fn(rho)
        chex.assert_trees_all_close(result, expected, atol=1e-6)
        chex.assert_tree_all_finite(result)

    @chex.variants(with_jit=True, without_jit=True)
    def test_von_neumann_rank_one_matrix(self):
        """Test von Neumann entropy for rank-1 matrix (should be zero)."""
        v = jnp.array([1.0, 0.0, 0.0])
        rho = jnp.outer(v, v)
        von_neumman_fn = self.variant(von_neumman)
        result = von_neumman_fn(rho)
        chex.assert_trees_all_close(result, 0.0, atol=1e-10)
        chex.assert_tree_all_finite(result)

    @chex.variants(with_jit=True, without_jit=True)
    def test_von_neumann_negative_eigenvalues(self):
        """Test that negative eigenvalues are handled correctly (relu applied)."""
        rho = jnp.array([[-1.0, 0.0], [0.0, 1.0]])
        expected = 0.0
        von_neumman_fn = self.variant(von_neumman)
        result = von_neumman_fn(rho)
        chex.assert_trees_all_close(result, expected)
        chex.assert_tree_all_finite(result)

    @chex.variants(with_jit=True, without_jit=True)
    def test_von_neumann_equal_eigenvalues(self):
        """Test von Neumann entropy with known equal eigenvalues."""
        rho = jnp.array([[0.5, 0.0], [0.0, 0.5]])
        expected = 2 * (-0.5 * jnp.log(0.5))
        von_neumman_fn = self.variant(von_neumman)
        result = von_neumman_fn(rho)
        chex.assert_trees_all_close(result, expected, atol=1e-6)
        chex.assert_tree_all_finite(result)

    @chex.variants(with_jit=True, without_jit=True)
    def test_von_neumann_single_nonzero_eigenvalue(self):
        """Test von Neumann entropy with a single non-zero eigenvalue."""
        rho = jnp.array([[0.8, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        expected = -0.8 * jnp.log(0.8)
        von_neumman_fn = self.variant(von_neumman)
        result = von_neumman_fn(rho)
        chex.assert_trees_all_close(result, expected, atol=1e-6)
        chex.assert_tree_all_finite(result)

    @chex.variants(with_jit=True, without_jit=True)
    def test_von_neumann_proper_density_matrix(self):
        """Test von Neumann entropy with a proper normalized density matrix."""
        rho = jnp.array([[0.7, 0.1], [0.1, 0.3]])
        von_neumman_fn = self.variant(von_neumman)
        result = von_neumman_fn(rho)
        eigenvals, _ = jnp.linalg.eigh(rho)
        expected_manual = jnp.sum(
            jnp.where(eigenvals > 0, -eigenvals * jnp.log(eigenvals), 0),
        )
        chex.assert_trees_all_close(result, expected_manual, atol=1e-6)
        chex.assert_tree_all_finite(result)


class TestMLEEntropy(chex.TestCase):
    """Test maximum likelihood entropy estimation."""

    @chex.variants(with_jit=True, without_jit=True)
    def test_mle_entropy_uniform_distribution(self):
        """Test MLE entropy for uniform distribution."""
        states = jnp.array([10, 10, 10, 10])
        expected = jnp.log(4.0)
        mle_entropy_fn = self.variant(mle_entropy)
        result = mle_entropy_fn(states)
        chex.assert_trees_all_close(result, expected)
        chex.assert_tree_all_finite(result)

    @chex.variants(with_jit=True, without_jit=True)
    def test_mle_entropy_single_state(self):
        """Test MLE entropy when all mass is on one state (should be zero)."""
        states = jnp.array([100, 0, 0, 0])
        mle_entropy_fn = self.variant(mle_entropy)
        result = mle_entropy_fn(states)
        chex.assert_trees_all_close(result, 0.0)
        chex.assert_tree_all_finite(result)

    @chex.variants(with_jit=True, without_jit=True)
    def test_mle_entropy_binary_distribution(self):
        """Test MLE entropy for binary distribution."""
        states = jnp.array([50, 50])
        expected = jnp.log(2.0)
        mle_entropy_fn = self.variant(mle_entropy)
        result = mle_entropy_fn(states)
        chex.assert_trees_all_close(result, expected)
        chex.assert_tree_all_finite(result)

    @chex.variants(with_jit=True, without_jit=True)
    def test_mle_entropy_flattening(self):
        """Test that MLE entropy correctly flattens multidimensional input."""
        states = jnp.array([[5, 5], [10, 10]])
        states_flat = jnp.array([5, 5, 10, 10])
        mle_entropy_fn = self.variant(mle_entropy)
        result_2d = mle_entropy_fn(states)
        result_1d = mle_entropy_fn(states_flat)
        chex.assert_trees_all_close(result_2d, result_1d)
        chex.assert_tree_all_finite(result_2d)


class TestPosteriorEntropyMean(chex.TestCase):
    """Test posterior entropy mean calculation."""

    @chex.variants(with_jit=True, without_jit=True)
    def test_posterior_entropy_mean_symmetric_alpha(self):
        """Test posterior entropy mean with symmetric Dirichlet prior."""
        alpha = jnp.array([1.0, 1.0, 1.0])
        expected = digamma(4.0) - digamma(2.0)
        posterior_entropy_mean_fn = self.variant(posterior_entropy_mean)
        result = posterior_entropy_mean_fn(alpha)
        chex.assert_trees_all_close(result, expected, atol=1e-6)
        chex.assert_tree_all_finite(result)

    @chex.variants(with_jit=True, without_jit=True)
    def test_posterior_entropy_mean_single_dimension(self):
        """Test posterior entropy mean with single dimension (should be zero)."""
        alpha = jnp.array([5.0])
        posterior_entropy_mean_fn = self.variant(posterior_entropy_mean)
        result = posterior_entropy_mean_fn(alpha)
        chex.assert_trees_all_close(result, 0.0)
        chex.assert_tree_all_finite(result)

    @chex.variants(with_jit=True, without_jit=True)
    def test_posterior_entropy_mean_concentrated_alpha(self):
        """Test posterior entropy mean with concentrated distribution."""
        alpha = jnp.array([100.0, 1.0, 1.0])
        posterior_entropy_mean_fn = self.variant(posterior_entropy_mean)
        result = posterior_entropy_mean_fn(alpha)
        assert result < 0.15
        chex.assert_tree_all_finite(result)


class TestPosteriorEntropySquaredMean(chex.TestCase):
    """Test posterior entropy squared mean calculation."""

    @chex.variants(with_jit=True, without_jit=True)
    def test_posterior_entropy_squared_mean_positive(self):
        """Test that posterior entropy squared mean is always non-negative."""
        alpha = jnp.array([2.0, 3.0, 1.0])
        posterior_entropy_squared_mean_fn = self.variant(
            posterior_entropy_squared_mean,
        )
        result = posterior_entropy_squared_mean_fn(alpha)
        assert result >= 0.0
        chex.assert_tree_all_finite(result)

    @chex.variants(with_jit=True, without_jit=True)
    def test_posterior_entropy_squared_mean_single_dimension(self):
        """Test posterior entropy squared mean with single dimension."""
        alpha = jnp.array([5.0])
        posterior_entropy_squared_mean_fn = self.variant(
            posterior_entropy_squared_mean,
        )
        result = posterior_entropy_squared_mean_fn(alpha)
        chex.assert_trees_all_close(result, 0.0)
        chex.assert_tree_all_finite(result)


class TestPosteriorEntropyMoments(chex.TestCase):
    """Test posterior entropy moments calculation."""

    @chex.variants(with_jit=True, without_jit=True)
    def test_posterior_entropy_moments_shape(self):
        """Test that posterior entropy moments returns array of shape (2,)."""
        alpha = jnp.array([1.0, 2.0, 3.0])
        posterior_entropy_moments_fn = self.variant(posterior_entropy_moments)
        result = posterior_entropy_moments_fn(alpha)
        chex.assert_shape(result, (2,))
        chex.assert_tree_all_finite(result)

    @chex.variants(with_jit=True, without_jit=True)
    def test_posterior_entropy_moments_consistency(self):
        """Test consistency between moments and individual function calls."""
        alpha = jnp.array([2.0, 3.0, 1.5])
        posterior_entropy_moments_fn = self.variant(posterior_entropy_moments)
        moments = posterior_entropy_moments_fn(alpha)
        mean = posterior_entropy_mean(alpha)
        squared_mean = posterior_entropy_squared_mean(alpha)
        chex.assert_trees_all_close(moments[0], mean)
        chex.assert_trees_all_close(moments[1], squared_mean)
        chex.assert_tree_all_finite(moments)

    @chex.variants(with_jit=True, without_jit=True)
    def test_posterior_entropy_moments_flattening(self):
        """Test that moments function correctly flattens input."""
        alpha_2d = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        alpha_1d = jnp.array([1.0, 2.0, 3.0, 4.0])
        posterior_entropy_moments_fn = self.variant(posterior_entropy_moments)
        result_2d = posterior_entropy_moments_fn(alpha_2d)
        result_1d = posterior_entropy_moments_fn(alpha_1d)
        chex.assert_trees_all_close(result_2d, result_1d)
        chex.assert_tree_all_finite(result_2d)


class TestPosteriorMeanStd(chex.TestCase):
    """Test posterior mean and standard deviation calculation."""

    @chex.variants(with_jit=True, without_jit=True)
    def test_posterior_mean_std_shape(self):
        """Test that posterior mean std returns array of shape (2,)."""
        alpha = jnp.array([1.0, 2.0, 3.0])
        posterior_mean_std_fn = self.variant(posterior_mean_std)
        result = posterior_mean_std_fn(alpha)
        chex.assert_shape(result, (2,))
        chex.assert_tree_all_finite(result)

    @chex.variants(with_jit=True, without_jit=True)
    def test_posterior_mean_std_positive_std(self):
        """Test that standard deviation is always non-negative."""
        alpha = jnp.array([2.0, 3.0, 1.5])
        posterior_mean_std_fn = self.variant(posterior_mean_std)
        mean, std = posterior_mean_std_fn(alpha)
        assert std >= 0.0
        chex.assert_tree_all_finite(mean)
        chex.assert_tree_all_finite(std)

    @chex.variants(with_jit=True, without_jit=True)
    def test_posterior_mean_std_consistency_with_moments(self):
        """Test consistency with moments calculation."""
        alpha = jnp.array([3.0, 2.0, 1.0])
        posterior_mean_std_fn = self.variant(posterior_mean_std)
        mean, std = posterior_mean_std_fn(alpha)
        moments = posterior_entropy_moments(alpha)
        expected_mean = moments[0]
        expected_std = jnp.sqrt(moments[1] - moments[0] ** 2)
        chex.assert_trees_all_close(mean, expected_mean)
        chex.assert_trees_all_close(std, expected_std)
        chex.assert_tree_all_finite(mean)
        chex.assert_tree_all_finite(std)

    @chex.variants(with_jit=True, without_jit=True)
    def test_posterior_mean_std_zero_variance_case(self):
        """Test behavior when variance is effectively zero."""
        alpha = jnp.array([10.0])
        posterior_mean_std_fn = self.variant(posterior_mean_std)
        mean, std = posterior_mean_std_fn(alpha)
        chex.assert_trees_all_close(std, 0.0, atol=1e-10)
        chex.assert_tree_all_finite(mean)
        chex.assert_tree_all_finite(std)
