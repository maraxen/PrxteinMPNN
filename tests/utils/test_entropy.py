"""Tests for entropy utility functions.

Tests cover MLE entropy and Bayesian/Dirichlet posterior entropy estimators.
"""

import chex
import jax.numpy as jnp
from jax.lax import digamma

from aminx.utils.entropy import (
    mle_entropy,
    posterior_entropy_mean,
    posterior_entropy_moments,
    posterior_entropy_squared_mean,
)


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


