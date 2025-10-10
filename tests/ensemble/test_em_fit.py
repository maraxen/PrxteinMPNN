"""Tests for Expectation-Maximization algorithm for Gaussian Mixture Models."""

import chex
import jax
import jax.numpy as jnp
import pytest
from prxteinmpnn.utils.data_structures import GMM, EMFitterResult

from prxteinmpnn.ensemble.em_fit import (
    _e_step,
    _m_step_from_responsibilities,
    fit_gmm_states,
    log_likelihood,
)


@pytest.fixture
def sample_data() -> jax.Array:
    """Generate sample 2D data for testing."""
    key = jax.random.PRNGKey(42)
    return jax.random.normal(key, (100, 2))


@pytest.fixture
def initial_gmm(sample_data) -> GMM:
    """Create an initial GMM for testing."""
    n_components = 2
    n_features = sample_data.shape[1]
    return GMM(
        means=jax.random.normal(jax.random.PRNGKey(0), (n_components, n_features)),
        covariances=jnp.array([jnp.eye(n_features)] * n_components),
        weights=jnp.ones(n_components) / n_components,
        responsibilities=jnp.zeros((sample_data.shape[0], n_components)),
        n_components=n_components,
        n_features=n_features,
    )


class TestLogLikelihood:
    """Test the log_likelihood function."""

    def test_log_likelihood_diag(self, sample_data, initial_gmm):
        """Test log_likelihood with diagonal covariances."""
        diag_covariances = jnp.array([jnp.diag(cov) for cov in initial_gmm.covariances])
        ll = log_likelihood(sample_data, initial_gmm.means, diag_covariances)
        chex.assert_shape(ll, (sample_data.shape[0], initial_gmm.n_components))

    def test_log_likelihood_full(self, sample_data, initial_gmm):
        """Test log_likelihood with full covariances."""
        ll = log_likelihood(sample_data, initial_gmm.means, initial_gmm.covariances)
        chex.assert_shape(ll, (sample_data.shape[0], initial_gmm.n_components))


class TestEStep:
    """Test the E-step of the EM algorithm."""

    def test_e_step_returns_correct_shapes(
        self,
        sample_data: jax.Array,
        initial_gmm: GMM,
    ) -> None:
        """Test that E-step returns arrays with correct shapes."""
        log_likelihood, log_resp = _e_step(sample_data, initial_gmm)

        assert log_likelihood.shape == ()
        assert log_resp.shape == (sample_data.shape[0], initial_gmm.n_components)

    def test_e_step_log_responsibilities_sum_to_zero(
        self,
        sample_data: jax.Array,
        initial_gmm: GMM,
    ) -> None:
        """Test that log responsibilities sum to zero (probabilities sum to 1)."""
        _, log_resp = _e_step(sample_data, initial_gmm)
        resp_sum = jnp.sum(jnp.exp(log_resp), axis=1)

        chex.assert_trees_all_close(resp_sum, jnp.ones(sample_data.shape[0]), rtol=1e-6)

    def test_e_step_log_likelihood_is_finite(
        self,
        sample_data: jax.Array,
        initial_gmm: GMM,
    ) -> None:
        """Test that log likelihood is finite."""
        log_likelihood, _ = _e_step(sample_data, initial_gmm)

        assert jnp.isfinite(log_likelihood)


class TestMStepFromResponsibilities:
    """Test the M-step using responsibilities."""

    def test_m_step_returns_valid_gmm(
        self,
        sample_data: jax.Array,
        initial_gmm: GMM,
    ) -> None:
        """Test that M-step returns a valid GMM."""
        _, log_resp = _e_step(sample_data, initial_gmm)
        resp = jnp.exp(log_resp)

        weights, means, covariances = _m_step_from_responsibilities(
            sample_data,
            initial_gmm.means,
            initial_gmm.covariances,
            resp,
            covariance_regularization=1e-6,
        )

        assert weights.shape == (initial_gmm.n_components,)
        assert means.shape == (initial_gmm.n_components, initial_gmm.n_features)
        assert covariances.shape == (
            initial_gmm.n_components,
            initial_gmm.n_features,
            initial_gmm.n_features,
        )
        assert jnp.allclose(jnp.sum(weights), 1.0)

    def test_m_step_weights_are_positive(
        self,
        sample_data: jax.Array,
        initial_gmm: GMM,
    ) -> None:
        """Test that updated weights are positive."""
        _, log_resp = _e_step(sample_data, initial_gmm)
        resp = jnp.exp(log_resp)

        weights, _, _ = _m_step_from_responsibilities(
            sample_data,
            initial_gmm.means,
            initial_gmm.covariances,
            resp,
            covariance_regularization=1e-6,
        )

        assert jnp.all(weights >= 0)


class TestFitGMMStates:
    """Test the in-memory GMM fitting function."""

    def test_fit_gmm_in_memory_returns_result(
        self,
        sample_data: jax.Array,
        initial_gmm: GMM,
    ) -> None:
        """Test that in-memory fitting returns an EMFitterResult."""
        result = fit_gmm_states(sample_data, initial_gmm, max_iter=5)

        assert isinstance(result, EMFitterResult)
        assert result.n_iter <= 5
        assert isinstance(result.converged, jax.Array)

    def test_fit_gmm_in_memory_improves_likelihood(
        self,
        sample_data: jax.Array,
        initial_gmm: GMM,
    ) -> None:
        """Test that fitting improves log likelihood."""
        initial_ll, _ = _e_step(sample_data, initial_gmm)
        result = fit_gmm_states(sample_data, initial_gmm, max_iter=10)

        assert result.log_likelihood >= initial_ll

    def test_fit_gmm_in_memory_respects_max_iter(
        self,
        sample_data: jax.Array,
        initial_gmm: GMM,
    ) -> None:
        """Test that fitting respects max_iter parameter."""
        max_iter = 3
        result = fit_gmm_states(sample_data, initial_gmm, max_iter=max_iter)

        assert result.n_iter <= max_iter

    def test_fit_gmm_in_memory_convergence_with_tight_tolerance(
        self,
        sample_data: jax.Array,
        initial_gmm: GMM,
    ) -> None:
        """Test convergence with tight tolerance."""
        result = fit_gmm_states(sample_data, initial_gmm, max_iter=100, tol=1e-8)

        if result.converged:
            assert result.log_likelihood_diff < 1e-8