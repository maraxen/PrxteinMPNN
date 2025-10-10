"""Test Bayesian Information Criterion for model selection."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from prxteinmpnn.ensemble.bic import compute_bic


class TestBICCalculation:
    """Test Bayesian Information Criterion for model selection."""

    def test_bic_full_covariance(self):
        """Test BIC calculation with full covariance matrices."""
        log_likelihood = -1000.0
        n_samples = 500
        n_components = 3
        n_features = 10

        bic = compute_bic(
            log_likelihood=log_likelihood,
            n_samples=n_samples,
            n_components=n_components,
            n_features=n_features,
            covariance_type="full",
        )

        assert jnp.isfinite(bic), "BIC should be finite"
        assert bic > 0, "BIC should be positive for negative log-likelihood"

    def test_bic_diagonal_covariance(self):
        """Test BIC calculation with diagonal covariance matrices."""
        log_likelihood = -1000.0
        n_samples = 500
        n_components = 3
        n_features = 10

        bic = compute_bic(
            log_likelihood=log_likelihood,
            n_samples=n_samples,
            n_components=n_components,
            n_features=n_features,
            covariance_type="diag",
        )

        assert jnp.isfinite(bic), "BIC should be finite"

    def test_bic_penalty_increases_with_components(self):
        """Test that BIC penalty increases with model complexity."""
        log_likelihood = -1000.0
        n_samples = 500
        n_features = 10

        bic_3 = compute_bic(log_likelihood, n_samples, 3, n_features, "full")
        bic_5 = compute_bic(log_likelihood, n_samples, 5, n_features, "full")

        assert bic_5 > bic_3, "BIC should increase with more components (for same log-likelihood)"

    def test_bic_penalty_increases_with_features(self):
        """Test that BIC penalty increases with feature count."""
        log_likelihood = -1000.0
        n_samples = 500
        n_components = 3

        bic_10 = compute_bic(log_likelihood, n_samples, n_components, 10, "full")
        bic_20 = compute_bic(log_likelihood, n_samples, n_components, 20, "full")

        assert bic_20 > bic_10, "BIC should increase with more features"

    @pytest.mark.parametrize("covariance_type", ["full", "diag"])
    def test_bic_value(self, covariance_type):
        """Test the BIC value against a known result."""
        log_likelihood = -1000.0
        n_samples = 500
        n_components = 3
        n_features = 10

        if covariance_type == "full":
            # n_mean_params = 3 * 10 = 30
            # n_cov_params = 3 * 10 * 11 / 2 = 165
            # n_weight_params = 2
            # n_params = 30 + 165 + 2 = 197
            # bic = -2 * -1000 + 197 * log(500) = 2000 + 197 * 6.2146 = 3222.27
            expected_bic = 3222.27
        else:
            # n_mean_params = 3 * 10 = 30
            # n_cov_params = 3 * 10 = 30
            # n_weight_params = 2
            # n_params = 30 + 30 + 2 = 62
            # bic = -2 * -1000 + 62 * log(500) = 2000 + 62 * 6.2146 = 2385.3
            expected_bic = 2385.3

        bic = compute_bic(
            log_likelihood=log_likelihood,
            n_samples=n_samples,
            n_components=n_components,
            n_features=n_features,
            covariance_type=covariance_type,
        )

        assert jnp.isclose(bic, expected_bic, rtol=1e-2)