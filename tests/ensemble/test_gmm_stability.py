"""Test GMM stability improvements for overlapping energy wells in protein conformations.

This test suite validates the stability enhancements for GMM fitting in the context
of protein conformational analysis, where overlapping mixture components represent
energy wells that are later refined by DBSCAN clustering.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from prxteinmpnn.ensemble.bic import compute_bic


class TestBICCalculation:
  """Test Bayesian Information Criterion for model selection."""

  def test_bic_full_covariance(self):
    """Test BIC calculation with full covariance matrices.

    Returns:
      None

    Raises:
      AssertionError: If BIC value is not finite or has wrong sign.

    Example:
      >>> test_bic_full_covariance()

    """
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
    """Test BIC calculation with diagonal covariance matrices.

    Returns:
      None

    Raises:
      AssertionError: If BIC value is incorrect.

    Example:
      >>> test_bic_diagonal_covariance()

    """
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
    """Test that BIC penalty increases with model complexity.

    Returns:
      None

    Raises:
      AssertionError: If BIC doesn't increase with more components.

    Example:
      >>> test_bic_penalty_increases_with_components()

    """
    log_likelihood = -1000.0
    n_samples = 500
    n_features = 10

    bic_3 = compute_bic(log_likelihood, n_samples, 3, n_features, "full")
    bic_5 = compute_bic(log_likelihood, n_samples, 5, n_features, "full")

    assert bic_5 > bic_3, "BIC should increase with more components (for same log-likelihood)"


