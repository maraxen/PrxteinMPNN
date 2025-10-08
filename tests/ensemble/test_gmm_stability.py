"""Test GMM stability improvements for overlapping energy wells in protein conformations.

This test suite validates the stability enhancements for GMM fitting in the context
of protein conformational analysis, where overlapping mixture components represent
energy wells that are later refined by DBSCAN clustering.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from prxteinmpnn.ensemble.gmm import (
  compute_bic,
  gmm_from_responsibilities,
  prune_components,
)


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


class TestComponentPruning:
  """Test component pruning for GMM stability."""

  @pytest.fixture
  def sample_gmm(self):
    """Create a sample GMM with components of varying weights.

    Returns:
      GMM: A sample Gaussian Mixture Model.

    Example:
      >>> gmm = sample_gmm()

    """
    from prxteinmpnn.ensemble.em_fit import GMM

    n_components = 5
    n_features = 3
    n_samples = 100

    # Create GMM with one very small weight and one very large weight
    weights = jnp.array([0.0005, 0.15, 0.30, 0.25, 0.2995])  # Sum to 1
    means = jax.random.normal(jax.random.PRNGKey(42), (n_components, n_features))
    covariances = jnp.tile(jnp.eye(n_features), (n_components, 1, 1))
    responsibilities = jnp.zeros((n_samples, n_components))

    return GMM(
      weights=weights,
      means=means,
      covariances=covariances,
      responsibilities=responsibilities,
      n_components=n_components,
      n_features=n_features,
    )

  def test_prune_small_weights(self, sample_gmm):
    """Test that components with very small weights are pruned.

    Args:
      sample_gmm: Sample GMM fixture.

    Returns:
      None

    Raises:
      AssertionError: If pruning doesn't work correctly.

    Example:
      >>> test_prune_small_weights(sample_gmm)

    """
    pruned_gmm, n_removed = prune_components(
      sample_gmm,
      min_weight=1e-3,
      max_weight=0.99,
    )

    assert n_removed > 0, "Should have removed at least one component"
    assert pruned_gmm.n_components < sample_gmm.n_components
    assert jnp.all(pruned_gmm.weights >= 1e-3), "All remaining weights should be >= min_weight"

  def test_weights_renormalized_after_pruning(self, sample_gmm):
    """Test that weights are properly renormalized after pruning.

    Args:
      sample_gmm: Sample GMM fixture.

    Returns:
      None

    Raises:
      AssertionError: If weights don't sum to 1 after pruning.

    Example:
      >>> test_weights_renormalized_after_pruning(sample_gmm)

    """
    pruned_gmm, _ = prune_components(
      sample_gmm,
      min_weight=1e-3,
      max_weight=0.99,
    )

    weight_sum = jnp.sum(pruned_gmm.weights)
    assert jnp.isclose(weight_sum, 1.0, atol=1e-6), "Weights should sum to 1 after pruning"

  def test_no_pruning_when_not_needed(self, sample_gmm):
    """Test that no pruning occurs when all components are valid.

    Args:
      sample_gmm: Sample GMM fixture.

    Returns:
      None

    Raises:
      AssertionError: If unnecessary pruning occurs.

    Example:
      >>> test_no_pruning_when_not_needed(sample_gmm)

    """
    # Set very permissive thresholds
    pruned_gmm, n_removed = prune_components(
      sample_gmm,
      min_weight=1e-5,
      max_weight=0.999,
    )

    assert int(n_removed) == 0, "Should not remove components with permissive thresholds"
    assert pruned_gmm.n_components == sample_gmm.n_components


class TestVarianceConstraints:
  """Test that variance constraints ensure positive definite covariance matrices."""

  def test_positive_variance_full_covariance(self):
    """Test that full covariance matrices maintain positive diagonal values.

    Returns:
      None

    Raises:
      AssertionError: If variance values are not positive.

    Example:
      >>> test_positive_variance_full_covariance()

    """
    key = jax.random.PRNGKey(42)
    n_samples = 100
    n_components = 3
    n_features = 5

    # Create synthetic data
    data = jax.random.normal(key, (n_samples, n_features))
    means = jax.random.normal(key, (n_components, n_features))

    # Create responsibilities (hard assignment for simplicity)
    labels = jax.random.randint(key, (n_samples,), 0, n_components)
    responsibilities = jax.nn.one_hot(labels, n_components)
    nk = jnp.sum(responsibilities, axis=0)

    gmm = gmm_from_responsibilities(
      data=data,
      means=means,
      responsibilities=responsibilities,
      nk=nk,
      covariance_type="full",
      covariance_regularization=1e-6,
      min_variance=1e-3,
    )

    # Check that all diagonal elements are positive
    for k in range(n_components):
      diag_values = jnp.diag(gmm.covariances[k])
      assert jnp.all(diag_values > 0), f"Component {k} has non-positive variance"
      assert jnp.all(diag_values >= 1e-3), f"Component {k} variance below minimum"

  def test_positive_variance_diagonal_covariance(self):
    """Test that diagonal covariance matrices maintain positive values.

    Returns:
      None

    Raises:
      AssertionError: If variance values are not positive.

    Example:
      >>> test_positive_variance_diagonal_covariance()

    """
    key = jax.random.PRNGKey(42)
    n_samples = 100
    n_components = 3
    n_features = 5

    data = jax.random.normal(key, (n_samples, n_features))
    means = jax.random.normal(key, (n_components, n_features))

    labels = jax.random.randint(key, (n_samples,), 0, n_components)
    responsibilities = jax.nn.one_hot(labels, n_components)
    nk = jnp.sum(responsibilities, axis=0)

    gmm = gmm_from_responsibilities(
      data=data,
      means=means,
      responsibilities=responsibilities,
      nk=nk,
      covariance_type="diag",
      covariance_regularization=1e-6,
      min_variance=1e-3,
    )

    # Check that all variances are positive
    assert jnp.all(gmm.covariances > 0), "All variances should be positive"
    assert jnp.all(gmm.covariances >= 1e-3), "All variances should be >= min_variance"

  def test_softplus_transformation_prevents_negative_variance(self):
    """Test that softplus transformation prevents negative variance values.

    Returns:
      None

    Raises:
      AssertionError: If negative variance can occur.

    Example:
      >>> test_softplus_transformation_prevents_negative_variance()

    """
    # Test the softplus transformation directly
    min_variance = 1e-3

    # Even with very negative input, output should be >= min_variance
    negative_values = jnp.array([-100.0, -10.0, -1.0, -0.1])
    transformed = jax.nn.softplus(negative_values - min_variance) + min_variance

    assert jnp.all(transformed >= min_variance), "Softplus should enforce minimum variance"
    assert jnp.all(jnp.isfinite(transformed)), "Transformed values should be finite"


class TestWeightConstraints:
  """Test that weight constraints ensure valid probability distributions."""

  def test_weights_sum_to_one(self):
    """Test that GMM weights always sum to 1.

    Returns:
      None

    Raises:
      AssertionError: If weights don't sum to 1.

    Example:
      >>> test_weights_sum_to_one()

    """
    key = jax.random.PRNGKey(42)
    n_samples = 100
    n_components = 3
    n_features = 5

    data = jax.random.normal(key, (n_samples, n_features))
    means = jax.random.normal(key, (n_components, n_features))

    labels = jax.random.randint(key, (n_samples,), 0, n_components)
    responsibilities = jax.nn.one_hot(labels, n_components)
    nk = jnp.sum(responsibilities, axis=0)

    gmm = gmm_from_responsibilities(
      data=data,
      means=means,
      responsibilities=responsibilities,
      nk=nk,
      covariance_type="diag",
    )

    weight_sum = jnp.sum(gmm.weights)
    assert jnp.isclose(weight_sum, 1.0, atol=1e-6), "Weights should sum to 1"

  def test_weights_are_positive(self):
    """Test that all GMM weights are positive.

    Returns:
      None

    Raises:
      AssertionError: If any weights are negative.

    Example:
      >>> test_weights_are_positive()

    """
    key = jax.random.PRNGKey(42)
    n_samples = 100
    n_components = 3
    n_features = 5

    data = jax.random.normal(key, (n_samples, n_features))
    means = jax.random.normal(key, (n_components, n_features))

    labels = jax.random.randint(key, (n_samples,), 0, n_components)
    responsibilities = jax.nn.one_hot(labels, n_components)
    nk = jnp.sum(responsibilities, axis=0)

    gmm = gmm_from_responsibilities(
      data=data,
      means=means,
      responsibilities=responsibilities,
      nk=nk,
      covariance_type="diag",
    )

    assert jnp.all(gmm.weights >= 0), "All weights should be non-negative"
    assert jnp.all(gmm.weights <= 1), "All weights should be <= 1"
