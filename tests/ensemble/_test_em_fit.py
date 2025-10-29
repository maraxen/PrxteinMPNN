from collections.abc import Generator
from typing import Any
import jax
import pytest
import chex
import numpy as np
from gmmx import GaussianMixtureModelJax
from jax import numpy as jnp

"""Tests for Expectation-Maximization algorithm for Gaussian Mixture Models."""



from prxteinmpnn.ensemble.em_fit import (
  EMFitterResult,
  _e_step,
  _m_step_from_responsibilities,
  GMM,
)


@pytest.fixture
def sample_data() -> jax.Array:
  """Generate sample 2D data for testing.

  Returns:
      jax.Array: Sample data array of shape (100, 2).

  """
  key = jax.random.PRNGKey(42)
  return jax.random.normal(key, (100, 2))


@pytest.fixture
def initial_gmm() -> GaussianMixtureModelJax:
  """Create an initial GMM for testing.

  Returns:
      GaussianMixtureModelJax: Initial GMM with 2 components and 2 features.

  """
  return GaussianMixtureModelJax.create(
    n_components=2,
    n_features=2,
  )


class TestEStep:
  """Test the E-step of the EM algorithm."""

  def test_e_step_returns_correct_shapes(
    self,
    sample_data: jax.Array,
    initial_gmm: GaussianMixtureModelJax,
  ) -> None:
    """Test that E-step returns arrays with correct shapes.

    Args:
        sample_data: Sample data for testing.
        initial_gmm: Initial GMM for testing.

    """
    log_likelihood, log_resp = _e_step(sample_data, initial_gmm)

    assert log_likelihood.shape == ()
    assert log_resp.shape == (sample_data.shape[0], initial_gmm.n_components, 1, 1)

  def test_e_step_log_responsibilities_sum_to_zero(
    self,
    sample_data: jax.Array,
    initial_gmm: GaussianMixtureModelJax,
  ) -> None:
    """Test that log responsibilities sum to zero (probabilities sum to 1).

    Args:
        sample_data: Sample data for testing.
        initial_gmm: Initial GMM for testing.

    """
    _, log_resp = _e_step(sample_data, initial_gmm)
    resp_sum = jnp.sum(jnp.exp(log_resp), axis=1)

    chex.assert_trees_all_close(jnp.squeeze(resp_sum), 1.0, rtol=1e-6)

  def test_e_step_log_likelihood_is_finite(
    self,
    sample_data: jax.Array,
    initial_gmm: GaussianMixtureModelJax,
  ) -> None:
    """Test that log likelihood is finite.

    Args:
        sample_data: Sample data for testing.
        initial_gmm: Initial GMM for testing.

    """
    log_likelihood, _ = _e_step(sample_data, initial_gmm)

    assert jnp.isfinite(log_likelihood)


class TestMStepFromResponsibilities:
  """Test the M-step using responsibilities."""

  def test_m_step_returns_valid_gmm(
    self,
    sample_data: jax.Array,
    initial_gmm: GaussianMixtureModelJax,
  ) -> None:
    """Test that M-step returns a valid GMM.

    Args:
        sample_data: Sample data for testing.
        initial_gmm: Initial GMM for testing.

    """
    _, log_resp = _e_step(sample_data, initial_gmm)
    resp = jnp.exp(log_resp)

    updated_gmm = _m_step_from_responsibilities(
      sample_data,
      resp,
      initial_gmm,
      covariance_regularization=1e-6,
    )

    assert updated_gmm.n_components == initial_gmm.n_components
    assert updated_gmm.n_features == initial_gmm.n_features
    assert jnp.allclose(jnp.sum(updated_gmm.weights), 1.0)

  def test_m_step_weights_are_positive(
    self,
    sample_data: jax.Array,
    initial_gmm: GaussianMixtureModelJax,
  ) -> None:
    """Test that updated weights are positive.

    Args:
        sample_data: Sample data for testing.
        initial_gmm: Initial GMM for testing.

    """
    _, log_resp = _e_step(sample_data, initial_gmm)
    resp = jnp.exp(log_resp)

    updated_gmm = _m_step_from_responsibilities(
      sample_data,
      resp,
      initial_gmm,
      covariance_regularization=1e-6,
    )

    assert jnp.all(updated_gmm.weights >= 0)






class TestEMFitterResult:
  """Test the EMFitterResult dataclass."""

  def test_em_fitter_result_creation(self, initial_gmm: GaussianMixtureModelJax) -> None:
    """Test EMFitterResult creation.

    Args:
        initial_gmm: Initial GMM for testing.

    """
    result = EMFitterResult(
      gmm=initial_gmm,
      n_iter=10,
      log_likelihood=jnp.array(-100.0),
      log_likelihood_diff=jnp.array(0.001),
      converged=jnp.bool_(True),
    )

    assert result.gmm is initial_gmm
    assert result.n_iter == 10
    assert result.log_likelihood == -100.0
    assert result.log_likelihood_diff == 0.001
    assert bool(result.converged) is True

  def test_em_fitter_result_attributes_types(self, initial_gmm: GaussianMixtureModelJax) -> None:
    """Test that EMFitterResult attributes have correct types.

    Args:
        initial_gmm: Initial GMM for testing.

    """
    result = EMFitterResult(
      gmm=initial_gmm,
      n_iter=5,
      log_likelihood=jnp.array(-50.0),
      log_likelihood_diff=jnp.array(0.01),
      converged=jnp.bool_(False),
    )

    assert isinstance(result.gmm, GaussianMixtureModelJax)
    assert isinstance(result.n_iter, int)
    assert isinstance(result.log_likelihood, jax.Array)
    assert isinstance(result.log_likelihood_diff, jax.Array)
    assert isinstance(bool(result.converged), bool)
