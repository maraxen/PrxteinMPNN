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
  _m_step_from_stats,
  fit_gmm_generator,
  fit_gmm_in_memory,
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


class TestMStepFromStats:
  """Test the M-step using sufficient statistics."""

  def test_m_step_from_stats_returns_valid_gmm(self, initial_gmm: GaussianMixtureModelJax) -> None:
    """Test that M-step from stats returns a valid GMM.
    
    Args:
        initial_gmm: Initial GMM for testing.
        
    """
    # Create mock sufficient statistics
    nk = jnp.array([30.0, 70.0])
    xk = jnp.array([[10.0, 20.0], [30.0, 40.0]])
    sk = jnp.array([[[100.0, 50.0], [50.0, 200.0]], [[300.0, 150.0], [150.0, 400.0]]])
    n_total_samples = 100
    
    updated_gmm = _m_step_from_stats(
      initial_gmm,
      nk,
      xk,
      sk,
      n_total_samples,
      covariance_regularization=1e-6,
    )
    
    assert updated_gmm.n_components == initial_gmm.n_components
    assert updated_gmm.n_features == initial_gmm.n_features
    assert jnp.allclose(jnp.sum(updated_gmm.weights), 1.0)
    assert jnp.all(updated_gmm.weights >= 0)

  def test_m_step_from_stats_correct_weights(self, initial_gmm: GaussianMixtureModelJax) -> None:
    """Test that weights are computed correctly from statistics.
    
    Args:
        initial_gmm: Initial GMM for testing.
        
    """
    nk = jnp.array([30.0, 70.0])
    xk = jnp.array([[10.0, 20.0], [30.0, 40.0]])
    sk = jnp.array([[[100.0, 50.0], [50.0, 200.0]], [[300.0, 150.0], [150.0, 400.0]]])
    n_total_samples = 100
    
    updated_gmm = _m_step_from_stats(
      initial_gmm,
      nk,
      xk,
      sk,
      n_total_samples,
      covariance_regularization=1e-6,
    )
    
    expected_weights = nk / n_total_samples
    jnp.testing.assert_allclose(updated_gmm.weights, expected_weights)


class TestFitGMMInMemory:
  """Test the in-memory GMM fitting function."""

  def test_fit_gmm_in_memory_returns_result(
    self,
    sample_data: jax.Array,
    initial_gmm: GaussianMixtureModelJax,
  ) -> None:
    """Test that in-memory fitting returns an EMFitterResult.
    
    Args:
        sample_data: Sample data for testing.
        initial_gmm: Initial GMM for testing.
        
    """
    result = fit_gmm_in_memory(sample_data, initial_gmm, max_iter=5)
    
    assert isinstance(result, EMFitterResult)
    assert result.n_iter <= 5
    assert isinstance(result.converged, bool)

  def test_fit_gmm_in_memory_improves_likelihood(
    self,
    sample_data: jax.Array,
    initial_gmm: GaussianMixtureModelJax,
  ) -> None:
    """Test that fitting improves log likelihood.
    
    Args:
        sample_data: Sample data for testing.
        initial_gmm: Initial GMM for testing.
        
    """
    initial_ll, _ = _e_step(sample_data, initial_gmm)
    result = fit_gmm_in_memory(sample_data, initial_gmm, max_iter=10)
    
    assert result.log_likelihood >= initial_ll

  def test_fit_gmm_in_memory_respects_max_iter(
    self,
    sample_data: jax.Array,
    initial_gmm: GaussianMixtureModelJax,
  ) -> None:
    """Test that fitting respects max_iter parameter.
    
    Args:
        sample_data: Sample data for testing.
        initial_gmm: Initial GMM for testing.
        
    """
    max_iter = 3
    result = fit_gmm_in_memory(sample_data, initial_gmm, max_iter=max_iter)
    
    assert result.n_iter <= max_iter

  def test_fit_gmm_in_memory_convergence_with_tight_tolerance(
    self,
    sample_data: jax.Array,
    initial_gmm: GaussianMixtureModelJax,
  ) -> None:
    """Test convergence with tight tolerance.
    
    Args:
        sample_data: Sample data for testing.
        initial_gmm: Initial GMM for testing.
        
    """
    result = fit_gmm_in_memory(sample_data, initial_gmm, max_iter=100, tol=1e-8)
    
    if result.converged:
      assert result.log_likelihood_diff < 1e-8


class TestFitGMMGenerator:
  """Test the generator-based GMM fitting function."""

  def _create_data_generator(self, data: jax.Array, batch_size: int) -> Generator[jax.Array, None, None]:
    """Create a data generator for testing.
    
    Args:
        data: Data to generate batches from.
        batch_size: Size of each batch.
        
    Yields:
        jax.Array: Batches of data.
        
    """
    for i in range(0, data.shape[0], batch_size):
      yield data[i:i + batch_size]

  def test_fit_gmm_generator_returns_result(
    self,
    sample_data: jax.Array,
    initial_gmm: GaussianMixtureModelJax,
  ) -> None:
    """Test that generator-based fitting returns an EMFitterResult.
    
    Args:
        sample_data: Sample data for testing.
        initial_gmm: Initial GMM for testing.
        
    """
    data_gen = self._create_data_generator(sample_data, batch_size=20)
    result = fit_gmm_generator(
      data_gen,
      initial_gmm,
      n_total_samples=sample_data.shape[0],
      max_iter=5,
    )
    
    assert isinstance(result, EMFitterResult)
    assert result.n_iter <= 5
    assert isinstance(result.converged, bool)

  def test_fit_gmm_generator_respects_max_iter(
    self,
    sample_data: jax.Array,
    initial_gmm: GaussianMixtureModelJax,
  ) -> None:
    """Test that generator fitting respects max_iter parameter.
    
    Args:
        sample_data: Sample data for testing.
        initial_gmm: Initial GMM for testing.
        
    """
    max_iter = 3
    data_gen = self._create_data_generator(sample_data, batch_size=20)
    result = fit_gmm_generator(
      data_gen,
      initial_gmm,
      n_total_samples=sample_data.shape[0],
      max_iter=max_iter,
    )
    
    assert result.n_iter <= max_iter

  def test_fit_gmm_generator_convergence(
    self,
    sample_data: jax.Array,
    initial_gmm: GaussianMixtureModelJax,
  ) -> None:
    """Test that generator fitting can converge.
    
    Args:
        sample_data: Sample data for testing.
        initial_gmm: Initial GMM for testing.
        
    """
    data_gen = self._create_data_generator(sample_data, batch_size=20)
    result = fit_gmm_generator(
      data_gen,
      initial_gmm,
      n_total_samples=sample_data.shape[0],
      max_iter=50,
      tol=1e-3,
    )
    
    if result.converged:
      assert result.log_likelihood_diff < 1e-3

  def test_fit_gmm_generator_with_single_batch(
    self,
    sample_data: jax.Array,
    initial_gmm: GaussianMixtureModelJax,
  ) -> None:
    """Test generator fitting with a single batch (equivalent to in-memory).
    
    Args:
        sample_data: Sample data for testing.
        initial_gmm: Initial GMM for testing.
        
    """
    data_gen = self._create_data_generator(sample_data, batch_size=sample_data.shape[0])
    result = fit_gmm_generator(
      data_gen,
      initial_gmm,
      n_total_samples=sample_data.shape[0],
      max_iter=10,
    )
    
    assert isinstance(result, EMFitterResult)
    assert result.gmm.n_components == initial_gmm.n_components


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
      converged=True,
    )
    
    assert result.gmm is initial_gmm
    assert result.n_iter == 10
    assert result.log_likelihood == -100.0
    assert result.log_likelihood_diff == 0.001
    assert result.converged is True

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
      converged=False,
    )
    
    assert isinstance(result.gmm, GaussianMixtureModelJax)
    assert isinstance(result.n_iter, int)
    assert isinstance(result.log_likelihood, jax.Array)
    assert isinstance(result.log_likelihood_diff, jax.Array)
    assert isinstance(result.converged, bool)


class TestEdgeCases:
  """Test edge cases and error conditions."""

  def test_empty_data_generator(self, initial_gmm: GaussianMixtureModelJax) -> None:
    """Test behavior with empty data generator.
    
    Args:
        initial_gmm: Initial GMM for testing.
        
    """
    def empty_generator() -> Generator[jax.Array, None, None]:
      return
      yield  # This line is never reached but satisfies type checker

    result = fit_gmm_generator(
      empty_generator(),
      initial_gmm,
      n_total_samples=0,
      max_iter=1,
    )
    
    assert isinstance(result, EMFitterResult)
    assert result.n_iter == 0

  def test_single_component_gmm(self, sample_data: jax.Array) -> None:
    """Test fitting with single component GMM.
    
    Args:
        sample_data: Sample data for testing.
        
    """
    single_gmm = GaussianMixtureModelJax.create(
      n_components=1,
      n_features=2,
    )
    
    result = fit_gmm_in_memory(sample_data, single_gmm, max_iter=5)
    
    assert result.gmm.n_components == 1
    assert jnp.allclose(result.gmm.weights, 1.0)

  def test_regularization_parameter(
    self,
    sample_data: jax.Array,
    initial_gmm: GaussianMixtureModelJax,
  ) -> None:
    """Test that regularization parameter affects covariance matrices.
    
    Args:
        sample_data: Sample data for testing.
        initial_gmm: Initial GMM for testing.
        
    """
    result_low_reg = fit_gmm_in_memory(sample_data, initial_gmm, covariance_regularization=1e-10, max_iter=2)
    result_high_reg = fit_gmm_in_memory(sample_data, initial_gmm, covariance_regularization=1e-2, max_iter=2)
    
    # High regularization should result in larger diagonal elements in the covariance
    diag_low = jnp.trace(result_low_reg.gmm.covariances.values, axis1=1, axis2=2)
    diag_high = jnp.trace(result_high_reg.gmm.covariances.values, axis1=1, axis2=2)
    
    assert jnp.all(diag_high >= diag_low)
