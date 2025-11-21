import jax
import pytest
from gmmx import GaussianMixtureModelJax
from jax import numpy as jnp

"""Tests for Expectation-Maximization algorithm for Gaussian Mixture Models."""



from prxteinmpnn.ensemble.em_fit import (
  EMFitterResult,
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


