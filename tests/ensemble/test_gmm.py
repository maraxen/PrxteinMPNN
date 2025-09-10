"""Tests for GMM fitting functionality."""

import jax
import jax.numpy as jnp
import pytest
from unittest.mock import Mock, patch

from prxteinmpnn.ensemble.gmm import make_fit_gmm


@pytest.fixture
def mock_logits():
  """Create mock logits data for testing.
  
  Returns:
    Mock logits array of shape (n_timesteps, n_features).
  """
  n_timesteps = 50
  n_features = 21
  key = jax.random.PRNGKey(42)
  return jax.random.normal(key, (n_timesteps, n_features))


@patch('prxteinmpnn.ensemble.gmm.GaussianMixtureModelJax')
@patch('prxteinmpnn.ensemble.gmm.EMFitter')
def test_make_fit_gmm_default_parameters(mock_em_fitter, mock_gmm, mock_logits):
  """Test GMM fitting with default parameters.
  
  Args:
    mock_em_fitter: Mock EMFitter class.
    mock_gmm: Mock GaussianMixtureModelJax class.
    mock_logits: Mock logits fixture.
    
  Raises:
    AssertionError: If GMM fitting does not work correctly.
    
  Example:
    >>> test_make_fit_gmm_default_parameters(...)
  """
  # Setup mocks
  mock_gmm_instance = Mock()
  mock_gmm.create.return_value = mock_gmm_instance
  
  mock_fitter_instance = Mock()
  mock_fitted_gmm = Mock()
  mock_fitter_instance.fit.return_value = mock_fitted_gmm
  mock_em_fitter.return_value = mock_fitter_instance
  
  # Test GMM fitting function creation
  gmm_fit_fn = make_fit_gmm()
  
  # Verify GMM was created with default parameters
  mock_gmm.create.assert_called_once_with(n_components=100, n_features=21)
  
  # Verify EMFitter was created with correct parameters
  mock_em_fitter.assert_called_once_with(tol=1e-3, max_iter=100, reg_covar=1e-6)
  
  # Test fitting
  result = gmm_fit_fn(mock_logits)
  
  # Verify fit was called correctly
  mock_fitter_instance.fit.assert_called_once_with(mock_logits, gmm=mock_gmm_instance)
  
  # Verify result is the fitted GMM
  assert result == mock_fitted_gmm


@patch('prxteinmpnn.ensemble.gmm.GaussianMixtureModelJax')
@patch('prxteinmpnn.ensemble.gmm.EMFitter')
def test_make_fit_gmm_custom_parameters(mock_em_fitter, mock_gmm, mock_logits):
  """Test GMM fitting with custom parameters.
  
  Args:
    mock_em_fitter: Mock EMFitter class.
    mock_gmm: Mock GaussianMixtureModelJax class.
    mock_logits: Mock logits fixture.
    
  Raises:
    AssertionError: If custom parameters are not used correctly.
    
  Example:
    >>> test_make_fit_gmm_custom_parameters(...)
  """
  # Setup mocks
  mock_gmm_instance = Mock()
  mock_gmm.create.return_value = mock_gmm_instance
  
  mock_fitter_instance = Mock()
  mock_fitted_gmm = Mock()
  mock_fitter_instance.fit.return_value = mock_fitted_gmm
  mock_em_fitter.return_value = mock_fitter_instance
  
  # Test with custom parameters
  custom_n_components = 50
  custom_n_features = 15
  
  gmm_fit_fn = make_fit_gmm(
    n_components=custom_n_components,
    n_features=custom_n_features
  )
  
  # Verify GMM was created with custom parameters
  mock_gmm.create.assert_called_once_with(
    n_components=custom_n_components,
    n_features=custom_n_features
  )
  
  # Test fitting
  result = gmm_fit_fn(mock_logits)
  
  # Verify result
  assert result == mock_fitted_gmm


def test_make_fit_gmm_return_type():
  """Test that make_fit_gmm returns a callable.
  
  Raises:
    AssertionError: If return type is not callable.
    
  Example:
    >>> test_make_fit_gmm_return_type()
  """
  with patch('prxteinmpnn.ensemble.gmm.GaussianMixtureModelJax'), \
       patch('prxteinmpnn.ensemble.gmm.EMFitter'):
    
    gmm_fit_fn = make_fit_gmm()
    assert callable(gmm_fit_fn), "make_fit_gmm should return a callable function"


@patch('prxteinmpnn.ensemble.gmm.GaussianMixtureModelJax')
@patch('prxteinmpnn.ensemble.gmm.EMFitter')
def test_make_fit_gmm_function_signature(mock_em_fitter, mock_gmm):
  """Test that the returned function has correct signature.
  
  Args:
    mock_em_fitter: Mock EMFitter class.
    mock_gmm: Mock GaussianMixtureModelJax class.
    
  Raises:
    AssertionError: If function signature is incorrect.
    
  Example:
    >>> test_make_fit_gmm_function_signature(...)
  """
  # Setup mocks
  mock_gmm.create.return_value = Mock()
  mock_fitter_instance = Mock()
  mock_fitter_instance.fit.return_value = Mock()
  mock_em_fitter.return_value = mock_fitter_instance
  
  gmm_fit_fn = make_fit_gmm()
  
  # Test that function accepts logits array
  mock_logits = jnp.ones((10, 21))
  
  # Should not raise an exception
  result = gmm_fit_fn(mock_logits)
  assert result is not None


@patch('prxteinmpnn.ensemble.gmm.GaussianMixtureModelJax')
@patch('prxteinmpnn.ensemble.gmm.EMFitter')
def test_make_fit_gmm_multiple_calls(mock_em_fitter, mock_gmm, mock_logits):
  """Test that the GMM fitting function can be called multiple times.
  
  Args:
    mock_em_fitter: Mock EMFitter class.
    mock_gmm: Mock GaussianMixtureModelJax class.
    mock_logits: Mock logits fixture.
    
  Raises:
    AssertionError: If multiple calls don't work correctly.
    
  Example:
    >>> test_make_fit_gmm_multiple_calls(...)
  """
  # Setup mocks
  mock_gmm_instance = Mock()
  mock_gmm.create.return_value = mock_gmm_instance
  
  mock_fitter_instance = Mock()
  mock_fitted_gmm1 = Mock()
  mock_fitted_gmm2 = Mock()
  
  # Set up different return values for multiple calls
  mock_fitter_instance.fit.side_effect = [mock_fitted_gmm1, mock_fitted_gmm2]
  mock_em_fitter.return_value = mock_fitter_instance
  
  gmm_fit_fn = make_fit_gmm()
  
  # Call the function twice
  result1 = gmm_fit_fn(mock_logits)
  result2 = gmm_fit_fn(mock_logits)
  
  # Verify both calls worked and returned different results
  assert result1 == mock_fitted_gmm1
  assert result2 == mock_fitted_gmm2
  assert mock_fitter_instance.fit.call_count == 2


@patch('prxteinmpnn.ensemble.gmm.GaussianMixtureModelJax')
@patch('prxteinmpnn.ensemble.gmm.EMFitter')
def test_make_fit_gmm_em_fitter_configuration(mock_em_fitter, mock_gmm):
  """Test that EMFitter is configured with correct parameters.
  
  Args:
    mock_em_fitter: Mock EMFitter class.
    mock_gmm: Mock GaussianMixtureModelJax class.
    
  Raises:
    AssertionError: If EMFitter configuration is incorrect.
    
  Example:
    >>> test_make_fit_gmm_em_fitter_configuration(...)
  """
  # Setup mocks
  mock_gmm.create.return_value = Mock()
  mock_em_fitter.return_value = Mock()
  
  # Create GMM fitting function
  make_fit_gmm()
  
  # Verify EMFitter was called with expected tolerance and max iterations
  mock_em_fitter.assert_called_once_with(tol=1e-3, max_iter=100, reg_covar=1e-6)


def test_make_fit_gmm_edge_cases():
  """Test GMM fitting with edge case parameters.
  
  Raises:
    AssertionError: If edge cases are not handled correctly.
    
  Example:
    >>> test_make_fit_gmm_edge_cases()
  """
  with patch('prxteinmpnn.ensemble.gmm.GaussianMixtureModelJax') as mock_gmm, \
       patch('prxteinmpnn.ensemble.gmm.EMFitter') as mock_em_fitter:
    
    mock_gmm.create.return_value = Mock()
    mock_em_fitter.return_value = Mock()
    
    # Test with minimum components
    gmm_fit_fn = make_fit_gmm(n_components=1, n_features=1)
    mock_gmm.create.assert_called_with(n_components=1, n_features=1)
    
    # Test with large number of components
    gmm_fit_fn = make_fit_gmm(n_components=1000, n_features=100)
    mock_gmm.create.assert_called_with(n_components=1000, n_features=100)
