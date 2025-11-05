"""Tests for DBSCAN clustering functionality."""

import chex
import jax
import jax.numpy as jnp
import pytest
from unittest.mock import Mock, patch

from prxteinmpnn.ensemble.dbscan import (
  ConformationalStates,
  GMMClusteringResult,
  EntropyTrace,
  compute_component_distances,
  dbscan_cluster,
  trace_entropy_across_eps,
  _get_neighborhood,
)


class TestResidueConformationalStates:
  """Test the ResidueConformationalStates dataclass."""
  
  def test_residue_conformational_states_creation(self):
    """Test creation of ResidueConformationalStates.
    
    Raises:
      AssertionError: If dataclass creation fails.
      
    Example:
      >>> test_residue_conformational_states_creation()
    """
    states = ConformationalStates(
      n_states=5,
      mle_entropy=1.2,
      mle_entropy_se=0.1,
      state_trajectory=jnp.array([0, 1, 0, 2, 1]),
      state_counts=jnp.array([2, 2, 1]),
      cluster_entropy=1.1,
      cluster_probabilities=jnp.array([0.4, 0.4, 0.2]),
      dbscan_eps=0.5,
      min_cluster_weight=0.01,
    )
    
    assert states.n_states == 5
    assert states.mle_entropy == 1.2
    chex.assert_shape(states.state_trajectory, (5,))
    chex.assert_shape(states.state_counts, (3,))


class TestGMMClusteringResult:
  """Test the GMMClusteringResult dataclass."""
  
  def test_gmm_clustering_result_creation(self):
    """Test creation of GMMClusteringResult.
    
    Raises:
      AssertionError: If dataclass creation fails.
      
    Example:
      >>> test_gmm_clustering_result_creation()
    """
    n_components = 10
    result = GMMClusteringResult(
      coarse_graining_matrix=jnp.eye(n_components),
      core_component_connectivity=jnp.zeros((n_components, n_components)),
      non_noise_connectivity=jnp.zeros((n_components, n_components)),
      state_probabilities=jnp.ones(n_components) / n_components,
      plug_in_entropy=2.3,
      von_neumann_entropy=2.2,
      posterior_mean_entropy=2.25,
      posterior_entropy_std_err=0.05,
      dbscan_eps=0.3,
      min_cluster_weight=0.01,
      state_density_matrix=jnp.eye(n_components) / n_components,
    )
    
    assert result.plug_in_entropy == 2.3
    chex.assert_shape(result.coarse_graining_matrix, (n_components, n_components))
    chex.assert_shape(result.state_probabilities, (n_components,))


class TestEntropyTrace:
  """Test the EntropyTrace dataclass."""
  
  def test_entropy_trace_creation(self):
    """Test creation of EntropyTrace.
    
    Raises:
      AssertionError: If dataclass creation fails.
      
    Example:
      >>> test_entropy_trace_creation()
    """
    n_eps = 50
    trace = EntropyTrace(
      plug_in_entropy=jnp.ones(n_eps) * 2.0,
      von_neumann_entropy=jnp.ones(n_eps) * 1.9,
      posterior_mean_entropy=jnp.ones(n_eps) * 1.95,
      posterior_entropy_std_err=jnp.ones(n_eps) * 0.1,
      z_score_sq=jnp.ones(n_eps) * 0.5,
      eps_values=jnp.linspace(0.01, 0.99, n_eps),
    )
    
    chex.assert_shape(trace.plug_in_entropy, (n_eps,))
    chex.assert_shape(trace.eps_values, (n_eps,))
    assert jnp.allclose(trace.plug_in_entropy, 2.0)


class TestComputeComponentDistances:
  """Test component distance computation."""
  
  def test_compute_component_distances_shape(self):
    """Test output shape of distance computation.
    
    Raises:
      AssertionError: If output shape is incorrect.
      
    Example:
      >>> test_compute_component_distances_shape()
    """
    n_components = 5
    n_features = 21
    key = jax.random.PRNGKey(42)
    means = jax.random.normal(key, (n_components, n_features))
    
    distances = compute_component_distances(means)
    
    chex.assert_shape(distances, (n_components, n_components))
    chex.assert_type(distances, jnp.floating)
  
  def test_compute_component_distances_symmetry(self):
    """Test that distance matrix is symmetric.
    
    Raises:
      AssertionError: If distance matrix is not symmetric.
      
    Example:
      >>> test_compute_component_distances_symmetry()
    """
    n_components = 4
    n_features = 21
    key = jax.random.PRNGKey(123)
    means = jax.random.normal(key, (n_components, n_features))
    
    distances = compute_component_distances(means)
    
    assert jnp.allclose(distances, distances.T), "Distance matrix should be symmetric"
  
  def test_compute_component_distances_diagonal_zeros(self):
    """Test that diagonal elements are zero.
    
    Raises:
      AssertionError: If diagonal elements are not zero.
      
    Example:
      >>> test_compute_component_distances_diagonal_zeros()
    """
    n_components = 3
    n_features = 10
    key = jax.random.PRNGKey(456)
    means = jax.random.normal(key, (n_components, n_features))
    
    distances = compute_component_distances(means)
    
    diagonal = jnp.diag(distances)
    assert jnp.allclose(diagonal, 0.0, atol=1e-6), "Diagonal should be zero"
  
  def test_compute_component_distances_positive(self):
    """Test that all distances are non-negative.
    
    Raises:
      AssertionError: If any distance is negative.
      
    Example:
      >>> test_compute_component_distances_positive()
    """
    n_components = 6
    n_features = 21
    key = jax.random.PRNGKey(789)
    means = jax.random.normal(key, (n_components, n_features))
    
    distances = compute_component_distances(means)
    
    assert jnp.all(distances >= 0), "All distances should be non-negative"


class TestGetNeighborhood:
  """Test neighborhood computation."""
  
  def test_get_neighborhood_shape(self):
    """Test output shape of neighborhood computation.
    
    Raises:
      AssertionError: If output shape is incorrect.
      
    Example:
      >>> test_get_neighborhood_shape()
    """
    n_components = 5
    distance_matrix = jax.random.uniform(jax.random.PRNGKey(0), (n_components, n_components))
    eps = 0.5
    
    neighborhood = _get_neighborhood(distance_matrix, eps)
    
    chex.assert_shape(neighborhood, (n_components, n_components))
    chex.assert_type(neighborhood, jnp.floating)
  
  def test_get_neighborhood_binary(self):
    """Test that neighborhood matrix is binary.
    
    Raises:
      AssertionError: If neighborhood matrix is not binary.
      
    Example:
      >>> test_get_neighborhood_binary()
    """
    n_components = 4
    distance_matrix = jnp.array([[0, 0.3, 0.7, 0.2],
                                 [0.3, 0, 0.4, 0.8],
                                 [0.7, 0.4, 0, 0.6],
                                 [0.2, 0.8, 0.6, 0]])
    eps = 0.5
    
    neighborhood = _get_neighborhood(distance_matrix, eps)
    
    # Should be binary (0 or 1)
    assert jnp.all(jnp.isin(neighborhood, jnp.array([0.0, 1.0]))), "Neighborhood should be binary"
  
  def test_get_neighborhood_threshold(self):
    """Test neighborhood threshold behavior.
    
    Raises:
      AssertionError: If threshold behavior is incorrect.
      
    Example:
      >>> test_get_neighborhood_threshold()
    """
    distance_matrix = jnp.array([[0, 0.3, 0.7],
                                 [0.3, 0, 0.4],
                                 [0.7, 0.4, 0]])
    eps = 0.5
    
    neighborhood = _get_neighborhood(distance_matrix, eps)
    
    # Distances <= eps should be 1, distances > eps should be 0
    expected = jnp.array([[1, 1, 0],  # 0 <= 0.5, 0.3 <= 0.5, 0.7 > 0.5
                          [1, 1, 1],  # 0.3 <= 0.5, 0 <= 0.5, 0.4 <= 0.5
                          [0, 1, 1]]) # 0.7 > 0.5, 0.4 <= 0.5, 0 <= 0.5
    
    assert jnp.allclose(neighborhood, expected), f"Expected {expected}, got {neighborhood}"


class TestPerformDbscanClustering:
  """Test DBSCAN clustering functionality."""
  
  @pytest.fixture
  def mock_clustering_data(self):
    """Create mock data for clustering tests.
    
    Returns:
      Dictionary containing mock clustering data.
    """
    n_components = 6
    n_observations = 20
    key = jax.random.PRNGKey(42)
    
    # Create simple distance matrix
    distance_matrix = jax.random.uniform(key, (n_components, n_components))
    distance_matrix = (distance_matrix + distance_matrix.T) / 2  # Make symmetric
    distance_matrix = distance_matrix.at[jnp.diag_indices(n_components)].set(0)  # Zero diagonal
    
    component_weights = jnp.ones(n_components) / n_components
    responsibility_matrix = jax.random.uniform(
      jax.random.split(key)[1], (n_observations, n_components)
    )
    # Normalize responsibilities
    responsibility_matrix = responsibility_matrix / responsibility_matrix.sum(axis=1, keepdims=True)
    
    return {
      'distance_matrix': distance_matrix,
      'component_weights': component_weights,
      'responsibility_matrix': responsibility_matrix,
      'eps': 0.3,
      'min_cluster_weight': 0.05,
    }
  
  def test_perform_dbscan_clustering_output_types(self, mock_clustering_data):
      """Test that DBSCAN clustering returns correct types.
      
      Args:
        mock_clustering_data: Mock clustering data fixture.
        
      Raises:
        AssertionError: If output types are incorrect.
        
      Example:
        >>> test_perform_dbscan_clustering_output_types(mock_clustering_data)
      """
      result = dbscan_cluster(**mock_clustering_data)
      
      assert isinstance(result, GMMClusteringResult)
      chex.assert_type(result.coarse_graining_matrix, jnp.floating)
      chex.assert_type(result.state_probabilities, jnp.floating)
      chex.assert_type(result.plug_in_entropy, jnp.floating)
  
  def test_perform_dbscan_clustering_output_shapes(self, mock_clustering_data):
    """Test that DBSCAN clustering returns correct shapes.
    
    Args:
      mock_clustering_data: Mock clustering data fixture.
      
    Raises:
      AssertionError: If output shapes are incorrect.
      
    Example:
      >>> test_perform_dbscan_clustering_output_shapes(mock_clustering_data)
    """
    data = mock_clustering_data
    n_components = data['distance_matrix'].shape[0]
    
    result = dbscan_cluster(**data)
    
    chex.assert_shape(result.coarse_graining_matrix, (n_components, n_components))
    chex.assert_shape(result.core_component_connectivity, (n_components, n_components))
    chex.assert_shape(result.state_probabilities, (n_components,))
    chex.assert_shape(result.state_density_matrix, (n_components, n_components))
  
  def test_perform_dbscan_clustering_probability_sum(self, mock_clustering_data):
    """Test that state probabilities sum to approximately 1.
    
    Args:
      mock_clustering_data: Mock clustering data fixture.
      
    Raises:
      AssertionError: If probabilities don't sum to 1.
      
    Example:
      >>> test_perform_dbscan_clustering_probability_sum(mock_clustering_data)
    """
    result = dbscan_cluster(**mock_clustering_data)
    
    prob_sum = jnp.sum(result.state_probabilities)
    assert jnp.allclose(prob_sum, 1.0, atol=1e-5), f"Probabilities sum to {prob_sum}, not 1.0"
  
  def test_perform_dbscan_clustering_connectivity_methods(self, mock_clustering_data):
    """Test different connectivity methods.
    
    Args:
      mock_clustering_data: Mock clustering data fixture.
      
    Raises:
      AssertionError: If connectivity methods don't work correctly.
      
    Example:
      >>> test_perform_dbscan_clustering_connectivity_methods(mock_clustering_data)
    """
    data = mock_clustering_data
    
    # Test 'expm' method (default)
    result_expm = dbscan_cluster(**data, connectivity_method='expm')
    assert isinstance(result_expm, GMMClusteringResult)
    
    # Test 'power' method
    result_power = dbscan_cluster(**data, connectivity_method='power')
    assert isinstance(result_power, GMMClusteringResult)
    
    # Results may be different but should have same structure
    chex.assert_shape(
      result_expm.coarse_graining_matrix,
      result_power.coarse_graining_matrix.shape
    )
  

class TestTraceEntropyAcrossEps:
  """Test entropy tracing functionality."""
  
  @pytest.fixture
  def mock_gmm(self):
    """Create mock GMM for testing.
    
    Returns:
      Mock GMM object.
    """
    gmm = Mock()
    n_components = 10
    n_features = 21
    key = jax.random.PRNGKey(42)
    
    gmm.means_ = jax.random.normal(key, (n_components, n_features))
    gmm.weights_ = jnp.ones(n_components) / n_components
    gmm.predict_proba.return_value = jax.random.uniform(
      jax.random.split(key)[1], (50, n_components)
    )
    
    return gmm
  
  @pytest.fixture
  def mock_logits_for_trace(self):
    """Create mock logits for entropy tracing.
    
    Returns:
      Mock logits array.
    """
    n_timesteps = 50
    n_features = 21
    key = jax.random.PRNGKey(123)
    return jax.random.normal(key, (n_timesteps, n_features))

  @patch('prxteinmpnn.ensemble.dbscan.dbscan_cluster')
  def test_trace_entropy_across_eps_default_eps(
    self,
    mock_clustering,
    mock_gmm,
    mock_logits_for_trace,
  ):
    """Test entropy tracing with default eps values.
    
    Args:
      mock_clustering: Mock DBSCAN clustering function.
      mock_gmm: Mock GMM fixture.
      mock_logits_for_trace: Mock logits fixture.
      
    Raises:
      AssertionError: If entropy tracing doesn't work correctly.
      
    Example:
      >>> test_trace_entropy_across_eps_default_eps(...)
    """
    # Setup mock clustering result
    mock_result = Mock()
    mock_result.plug_in_entropy = 2.0
    mock_result.von_neumann_entropy = 1.9
    mock_result.posterior_mean_entropy = 1.95
    mock_result.posterior_entropy_std_err = 0.1
    mock_clustering.return_value = mock_result
    
    # Test tracing
    trace = trace_entropy_across_eps(mock_gmm, mock_logits_for_trace)
    
    assert isinstance(trace, EntropyTrace)
    chex.assert_shape(trace.eps_values, (99,))  # Default linspace(0.01, 0.99, 99)
    chex.assert_shape(trace.plug_in_entropy, (99,))
    chex.assert_shape(trace.von_neumann_entropy, (99,))
  
  @patch('prxteinmpnn.ensemble.dbscan.dbscan_cluster')
  def test_trace_entropy_across_eps_custom_eps(
    self,
    mock_clustering,
    mock_gmm,
    mock_logits_for_trace,
  ):
    """Test entropy tracing with custom eps values.
    
    Args:
      mock_clustering: Mock DBSCAN clustering function.
      mock_gmm: Mock GMM fixture.
      mock_logits_for_trace: Mock logits fixture.
      
    Raises:
      AssertionError: If custom eps values don't work correctly.
      
    Example:
      >>> test_trace_entropy_across_eps_custom_eps(...)
    """
    # Setup mock clustering result
    mock_result = Mock()
    mock_result.plug_in_entropy = 1.5
    mock_result.von_neumann_entropy = 1.4
    mock_result.posterior_mean_entropy = 1.45
    mock_result.posterior_entropy_std_err = 0.05
    mock_clustering.return_value = mock_result
    
    # Custom eps values
    custom_eps = jnp.array([0.1, 0.3, 0.5, 0.7])
    
    trace = trace_entropy_across_eps(
      mock_gmm,
      mock_logits_for_trace,
      eps_values=custom_eps
    )
    
    chex.assert_shape(trace.eps_values, (4,))
    chex.assert_shape(trace.plug_in_entropy, (4,))
    assert jnp.allclose(trace.eps_values, custom_eps)
  
  @patch('prxteinmpnn.ensemble.dbscan.dbscan_cluster')
  @patch('jax.lax.map')
  def test_trace_entropy_across_eps_chunked(
    self,
    mock_lax_map,
    mock_clustering,
    mock_gmm,
    mock_logits_for_trace,
  ):
    """Test entropy tracing with chunked processing.
    
    Args:
      mock_lax_map: Mock jax.lax.map function.
      mock_clustering: Mock DBSCAN clustering function.
      mock_gmm: Mock GMM fixture.
      mock_logits_for_trace: Mock logits fixture.
      
    Raises:
      AssertionError: If chunked processing doesn't work correctly.
      
    Example:
      >>> test_trace_entropy_across_eps_chunked(...)
    """
    # Setup mock returns
    mock_results = (
      jnp.array([2.0, 1.8, 1.6]),
      jnp.array([1.9, 1.7, 1.5]),
      jnp.array([1.95, 1.75, 1.55]),
      jnp.array([0.1, 0.1, 0.1]),
      jnp.array([0.5, 0.5, 0.5]),
    )
    mock_lax_map.return_value = mock_results
    
    large_eps = jnp.linspace(0.01, 0.99, 150)  # Large enough to trigger chunking
    
    trace = trace_entropy_across_eps(
      mock_gmm,
      mock_logits_for_trace,
      eps_values=large_eps,
      vmap_chunk_size=50,
    )
    
    # Should use lax.map for chunked processing
    mock_lax_map.assert_called_once()
    chex.assert_shape(trace.plug_in_entropy, (3,))  # Flattened mock results
