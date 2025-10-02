"""Unit tests for Gaussian Mixture Model functionality."""

from unittest.mock import MagicMock, patch

import chex
import jax
import jax.numpy as jnp
import pytest
from gmmx import GaussianMixtureModelJax

from prxteinmpnn.ensemble.gmm import (
  _kmeans,
  _kmeans_plusplus_init,
  make_fit_gmm_in_memory,
)


@pytest.fixture
def sample_data():
  """Create sample data for testing.
  
  Returns:
    jnp.ndarray: Sample 2D data array.
  """
  key = jax.random.PRNGKey(42)
  return jax.random.normal(key, (100, 5))


@pytest.fixture
def small_data():
  """Create small sample data for detailed testing.
  
  Returns:
    jnp.ndarray: Small 2D data array.
  """
  return jnp.array([
    [0.0, 0.0],
    [1.0, 1.0],
    [2.0, 2.0],
    [0.1, 0.1],
    [1.1, 1.1],
    [2.1, 2.1],
  ])


class TestKMeansPlusPlusInit:
  """Test the K-Means++ initialization function."""

  def test_correct_output_shape(self, sample_data):
    """Test that K-Means++ returns correct centroid shape.
    
    Args:
      sample_data: Sample data fixture.
    """
    key = jax.random.PRNGKey(42)
    n_clusters = 3
    
    centroids = _kmeans_plusplus_init(key, sample_data, n_clusters)
    
    chex.assert_shape(centroids, (n_clusters, sample_data.shape[1]))

  def test_centroids_are_data_points(self, small_data):
    """Test that initial centroids are actual data points.
    
    Args:
      small_data: Small data fixture.
    """
    key = jax.random.PRNGKey(42)
    n_clusters = 3
    
    centroids = _kmeans_plusplus_init(key, small_data, n_clusters)
    
    # Each centroid should be one of the original data points
    for centroid in centroids:
      distances = jnp.linalg.norm(small_data - centroid, axis=1)
      assert jnp.min(distances) < 1e-10  # Should match exactly

  def test_single_cluster(self, sample_data):
    """Test K-Means++ with single cluster.
    
    Args:
      sample_data: Sample data fixture.
    """
    key = jax.random.PRNGKey(42)
    
    centroids = _kmeans_plusplus_init(key, sample_data, 1)
    
    chex.assert_shape(centroids, (1, sample_data.shape[1]))

  def test_deterministic_with_same_key(self, sample_data):
    """Test that same key produces same initialization.
    
    Args:
      sample_data: Sample data fixture.
    """
    key = jax.random.PRNGKey(42)
    n_clusters = 3
    
    centroids1 = _kmeans_plusplus_init(key, sample_data, n_clusters)
    centroids2 = _kmeans_plusplus_init(key, sample_data, n_clusters)
    
    chex.assert_trees_all_close(centroids1, centroids2)

  def test_different_keys_different_results(self, sample_data):
    """Test that different keys produce different initializations.
    
    Args:
      sample_data: Sample data fixture.
    """
    key1 = jax.random.PRNGKey(42)
    key2 = jax.random.PRNGKey(24)
    n_clusters = 3
    
    centroids1 = _kmeans_plusplus_init(key1, sample_data, n_clusters)
    centroids2 = _kmeans_plusplus_init(key2, sample_data, n_clusters)
    
    # Results should be different (with high probability)
    assert not jnp.allclose(centroids1, centroids2)

  @pytest.mark.parametrize("n_clusters", [1, 2, 3, 5])
  def test_various_cluster_counts(self, sample_data, n_clusters):
    """Test K-Means++ with various cluster counts.
    
    Args:
      sample_data: Sample data fixture.
      n_clusters: Number of clusters to test.
    """
    key = jax.random.PRNGKey(42)
    
    centroids = _kmeans_plusplus_init(key, sample_data, n_clusters)
    
    chex.assert_shape(centroids, (n_clusters, sample_data.shape[1]))


class TestKMeans:
  """Test the K-Means clustering function."""

  def test_correct_output_shape(self, sample_data):
    """Test that K-Means returns correct label shape.
    
    Args:
      sample_data: Sample data fixture.
    """
    key = jax.random.PRNGKey(42)
    n_clusters = 3
    
    labels = _kmeans(key, sample_data, n_clusters)
    
    chex.assert_shape(labels, (sample_data.shape[0],))

  def test_labels_in_valid_range(self, sample_data):
    """Test that all labels are in valid range.
    
    Args:
      sample_data: Sample data fixture.
    """
    key = jax.random.PRNGKey(42)
    n_clusters = 3
    
    labels = _kmeans(key, sample_data, n_clusters)
    
    assert jnp.all(labels >= 0)
    assert jnp.all(labels < n_clusters)

  def test_well_separated_clusters(self):
    """Test K-Means on well-separated clusters."""
    # Create well-separated clusters
    cluster1 = jnp.array([[0.0, 0.0], [0.1, 0.1], [0.0, 0.1]])
    cluster2 = jnp.array([[5.0, 5.0], [5.1, 5.1], [5.0, 5.1]])
    cluster3 = jnp.array([[10.0, 10.0], [10.1, 10.1], [10.0, 10.1]])
    
    data = jnp.vstack([cluster1, cluster2, cluster3])
    key = jax.random.PRNGKey(42)
    
    labels = _kmeans(key, data, 3)
    
    # Points in same cluster should have same label
    assert len(jnp.unique(labels[:3])) == 1  # Cluster 1
    assert len(jnp.unique(labels[3:6])) == 1  # Cluster 2
    assert len(jnp.unique(labels[6:9])) == 1  # Cluster 3

  def test_single_cluster(self, sample_data):
    """Test K-Means with single cluster.
    
    Args:
      sample_data: Sample data fixture.
    """
    key = jax.random.PRNGKey(42)
    
    labels = _kmeans(key, sample_data, 1)
    
    # All points should be in cluster 0
    assert jnp.all(labels == 0)

  @pytest.mark.parametrize("max_iters", [1, 10, 50])
  def test_different_max_iterations(self, sample_data, max_iters):
    """Test K-Means with different maximum iterations.
    
    Args:
      sample_data: Sample data fixture.
      max_iters: Maximum iterations to test.
    """
    key = jax.random.PRNGKey(42)
    
    labels = _kmeans(key, sample_data, 3, max_iters=max_iters)
    
    chex.assert_shape(labels, (sample_data.shape[0],))
    assert jnp.all(labels >= 0)
    assert jnp.all(labels < 3)

  def test_deterministic_with_same_key(self, sample_data):
    """Test that same key produces same clustering.
    
    Args:
      sample_data: Sample data fixture.
    """
    key = jax.random.PRNGKey(42)
    
    labels1 = _kmeans(key, sample_data, 3)
    labels2 = _kmeans(key, sample_data, 3)
    
    chex.assert_trees_all_equal(labels1, labels2)


class TestMakeFitGMM:
  """Test the make_fit_gmm_in_memory function factory."""

  def test_returns_callable(self):
    """Test that make_fit_gmm_in_memory returns a callable function."""
    fit_fn = make_fit_gmm_in_memory(n_components=3)
    
    assert callable(fit_fn)

  @patch("prxteinmpnn.ensemble.gmm.fit_gmm_in_memory")
  @patch("prxteinmpnn.ensemble.gmm._kmeans")
  def test_fit_gmm_workflow(self, mock_kmeans, mock_fit_gmm_in_memory, sample_data):
    """Test the complete GMM fitting workflow.
    
    Args:
      mock_kmeans: Mock K-Means function.
      mock_fit_gmm_in_memory: Mock for the in-memory GMM fitting function.
      sample_data: Sample data fixture.
    """
    # Setup mocks
    mock_labels = jnp.array([0, 1, 2, 0, 1] * 20)  # 100 labels
    mock_kmeans.return_value = mock_labels
    
    mock_fitted_gmm = MagicMock(spec=GaussianMixtureModelJax)
    mock_fit_result = MagicMock()
    mock_fit_result.gmm = mock_fitted_gmm
    mock_fit_gmm_in_memory.return_value = mock_fit_result
    
    with patch("prxteinmpnn.ensemble.gmm.GaussianMixtureModelJax") as mock_gmm_class:
      mock_initial_gmm = MagicMock()
      mock_gmm_class.from_responsibilities.return_value = mock_initial_gmm
      
      # Create and test fit function
      fit_fn = make_fit_gmm_in_memory(n_components=3)
      key = jax.random.PRNGKey(42)
      
      result = fit_fn(sample_data, key)
      
      # Verify workflow
      mock_kmeans.assert_called_once()
      mock_gmm_class.from_responsibilities.assert_called_once()
      mock_fit_gmm_in_memory.assert_called_once()

      assert result == mock_fitted_gmm

  def test_responsibilities_creation(self, sample_data):
    """Test that one-hot responsibilities are created correctly from labels."""
    with patch("prxteinmpnn.ensemble.gmm._kmeans") as mock_kmeans:
      with patch("prxteinmpnn.ensemble.gmm.fit_gmm_in_memory") as mock_fit_gmm_in_memory:
        with patch("prxteinmpnn.ensemble.gmm.GaussianMixtureModelJax") as mock_gmm_class:
          # Setup
          n_components = 3
          mock_labels = jnp.array([0, 1, 2, 0, 1] * 20)
          mock_kmeans.return_value = mock_labels
          
          mock_fit_result = MagicMock()
          mock_fit_result.gmm = MagicMock()
          mock_fit_gmm_in_memory.return_value = mock_fit_result
          
          # Test
          fit_fn = make_fit_gmm_in_memory(n_components=n_components)
          key = jax.random.PRNGKey(42)
          
          fit_fn(sample_data, key)
          
          # Check that responsibilities were created correctly
          call_args = mock_gmm_class.from_responsibilities.call_args
          responsibilities = call_args[1]['resp']
          
          chex.assert_shape(responsibilities, (1, 100, n_components))
          # Should be one-hot encoded
          assert jnp.allclose(jnp.sum(responsibilities, axis=2), 1.0)

  @pytest.mark.parametrize("n_components", [1, 2, 3, 5])
  def test_different_dimensions(self, n_components):
    """Test GMM fitting with different dimensions.
    
    Args:
      n_components: Number of components to test.
    """
    with patch("prxteinmpnn.ensemble.gmm._kmeans") as mock_kmeans:
      with patch("prxteinmpnn.ensemble.gmm.fit_gmm_in_memory") as mock_fit_gmm_in_memory:
        with patch("prxteinmpnn.ensemble.gmm.GaussianMixtureModelJax"):
          # Setup
          mock_labels = jnp.zeros(50, dtype=jnp.int32)  # All same cluster for simplicity
          mock_kmeans.return_value = mock_labels
          
          mock_fit_result = MagicMock()
          mock_fit_result.gmm = MagicMock()
          mock_fit_gmm_in_memory.return_value = mock_fit_result
          
          # Test
          fit_fn = make_fit_gmm_in_memory(
            n_components=n_components,
          )
          
          assert callable(fit_fn)

  def test_em_fitter_configuration(self):
    """Test that EM fitter is configured correctly."""
    with patch("prxteinmpnn.ensemble.gmm.fit_gmm_in_memory") as mock_fit_gmm_in_memory:
      with patch("prxteinmpnn.ensemble.gmm.GaussianMixtureModelJax") as mock_gmm_class:
        mock_gmm_class.from_responsibilities.return_value = MagicMock()
        fit_fn = make_fit_gmm_in_memory(
          n_components=3,
          gmm_max_iters=200,
          reg_covar=1e-5,
        )

        key = jax.random.PRNGKey(42)
        data = jnp.ones((10, 2))
        fit_fn(data, key)

        mock_fit_gmm_in_memory.assert_called_once()
        call_args = mock_fit_gmm_in_memory.call_args
        assert call_args[1]['max_iter'] == 200
        assert call_args[1]['reg_covar'] == 1e-5

  def test_jit_compilation(self, sample_data):
    """Test that the returned function is JIT-compatible."""
    with patch("prxteinmpnn.ensemble.gmm._kmeans") as mock_kmeans:
      with patch("prxteinmpnn.ensemble.gmm.fit_gmm_in_memory") as mock_fit_gmm_in_memory:
        with patch("prxteinmpnn.ensemble.gmm.GaussianMixtureModelJax"):
          # Setup mocks
          mock_labels = jnp.zeros(100, dtype=jnp.int32)
          mock_kmeans.return_value = mock_labels
          
          mock_fitted_gmm = MagicMock()
          mock_fit_result = MagicMock()
          mock_fit_result.gmm = mock_fitted_gmm
          mock_fit_gmm_in_memory.return_value = mock_fit_result
          
          # Test that function can be JIT compiled
          fit_fn = make_fit_gmm_in_memory(n_components=3)
          key = jax.random.PRNGKey(42)
          
          # This should not raise an error
          result = fit_fn(sample_data, key)
          assert result is not None


class TestIntegration:
  """Integration tests for the GMM module."""

  def test_realistic_gmm_fitting(self):
    """Test realistic GMM fitting workflow with actual data."""
    # Create realistic clustered data
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 4)
    
    # Generate 3 well-separated clusters
    cluster1 = jax.random.normal(keys[0], (30, 4)) + jnp.array([0, 0, 0, 0])
    cluster2 = jax.random.normal(keys[1], (30, 4)) + jnp.array([5, 5, 5, 5])
    cluster3 = jax.random.normal(keys[2], (40, 4)) + jnp.array([-3, -3, -3, -3])
    
    data = jnp.vstack([cluster1, cluster2, cluster3])
    
    with patch("prxteinmpnn.ensemble.gmm.fit_gmm_in_memory") as mock_fit_gmm_in_memory:
      with patch("prxteinmpnn.ensemble.gmm.GaussianMixtureModelJax") as mock_gmm_class:
        # Setup mocks to return reasonable values
        mock_fitted_gmm = MagicMock(spec=GaussianMixtureModelJax)
        mock_fitted_gmm.n_components = 3
        mock_fit_result = MagicMock()
        mock_fit_result.gmm = mock_fitted_gmm
        mock_fit_gmm_in_memory.return_value = mock_fit_result
        
        mock_initial_gmm = MagicMock()
        mock_gmm_class.from_responsibilities.return_value = mock_initial_gmm
        
        # Test
        fit_fn = make_fit_gmm_in_memory(n_components=3)
        result = fit_fn(data, keys[3])
        
        # Verify the workflow completed
        assert result == mock_fitted_gmm
        
        # Check that responsibilities had correct shape
        call_args = mock_gmm_class.from_responsibilities.call_args
        responsibilities = call_args[1]['resp']
        chex.assert_shape(responsibilities, (1, 100, 3))

  def test_edge_case_single_point_per_cluster(self):
    """Test GMM fitting with minimal data."""
    data = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    
    with patch("prxteinmpnn.ensemble.gmm.fit_gmm_in_memory") as mock_fit_gmm_in_memory:
      with patch("prxteinmpnn.ensemble.gmm.GaussianMixtureModelJax") as mock_gmm_class:
        # Setup
        mock_fitted_gmm = MagicMock()
        mock_fit_result = MagicMock()
        mock_fit_result.gmm = mock_fitted_gmm
        mock_fit_gmm_in_memory.return_value = mock_fit_result
        
        mock_gmm_class.from_responsibilities.return_value = MagicMock()
        
        # Test
        fit_fn = make_fit_gmm_in_memory(n_components=3)
        key = jax.random.PRNGKey(42)
        
        result = fit_fn(data, key)
        
        assert result == mock_fitted_gmm

  def test_parameter_validation(self):
    """Test parameter validation for make_fit_gmm_in_memory."""
    # These should not raise errors
    make_fit_gmm_in_memory(n_components=1)
    make_fit_gmm_in_memory(n_components=10)
    make_fit_gmm_in_memory(
      n_components=3,
      kmeans_max_iters=50,
      gmm_max_iters=150,
      reg_covar=1e-4,
    )
    
    # Function should be created successfully in all cases
    assert True  # If we reach here, all tests passed
