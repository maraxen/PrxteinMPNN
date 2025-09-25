"""Unit tests for conformational inference functionality."""

from unittest.mock import MagicMock, patch

import chex
import jax.numpy as jnp
import jax
import pytest
from gmmx import GaussianMixtureModelJax

from prxteinmpnn.ensemble.ci import infer_states
from prxteinmpnn.ensemble.dbscan import ConformationalStates


@pytest.fixture
def mock_gmm():
  """Create a mock Gaussian Mixture Model for testing.
  
  Returns:
    MagicMock: A mock GMM with realistic attributes.
  """
  gmm = MagicMock(spec=GaussianMixtureModelJax)
  gmm.n_components = 3
  gmm.means = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
  gmm.weights = jnp.array([0.4, 0.3, 0.3])
  
  # Mock predict_proba to return realistic responsibility matrix
  responsibility_matrix = jnp.array([
    [0.8, 0.1, 0.1],
    [0.2, 0.7, 0.1],
    [0.1, 0.2, 0.7],
    [0.9, 0.05, 0.05],
    [0.1, 0.1, 0.8],
  ])
  gmm.predict_proba.return_value = responsibility_matrix[:, :, None, None]
  
  return gmm


@pytest.fixture
def sample_features():
  """Create sample features for testing.
  
  Returns:
    jnp.ndarray: Sample feature array.
  """
  return jnp.array([
    [0.1, 0.2],
    [1.1, 1.2],
    [2.1, 2.2],
    [0.05, 0.15],
    [2.05, 2.15],
  ])


class TestInferStates:
  """Test the infer_states function."""

  @patch("prxteinmpnn.ensemble.ci.compute_component_distances")
  @patch("prxteinmpnn.ensemble.ci.dbscan_cluster")
  def test_successful_inference(
    self,
    mock_dbscan_cluster,
    mock_compute_distances,
    mock_gmm,
    sample_features,
  ):
    """Test successful conformational state inference.
    
    Args:
      mock_dbscan_cluster: Mock DBSCAN clustering function.
      mock_compute_distances: Mock distance computation function.
      mock_gmm: Mock GMM fixture.
      sample_features: Sample features fixture.
    """
    # Setup mocks
    distance_matrix = jnp.array([
      [0.0, 1.4, 2.8],
      [1.4, 0.0, 1.4],
      [2.8, 1.4, 0.0],
    ])
    mock_compute_distances.return_value = distance_matrix
    
    # Mock cluster result
    mock_cluster_result = MagicMock()
    mock_cluster_result.coarse_graining_matrix = jnp.array([
      [1.0, 0.0],
      [0.0, 1.0],
      [0.0, 1.0],
    ])
    mock_cluster_result.plug_in_entropy = 0.5
    mock_cluster_result.state_probabilities = jnp.array([0.6, 0.4])
    mock_dbscan_cluster.return_value = mock_cluster_result
    
    # Test
    result = infer_states(
      gmm=mock_gmm,
      features=sample_features,
      eps_std_scale=1.0,
      min_cluster_weight=0.01,
    )
    
    # Assertions
    assert isinstance(result, ConformationalStates)
    mock_compute_distances.assert_called_once_with(mock_gmm.means)
    mock_dbscan_cluster.assert_called_once()
    
    # Check that GMM predict_proba was called
    mock_gmm.predict_proba.assert_called_once_with(sample_features)

  def test_eps_calculation(self, mock_gmm, sample_features):
    """Test epsilon calculation for DBSCAN.
    
    Args:
      mock_gmm: Mock GMM fixture.
      sample_features: Sample features fixture.
    """
    with patch("prxteinmpnn.ensemble.ci.compute_component_distances") as mock_distances:
      with patch("prxteinmpnn.ensemble.ci.dbscan_cluster") as mock_dbscan:
        # Setup distance matrix
        distance_matrix = jnp.array([
          [0.0, 1.0, 2.0],
          [1.0, 0.0, 1.0],
          [2.0, 1.0, 0.0],
        ])
        mock_distances.return_value = distance_matrix
        
        # Mock cluster result
        mock_cluster_result = MagicMock()
        mock_cluster_result.coarse_graining_matrix = jnp.eye(3)
        mock_cluster_result.plug_in_entropy = 0.0
        mock_cluster_result.state_probabilities = jnp.array([0.4, 0.3, 0.3])
        mock_dbscan.return_value = mock_cluster_result
        
        # Test with different eps_std_scale values
        eps_std_scale = 2.0
        infer_states(
          gmm=mock_gmm,
          features=sample_features,
          eps_std_scale=eps_std_scale,
        )
        
        # Check that eps was calculated correctly
        call_args = mock_dbscan.call_args
        eps_used = call_args[0][3]  # Fourth positional argument is eps
        
        # Expected calculation: 1.0 - 2.0 * std([1.0, 1.0, 2.0])
        triu_values = jnp.array([1.0, 2.0, 1.0])
        expected_eps = 1.0 - eps_std_scale * jnp.std(triu_values)
        
        chex.assert_trees_all_close(eps_used, expected_eps, rtol=1e-5)

  def test_state_trajectory_computation(self, mock_gmm, sample_features):
    """Test state trajectory computation from responsibilities.
    
    Args:
      mock_gmm: Mock GMM fixture.
      sample_features: Sample features fixture.
    """
    with patch("prxteinmpnn.ensemble.ci.compute_component_distances") as mock_distances:
      mock_distances.return_value = jnp.array([[0.0, 1.4, 2.8], [1.4, 0.0, 1.4], [2.8, 1.4, 0.0]])
      with patch("prxteinmpnn.ensemble.ci.dbscan_cluster") as mock_dbscan:
        # Setup cluster result that maps components to states
        coarse_graining_matrix = jnp.array([
          [1.0, 0.0],  # Component 0 -> State 0
          [0.0, 1.0],  # Component 1 -> State 1
          [0.0, 1.0],  # Component 2 -> State 1
        ])
        
        mock_cluster_result = MagicMock()
        mock_cluster_result.coarse_graining_matrix = coarse_graining_matrix
        mock_cluster_result.plug_in_entropy = 0.5
        mock_cluster_result.state_probabilities = jnp.array([0.6, 0.4])
        mock_dbscan.return_value = mock_cluster_result
        
        result = infer_states(gmm=mock_gmm, features=sample_features)
        
        # Check state trajectory shape
        chex.assert_shape(result.state_trajectory, (5,))
        
        # All trajectory values should be valid state indices
        assert jnp.all(result.state_trajectory >= 0)
        assert jnp.all(result.state_trajectory < 2)  # 2 states expected

  def test_entropy_calculations(self, mock_gmm, sample_features):
    """Test MLE entropy calculations.
    
    Args:
      mock_gmm: Mock GMM fixture.
      sample_features: Sample features fixture.
    """
    with patch("prxteinmpnn.ensemble.ci.compute_component_distances") as mock_distances:
      mock_distances.return_value = jnp.array([[0.0, 1.4, 2.8], [1.4, 0.0, 1.4], [2.8, 1.4, 0.0]])
      with patch("prxteinmpnn.ensemble.ci.dbscan_cluster") as mock_dbscan:
        with patch("prxteinmpnn.ensemble.ci.posterior_mean_std") as mock_posterior:
          # Setup mocks
          mock_cluster_result = MagicMock()
          mock_cluster_result.coarse_graining_matrix = jnp.eye(3)
          mock_cluster_result.plug_in_entropy = 0.8
          mock_cluster_result.state_probabilities = jnp.array([0.4, 0.3, 0.3])
          mock_dbscan.return_value = mock_cluster_result
          
          mock_posterior.return_value = (1.0, 0.1)  # (mean, std_error)
          
          result = infer_states(gmm=mock_gmm, features=sample_features)
          
          # Check that entropy values are reasonable
          assert result.mle_entropy >= 0.0
          assert result.mle_entropy_se >= 0.0
          assert result.cluster_entropy == 0.8

  @pytest.mark.parametrize("eps_std_scale", [0.5, 1.0, 1.5, 2.0])
  def test_different_eps_scales(self, mock_gmm, sample_features, eps_std_scale):
      """Test inference with different epsilon scaling factors."""
      # Note the "as mock_distances" and the new line setting the return_value
      with patch("prxteinmpnn.ensemble.ci.compute_component_distances") as mock_distances:
          # Define a realistic 2D distance matrix for the mock to return
          distance_matrix = jnp.array([
              [0.0, 1.0, 2.0],
              [1.0, 0.0, 1.0],
              [2.0, 1.0, 0.0],
          ])
          mock_distances.return_value = distance_matrix # <--- THIS IS THE FIX

          with patch("prxteinmpnn.ensemble.ci.dbscan_cluster") as mock_dbscan:
              # Setup mock
              mock_cluster_result = MagicMock()
              mock_cluster_result.coarse_graining_matrix = jnp.eye(3)
              mock_cluster_result.plug_in_entropy = 0.0
              mock_cluster_result.state_probabilities = jnp.array([0.4, 0.3, 0.3])
              mock_dbscan.return_value = mock_cluster_result
              
              result = infer_states(
                gmm=mock_gmm,
                features=sample_features,
                eps_std_scale=eps_std_scale,
              )
              
              # Check that the result contains the correct eps_std_scale
              assert hasattr(result, 'dbscan_eps')
              assert isinstance(result.dbscan_eps, (float, jnp.ndarray))

  @pytest.mark.parametrize("min_cluster_weight", [0.001, 0.01, 0.1])
  def test_different_min_cluster_weights(
    self,
    mock_gmm,
    sample_features,
    min_cluster_weight,
  ):
    """Test inference with different minimum cluster weight thresholds.
    
    Args:
      mock_gmm: Mock GMM fixture.
      sample_features: Sample features fixture.
      min_cluster_weight: Minimum cluster weight to test.
    """
    with patch("prxteinmpnn.ensemble.ci.compute_component_distances") as mock_distances:
      mock_distances.return_value = jnp.array([[0.0, 1.4, 2.8], [1.4, 0.0, 1.4], [2.8, 1.4, 0.0]])
      with patch("prxteinmpnn.ensemble.ci.dbscan_cluster") as mock_dbscan:
        # Setup mock
        mock_cluster_result = MagicMock()
        mock_cluster_result.coarse_graining_matrix = jnp.eye(3)
        mock_cluster_result.plug_in_entropy = 0.0
        mock_cluster_result.state_probabilities = jnp.array([0.4, 0.3, 0.3])
        mock_dbscan.return_value = mock_cluster_result
        
        result = infer_states(
          gmm=mock_gmm,
          features=sample_features,
          min_cluster_weight=min_cluster_weight,
        )
        
        assert result.min_cluster_weight == min_cluster_weight

  def test_edge_case_single_component(self, sample_features):
    """Test inference with single component GMM.
    
    Args:
      sample_features: Sample features fixture.
    """
    # Create single component GMM
    gmm = MagicMock(spec=GaussianMixtureModelJax)
    gmm.n_components = 1
    gmm.means = jnp.array([[1.0, 1.0]])
    gmm.weights = jnp.array([1.0])
    gmm.predict_proba.return_value = jnp.ones((5, 1, 1, 1))
    
    with patch("prxteinmpnn.ensemble.ci.compute_component_distances") as mock_distances:
      with patch("prxteinmpnn.ensemble.ci.dbscan_cluster") as mock_dbscan:
        mock_distances.return_value = jnp.array([[0.0]])
        
        mock_cluster_result = MagicMock()
        mock_cluster_result.coarse_graining_matrix = jnp.array([[1.0]])
        mock_cluster_result.plug_in_entropy = 0.0
        mock_cluster_result.state_probabilities = jnp.array([1.0])
        mock_dbscan.return_value = mock_cluster_result
        
        result = infer_states(gmm=gmm, features=sample_features)
        
        assert result.n_states == 1
        chex.assert_shape(result.state_trajectory, (5,))

  def test_feature_shape_compatibility(self, mock_gmm):
    """Test that function works with different feature shapes.
    
    Args:
      mock_gmm: Mock GMM fixture.
    """
    # Test with different feature dimensions
    features_2d = jnp.ones((10, 5))
    features_3d = jnp.ones((10, 5, 3))
    with patch("prxteinmpnn.ensemble.ci.compute_component_distances") as mock_distances:
      mock_distances.return_value = jnp.array([[0.0, 1.4, 2.8], [1.4, 0.0, 1.4], [2.8, 1.4, 0.0]])
      with patch("prxteinmpnn.ensemble.ci.dbscan_cluster") as mock_dbscan:
        mock_cluster_result = MagicMock()
        mock_cluster_result.coarse_graining_matrix = jnp.eye(3)
        mock_cluster_result.plug_in_entropy = 0.0
        mock_cluster_result.state_probabilities = jnp.array([0.4, 0.3, 0.3])
        mock_dbscan.return_value = mock_cluster_result
        
        # Should work with both shapes
        result_2d = infer_states(gmm=mock_gmm, features=features_2d)
        result_3d = infer_states(gmm=mock_gmm, features=features_3d)
        
        assert isinstance(result_2d, ConformationalStates)
        assert isinstance(result_3d, ConformationalStates)


class TestIntegration:
  """Integration tests for the conformational inference module."""

  def test_realistic_inference_workflow(self):
    """Test a realistic end-to-end inference workflow."""
    # Create realistic test data
    n_samples, n_features = 100, 10
    n_components = 3
    
    # Create mock GMM with realistic properties
    gmm = MagicMock(spec=GaussianMixtureModelJax)
    gmm.n_components = n_components
    gmm.means = jnp.array([
      jnp.ones(n_features) * i for i in range(n_components)
    ])
    gmm.weights = jnp.array([0.4, 0.35, 0.25])
    
    # Create realistic responsibility matrix
    key = jax.random.PRNGKey(42)
    responsibilities = jax.random.dirichlet(
      key, 
      jnp.ones(n_components), 
      shape=(n_samples,)
    )
    gmm.predict_proba.return_value = responsibilities[:, :, None, None]
    
    # Create features
    features = jax.random.normal(key, (n_samples, n_features))
    
    with patch("prxteinmpnn.ensemble.ci.compute_component_distances") as mock_distances:
      with patch("prxteinmpnn.ensemble.ci.dbscan_cluster") as mock_dbscan:
        # Setup realistic distance matrix
        distance_matrix = jnp.array([
          [0.0, 2.0, 4.0],
          [2.0, 0.0, 2.0],
          [4.0, 2.0, 0.0],
        ])
        mock_distances.return_value = distance_matrix
        
        # Setup clustering result
        mock_cluster_result = MagicMock()
        mock_cluster_result.coarse_graining_matrix = jnp.array([
          [1.0, 0.0],
          [0.0, 1.0],
          [0.0, 1.0],
        ])
        mock_cluster_result.plug_in_entropy = 0.6
        mock_cluster_result.state_probabilities = jnp.array([0.6, 0.4])
        mock_dbscan.return_value = mock_cluster_result
        
        # Run inference
        result = infer_states(
          gmm=gmm,
          features=features,
          eps_std_scale=1.0,
          min_cluster_weight=0.05,
        )
        
        # Validate results
        assert isinstance(result, ConformationalStates)
        assert result.n_states <= n_components
        assert len(result.state_trajectory) == n_samples
        assert result.mle_entropy >= 0.0
        assert result.cluster_entropy >= 0.0
        chex.assert_shape(result.coarse_graining_matrix, (n_components, 2))
