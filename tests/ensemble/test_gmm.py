"""Unit tests for Gaussian Mixture Model functionality."""

from unittest.mock import MagicMock, patch

import chex
import jax
import jax.numpy as jnp
import pytest

from prxteinmpnn.ensemble.gmm import make_fit_gmm
from prxteinmpnn.utils.data_structures import GMM, EMFitterResult


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    key = jax.random.PRNGKey(42)
    return jax.random.normal(key, (100, 5))


class TestMakeFitGMM:
    """Test the make_fit_gmm function factory."""

    def test_returns_callable(self):
        """Test that make_fit_gmm returns a callable function."""
        fit_fn = make_fit_gmm(n_components=3)

        assert callable(fit_fn)

    @patch("prxteinmpnn.ensemble.gmm.fit_gmm_states")
    @patch("prxteinmpnn.ensemble.gmm.kmeans")
    def test_fit_gmm_workflow(self, mock_kmeans, mock_fit_gmm_states, sample_data):
        """Test the complete GMM fitting workflow."""
        # Setup mocks
        mock_labels = jnp.array([0, 1, 2, 0, 1] * 20)  # 100 labels
        mock_kmeans.return_value = mock_labels

        mock_fit_result = MagicMock(spec=EMFitterResult)
        mock_fit_gmm_states.return_value = mock_fit_result

        # Create and test fit function
        fit_fn = make_fit_gmm(n_components=3)
        key = jax.random.PRNGKey(42)

        result = fit_fn(sample_data, key)

        # Verify workflow
        mock_kmeans.assert_called_once()
        mock_fit_gmm_states.assert_called_once()

        assert result == mock_fit_result

    @patch("prxteinmpnn.ensemble.gmm.fit_gmm_states")
    @patch("prxteinmpnn.ensemble.gmm.kmeans")
    def test_responsibilities_creation(self, mock_kmeans, mock_fit_gmm_states, sample_data):
        """Test that one-hot responsibilities are created correctly from labels."""
        # Setup
        n_components = 3
        mock_labels = jnp.array([0, 1, 2, 0, 1] * 20)
        mock_kmeans.return_value = mock_labels

        mock_fit_result = MagicMock(spec=EMFitterResult)
        mock_fit_gmm_states.return_value = mock_fit_result

        # Test
        fit_fn = make_fit_gmm(n_components=n_components)
        key = jax.random.PRNGKey(42)

        fit_fn(sample_data, key)

        # Check that responsibilities were created correctly
        call_args = mock_fit_gmm_states.call_args
        initial_gmm = call_args[1]["gmm"]
        responsibilities = initial_gmm.responsibilities

        chex.assert_shape(responsibilities, (100, 3))
        # Should be one-hot encoded
        assert jnp.allclose(jnp.sum(responsibilities, axis=1), 1.0)

    @pytest.mark.parametrize("n_components", [1, 2, 3, 5])
    def test_different_dimensions(self, n_components):
        """Test GMM fitting with different dimensions."""
        with patch("prxteinmpnn.ensemble.gmm.kmeans"), patch(
            "prxteinmpnn.ensemble.gmm.fit_gmm_states"
        ):
            fit_fn = make_fit_gmm(
                n_components=n_components,
            )

            assert callable(fit_fn)

    def test_em_fitter_configuration(self, sample_data):
        """Test that EM fitter is configured correctly."""
        with patch("prxteinmpnn.ensemble.gmm.fit_gmm_states") as mock_fit_gmm_states:
            with patch("prxteinmpnn.ensemble.gmm.kmeans") as mock_kmeans:
                mock_kmeans.return_value = jnp.zeros(10, dtype=jnp.int32)
                fit_fn = make_fit_gmm(
                    n_components=3,
                    gmm_max_iters=200,
                    covariance_regularization=1e-5,
                )

                key = jax.random.PRNGKey(42)
                data = jnp.ones((10, 2))
                fit_fn(data, key)

                mock_fit_gmm_states.assert_called_once()
                call_args = mock_fit_gmm_states.call_args
                assert call_args[1]["max_iter"] == 200
                assert call_args[1]["covariance_regularization"] == 1e-5

    def test_jit_compilation(self, sample_data):
        """Test that the returned function is JIT-compatible."""
        fit_fn = make_fit_gmm(n_components=3)
        key = jax.random.PRNGKey(42)

        # This should not raise an error
        jitted_fit_fn = jax.jit(fit_fn)
        result = jitted_fit_fn(sample_data, key)
        assert result is not None
        assert isinstance(result, EMFitterResult)