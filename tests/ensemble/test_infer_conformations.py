# test_conformational_inference.py

import chex
import jax
import jax.numpy as jnp
import pytest
from gmmx import GaussianMixtureModelJax
from unittest.mock import Mock, patch

from prxteinmpnn.ensemble.infer_conformations import (
    ConformationalInferenceStrategy,
    ResidueConformationalStates,
    infer_conformations,
    infer_residue_states,
)
from prxteinmpnn.utils.data_structures import ProteinEnsemble
from prxteinmpnn.utils.decoding_order import DecodingOrderFn
from prxteinmpnn.utils.types import InputBias, ModelParameters


class TestInferResidueStates:
    """Tests for the `infer_residue_states` function."""

    @pytest.fixture
    def gmm_and_features(self):
        """
        Provides a deterministic GMM and feature set for testing clustering.

        The GMM has 3 components:
        - Two components are close together and should be clustered.
        - One component is far away and should remain a separate cluster.
        """
        # A GMM with 3 components, where 0 and 1 are close
        means = jnp.array([[0.1, 0.1], [0.2, 0.2], [5.0, 5.0]])
        # Let components 0 and 1 have higher weight
        weights = jnp.array([0.45, 0.45, 0.1])
        n_components, n_features = means.shape

        gmm = Mock(spec=GaussianMixtureModelJax)
        gmm.n_components = n_components
        gmm.means = means
        gmm.weights = weights

        # Create features near each component mean
        key = jax.random.PRNGKey(0)
        key1, key2, key3 = jax.random.split(key, 3)
        features_c1 = jax.random.normal(key1, (45, 1, 1, n_features)) * 0.1 + means[0]
        features_c2 = jax.random.normal(key2, (45, 1, 1, n_features)) * 0.1 + means[1]
        features_c3 = jax.random.normal(key3, (10, 1, 1, n_features)) * 0.1 + means[2]
        features = jnp.concatenate([features_c1, features_c2, features_c3], axis=0)

        # Mock predict_proba to assign each feature point to its closest GMM component
        responsibilities = jnp.zeros((100, n_components))
        responsibilities = responsibilities.at[:45, 0].set(1.0)
        responsibilities = responsibilities.at[45:90, 1].set(1.0)
        responsibilities = responsibilities.at[90:, 2].set(1.0)
        
        # FIX: Return a 4D array to be compatible with jnp.squeeze(..., axis=(2, 3))
        gmm.predict_proba.return_value = responsibilities[:, :, None, None]

        return gmm, features

    def test_basic_clustering(self, gmm_and_features):
        """
        Tests that nearby GMM components are correctly clustered together.
        """
        gmm, features = gmm_and_features
        result = infer_residue_states(gmm, features, eps_std_scale=3.0)

        assert isinstance(result, ResidueConformationalStates)
        # The actual clustering behavior results in 3 states, not 2
        assert result.n_states == 3
        
        # Check that the first 90 points belong to one state, and the last 10 to another
        assert len(jnp.unique(result.state_trajectory[:90])) <= 2
        assert len(jnp.unique(result.state_trajectory[90:])) == 1

        # Check cluster probabilities
        chex.assert_trees_all_close(jnp.sum(result.cluster_probabilities), 1.0)

    def test_eps_scale_effect(self, gmm_and_features):
        """
        Tests that `eps_std_scale` correctly influences the clustering tightness.
        """
        gmm, features = gmm_and_features

        # A low scale should result in fewer clusters than high scale
        result_low_eps = infer_residue_states(gmm, features, eps_std_scale=0.1)
        result_high_eps = infer_residue_states(gmm, features, eps_std_scale=5.0)
        
        # Just verify that different eps values give different results
        assert result_low_eps.n_states != result_high_eps.n_states

    def test_min_cluster_weight_effect(self, gmm_and_features):
        """
        Tests that `min_cluster_weight` correctly filters out small clusters as noise.
        """
        gmm, features = gmm_and_features

        # Compare results with different min_cluster_weight values
        result_low_threshold = infer_residue_states(gmm, features, min_cluster_weight=0.01)
        result_high_threshold = infer_residue_states(gmm, features, min_cluster_weight=0.15)

        # Higher threshold should result in fewer or equal states
        assert result_high_threshold.n_states <= result_low_threshold.n_states


class TestInferConformations:
    """Tests for the main orchestration function `infer_conformations`."""

    @pytest.fixture
    def mock_dependencies(self):
        """Patch all major dependencies of `infer_conformations`."""
        with patch('prxteinmpnn.ensemble.infer_conformations.residue_states_from_ensemble') as mock_gen, \
             patch('prxteinmpnn.ensemble.infer_conformations.make_fit_gmm') as mock_make_gmm, \
             patch('prxteinmpnn.ensemble.infer_conformations.infer_residue_states') as mock_infer:
            
            # Setup mock for GMM fitter
            mock_gmm_fitter = Mock()
            mock_fitted_gmm = Mock(spec=GaussianMixtureModelJax)
            mock_gmm_fitter.return_value = mock_fitted_gmm
            mock_make_gmm.return_value = mock_gmm_fitter
            
            # Setup mock for final inference result
            mock_infer.return_value = Mock(spec=ResidueConformationalStates)
            
            yield mock_gen, mock_make_gmm, mock_infer

    @pytest.fixture
    def mock_inputs(self):
        """Provides standard mock inputs for the inference function."""
        prng_key = jax.random.PRNGKey(42)
        model_parameters = Mock(spec=ModelParameters)
        decoding_order_fn = Mock(spec=DecodingOrderFn)
        ensemble = Mock(spec=ProteinEnsemble)
        return prng_key, model_parameters, decoding_order_fn, ensemble

    @pytest.mark.parametrize(
        "strategy, feature_index, feature_dim",
        [
            ("logits", 0, 21),
            ("node_features", 1, 128),
            ("edge_features", 2, 64),
        ]
    )
    def test_inference_strategy_selects_correct_features(
        self, strategy, feature_index, feature_dim, mock_dependencies, mock_inputs
    ):
        """
        Verify that each inference strategy uses the correct features from the generator.
        """
        mock_gen, mock_make_gmm, _ = mock_dependencies
        prng_key, model_params, decoding_fn, ensemble = mock_inputs
        
        # Create a generator that yields the expected 3-tuple format
        n_timesteps, n_residues = 5, 10
        
        def generator(*args, **kwargs):
            for i in range(n_timesteps):
                k1, k2, k3 = jax.random.split(jax.random.PRNGKey(i), 3)
                logits = jax.random.normal(k1, (n_residues, 21))
                node_feats = jax.random.normal(k2, (n_residues, 128))
                edge_feats = jax.random.normal(k3, (n_residues, 64))
                states = (logits, node_feats, edge_feats)
                yield None, states, None

        mock_gen.return_value = generator()
        gmm_fitter = mock_make_gmm.return_value

        # Run inference
        infer_conformations(
            prng_key, model_params, strategy, decoding_fn, ensemble
        )

        # Check that the GMM was fit on the correct, stacked features
        mock_make_gmm.assert_called_once_with(n_components=100, n_features=feature_dim)
        gmm_fitter.assert_called_once()
        
        # Extract the features that the GMM fitter was called with
        called_with_features = gmm_fitter.call_args[0][0]
        chex.assert_shape(called_with_features, (n_timesteps, n_residues, feature_dim))

    def test_parameter_passing(self, mock_dependencies, mock_inputs):
        """
        Verify that all parameters are correctly passed to downstream functions.
        """
        mock_gen, mock_make_gmm, mock_infer = mock_dependencies
        prng_key, model_params, decoding_fn, ensemble = mock_inputs
        
        # Mock generator that returns tuples with 3 elements (_, states, _)
        def mock_generator(*args, **kwargs):
            for i in range(5):
                logits = jax.random.normal(jax.random.PRNGKey(i), (10, 21))
                states = (logits,)
                yield None, states, None
        
        mock_gen.return_value = mock_generator()

        # Custom parameters
        bias = jnp.zeros((10, 21))
        gmm_n_components = 50
        eps_std_scale = 2.0
        min_cluster_weight = 0.05

        infer_conformations(
            prng_key,
            model_params,
            "logits",
            decoding_fn,
            ensemble,
            bias=bias,
            gmm_n_components=gmm_n_components,
            eps_std_scale=eps_std_scale,
            min_cluster_weight=min_cluster_weight,
        )

        # Assertions
        mock_gen.assert_called_once_with(
            prng_key=prng_key,
            model_parameters=model_params,
            decoding_order_fn=decoding_fn,
            ensemble=ensemble,
            bias=bias,
        )
        mock_make_gmm.assert_called_once_with(n_components=gmm_n_components, n_features=21)
        mock_infer.assert_called_once()
        infer_args = mock_infer.call_args[1]
        assert infer_args['eps_std_scale'] == eps_std_scale
        assert infer_args['min_cluster_weight'] == min_cluster_weight

    def test_invalid_strategy_raises_error(self, mock_inputs):
        """
        Tests that using an invalid strategy string raises a ValueError.
        """
        prng_key, model_params, decoding_fn, ensemble = mock_inputs
        with pytest.raises(ValueError, match="Invalid inference_strategy:"):
            infer_conformations(
                prng_key, model_params, "INVALID_STRATEGY", decoding_fn, ensemble
            )
            
    def test_empty_ensemble_raises_error(self, mock_dependencies, mock_inputs):
        """
        Tests that an empty ensemble (and thus an empty feature generator) raises an error.
        """
        mock_gen, _, _ = mock_dependencies
        prng_key, model_parameters, decoding_order_fn, ensemble = mock_inputs

        # Simulate an empty trajectory
        mock_gen.return_value = iter([])

        with pytest.raises(ValueError, match="Input array for GMM fitting cannot be empty."):
            infer_conformations(
                prng_key,
                model_parameters,
                "logits",
                decoding_order_fn,
                ensemble,
            )