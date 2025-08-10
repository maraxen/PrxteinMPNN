"""Tests for ensemble residue states functionality."""

import jax
import jax.numpy as jnp
import pytest
import chex
from unittest.mock import Mock, patch

from prxteinmpnn.ensemble.residue_states import residue_states_from_ensemble
from prxteinmpnn.utils.data_structures import ProteinStructure

L_GLOBAL = 10  # Define a global sequence length for consistency

@pytest.fixture
def mock_model_parameters():
    """Create mock model parameters for testing."""
    return {"dummy_param": jnp.ones(1)}

@pytest.fixture
def mock_decoding_order_fn():
    """Create mock decoding order function."""
    def mock_fn(prng_key, seq_len):
        return jnp.arange(seq_len, dtype=jnp.int32), prng_key
    return mock_fn

@pytest.fixture
def mock_protein_ensemble():
    """Create a mock protein ensemble with 3 frames."""
    ensemble = []
    for i in range(3):
        structure = ProteinStructure(
            coordinates=jax.random.normal(jax.random.PRNGKey(i), (L_GLOBAL, 37, 3)),
            aatype=jax.random.randint(jax.random.PRNGKey(i + 10), (L_GLOBAL,), 0, 21, dtype=jnp.int8),
            atom_mask=jnp.ones((L_GLOBAL, 37)),
            residue_index=jnp.arange(L_GLOBAL),
            chain_index=jnp.zeros(L_GLOBAL, dtype=jnp.int32),
        )
        ensemble.append(structure)
    return iter(ensemble)

@patch('prxteinmpnn.ensemble.residue_states.make_conditional_logits_fn')
def test_residue_states_from_ensemble_structure(
    mock_make_logits_fn, mock_model_parameters, mock_decoding_order_fn, mock_protein_ensemble
):
    """Tests the output structure and length from a standard ensemble."""
    n_frames = 3
    key = jax.random.PRNGKey(42)

    # Mock the final function that returns the features tuple
    mock_logits_fn = Mock(return_value=(
        jnp.ones((L_GLOBAL, 21)), jnp.zeros((L_GLOBAL, 128)), jnp.zeros((L_GLOBAL, 32))
    ))
    mock_make_logits_fn.return_value = mock_logits_fn

    states_generator = residue_states_from_ensemble(
        prng_key=key,
        model_parameters=mock_model_parameters,
        decoding_order_fn=mock_decoding_order_fn,
        ensemble=mock_protein_ensemble,
    )
    states_list = list(states_generator)

    # 1. Check the number of yielded items
    assert len(states_list) == n_frames, "Should yield one tuple per frame in the ensemble."

    # 2. Check the structure of each yielded item
    logits, node_features, edge_features = states_list[0]
    
    assert isinstance(logits, jax.Array) and logits.shape == (L_GLOBAL, 21)
    assert isinstance(node_features, jax.Array) and node_features.shape == (L_GLOBAL, 128)
    assert isinstance(edge_features, jax.Array) and edge_features.shape == (L_GLOBAL, 32)
    
def test_residue_states_empty_ensemble(mock_model_parameters, mock_decoding_order_fn):
    """Tests that an empty ensemble correctly raises a ValueError."""
    with pytest.raises(ValueError, match="Ensemble is empty."):
        list(residue_states_from_ensemble(
            prng_key=jax.random.PRNGKey(42),
            model_parameters=mock_model_parameters,
            decoding_order_fn=mock_decoding_order_fn,
            ensemble=iter([]),
        ))

@patch('prxteinmpnn.ensemble.residue_states.make_conditional_logits_fn')
def test_residue_states_single_frame_ensemble(
    mock_make_logits_fn, mock_model_parameters, mock_decoding_order_fn
):
    """Tests that an ensemble with a single frame yields exactly one result."""
    key = jax.random.PRNGKey(456)
    single_frame = ProteinStructure(
        coordinates=jax.random.normal(key, (L_GLOBAL, 37, 3)),
        aatype=jax.random.randint(key, (L_GLOBAL,), 0, 21, dtype=jnp.int8),
        atom_mask=jnp.ones((L_GLOBAL, 37)),
        residue_index=jnp.arange(L_GLOBAL),
        chain_index=jnp.zeros(L_GLOBAL, dtype=jnp.int32),
    )
    mock_make_logits_fn.return_value = Mock(return_value=(None, None, None))
    
    states_list = list(residue_states_from_ensemble(
        prng_key=key,
        model_parameters=mock_model_parameters,
        decoding_order_fn=mock_decoding_order_fn,
        ensemble=iter([single_frame]),
    ))
    
    assert len(states_list) == 1, "A single-frame ensemble should yield exactly one result."

@patch('jax.random.fold_in')
@patch('prxteinmpnn.ensemble.residue_states.make_conditional_logits_fn')
def test_prng_key_generation_with_fold_in(
    mock_make_logits_fn, mock_fold_in, mock_model_parameters, mock_decoding_order_fn, mock_protein_ensemble
):
    """Tests that PRNG keys are correctly generated using fold_in for each frame."""
    key = jax.random.PRNGKey(789)
    n_frames = 3
    # The actual return value doesn't matter, only the number of calls
    mock_make_logits_fn.return_value.return_value = (None, None, None)

    # Consume the generator to trigger the calls
    list(residue_states_from_ensemble(
        prng_key=key,
        model_parameters=mock_model_parameters,
        decoding_order_fn=mock_decoding_order_fn,
        ensemble=mock_protein_ensemble,
    ))

    # The main key is split once for order vs. loop, then fold_in is used
    assert mock_fold_in.call_count == n_frames
    
    # Check that it was called with indices 0, 1, 2, ...
    for i in range(n_frames):
        call_index = mock_fold_in.call_args_list[i].args[1]
        assert call_index == i