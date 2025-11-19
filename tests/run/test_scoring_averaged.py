"""Tests for averaged scoring functionality."""
import jax
import jax.numpy as jnp
import pytest
from unittest.mock import patch, MagicMock
import chex

from prxteinmpnn.run.scoring import score
from prxteinmpnn.run.specs import ScoringSpecification
from prxteinmpnn.utils.data_structures import Protein

@pytest.fixture
def mock_protein():
    """Creates a mock Protein object for testing."""
    # Batch size 1, Length 10
    aatype = jnp.zeros((1, 10), dtype=jnp.int8)
    return Protein(
        coordinates=jnp.ones((1, 10, 4, 3)),
        mask=jnp.ones((1, 10)),
        residue_index=jnp.arange(10)[None, :],
        chain_index=jnp.zeros((1, 10)),
        aatype=aatype,
        one_hot_sequence=jnp.eye(21)[aatype],
    )

@pytest.fixture
def mock_model():
    """Creates a mock model for testing."""
    model = MagicMock()
    # Mock the return of make_encoding_sampling_split_fn components if needed
    # But score calls get_averaged_encodings which uses model.
    # We might need to mock get_averaged_encodings directly to avoid complex model mocking.
    return model

@patch("prxteinmpnn.run.scoring.prep_protein_stream_and_model")
@patch("prxteinmpnn.run.scoring.get_averaged_encodings")
@patch("prxteinmpnn.run.scoring.score_sequence_with_encoding")
def test_score_averaged_inputs_and_noise(mock_score_seq, mock_get_enc, mock_prep, mock_protein, mock_model):
    """Test scoring with average_node_features=True and mode='inputs_and_noise'."""
    # Arrange
    mock_prep.return_value = ([mock_protein], mock_model)
    
    # Mock averaged encodings: (avg_node, avg_edge, neighbors, mask, ar_mask)
    # Shapes: node (L, D), edge (L, L, D), neighbors (N, M, L, K), mask (N, M, L), ar_mask (N, M, L, L)
    # For inputs_and_noise, we expect flattened batch dims in the end?
    # Actually get_averaged_encodings returns what the model produces.
    # Let's mock simple shapes.
    L = 10
    mock_get_enc.return_value = (
        jnp.zeros((L, 1)), # node
        jnp.zeros((L, 1, 1)), # edge
        jnp.zeros((1, 1, L, 1)), # neighbors (N=1, M=1)
        jnp.zeros((1, 1, L)), # mask
        jnp.zeros((1, 1, L, L)), # ar_mask
    )
    
    # Mock score_sequence_with_encoding return: (score, logits, decoding_order)
    mock_score_seq.return_value = (
        jnp.array(0.5), # score
        jnp.zeros((L, 21)), # logits
        jnp.arange(L) # decoding order
    )

    spec = ScoringSpecification(
        inputs=["dummy_path"],
        sequences_to_score=["G" * L],
        average_node_features=True,
        average_encoding_mode="inputs_and_noise",
        backbone_noise=[0.1]
    )

    # Act
    results = score(spec)

    # Assert
    chex.assert_shape(results["scores"], (1, 1))
    chex.assert_shape(results["logits"], (1, 1, L, 21))
    chex.assert_tree_all_finite((results["scores"], results["logits"]))
    assert "scores" in results
    assert "logits" in results
    
    # Verify mocks called
    mock_get_enc.assert_called_once()
    # mock_score_seq called via vmap, so difficult to check exact call count easily without side effects,
    # but we can check it was called.
    # Actually, since we mock the function inside the module, we can check.
    assert mock_score_seq.call_count >= 1

@patch("prxteinmpnn.run.scoring.prep_protein_stream_and_model")
@patch("prxteinmpnn.run.scoring.get_averaged_encodings")
@patch("prxteinmpnn.run.scoring.score_sequence_with_encoding")
def test_score_averaged_inputs_only(mock_score_seq, mock_get_enc, mock_prep, mock_protein, mock_model):
    """Test scoring with average_node_features=True and mode='inputs'."""
    # Arrange
    mock_prep.return_value = ([mock_protein], mock_model)
    
    L = 10
    # mode='inputs' means we average over inputs (axis 0), keep noise (axis 1).
    # encodings: node (Batch, L, D) -> (Noise, L, D) ?
    # get_averaged_encodings returns (avg_node, avg_edge, neighbors, mask, ar_mask)
    # If mode='inputs', avg_node has shape (Noise, L, D).
    # neighbors has shape (Inputs, Noise, L, K).
    
    # Let's assume 2 noise levels.
    mock_get_enc.return_value = (
        jnp.zeros((2, L, 1)), # node (Noise=2)
        jnp.zeros((2, L, 1, 1)), # edge
        jnp.zeros((1, 2, L, 1)), # neighbors (Inputs=1, Noise=2)
        jnp.zeros((1, 2, L)), # mask
        jnp.zeros((1, 2, L, L)), # ar_mask
    )
    
    mock_score_seq.return_value = (
        jnp.array(0.5),
        jnp.zeros((L, 21)),
        jnp.arange(L)
    )

    spec = ScoringSpecification(
        inputs=["dummy_path"],
        sequences_to_score=["G" * L],
        average_node_features=True,
        average_encoding_mode="inputs", # Average over inputs, keep noise
        backbone_noise=[0.1, 0.2]
    )

    # Act
    results = score(spec)

    # Assert
    chex.assert_shape(results["scores"], (2, 1))
    chex.assert_shape(results["logits"], (2, 1, L, 21))
    chex.assert_tree_all_finite((results["scores"], results["logits"]))
    assert "scores" in results

