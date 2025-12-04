"""Tests for averaged scoring functionality."""
from unittest.mock import MagicMock, patch

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from prxteinmpnn.model.mpnn import PrxteinMPNN
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
    """Creates a small real model for testing."""
    key = jax.random.key(0)
    model = PrxteinMPNN(
        node_features=16,
        edge_features=16,
        hidden_features=16,
        num_encoder_layers=1,
        num_decoder_layers=1,
        k_neighbors=5,
        key=key,
    )
    return eqx.tree_inference(model, value=True)

@patch("prxteinmpnn.run.scoring.prep_protein_stream_and_model")
def test_score_averaged_inputs_and_noise(mock_prep, mock_protein, mock_model):
    """Test scoring with average_node_features=True and mode='inputs_and_noise'."""
    # Arrange
    mock_prep.return_value = ([mock_protein], mock_model)

    L = 10
    spec = ScoringSpecification(
        inputs=["dummy_path"],
        sequences_to_score=["G" * L],
        average_node_features=True,
        average_encoding_mode="inputs_and_noise",
        backbone_noise=[0.1],
    )

    # Act
    results = score(spec)

    # Assert
    chex.assert_shape(results["scores"], (1, 1))
    chex.assert_shape(results["logits"], (1, 1, L, 21))
    chex.assert_tree_all_finite((results["scores"], results["logits"]))
    assert "scores" in results
    assert "logits" in results

@patch("prxteinmpnn.run.scoring.prep_protein_stream_and_model")
def test_score_averaged_inputs_only(mock_prep, mock_protein, mock_model):
    """Test scoring with average_node_features=True and mode='inputs'."""
    # Arrange
    mock_prep.return_value = ([mock_protein], mock_model)

    L = 10
    spec = ScoringSpecification(
        inputs=["dummy_path"],
        sequences_to_score=["G" * L],
        average_node_features=True,
        average_encoding_mode="inputs", # Average over inputs, keep noise
        backbone_noise=[0.1, 0.2],
    )

    # Act
    results = score(spec)

    # Assert
    chex.assert_shape(results["scores"], (2, 1))
    chex.assert_shape(results["logits"], (2, 1, L, 21))
    chex.assert_tree_all_finite((results["scores"], results["logits"]))
    assert "scores" in results

