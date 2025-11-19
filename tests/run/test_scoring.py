"""Tests for the scoring module."""
import h5py
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
    aatype = jnp.zeros(10, dtype=jnp.int8)
    return Protein(
        coordinates=jnp.ones((10, 27, 3)),
        mask=jnp.ones(10),
        residue_index=jnp.arange(10),
        chain_index=jnp.zeros(10),
        aatype=aatype,
        one_hot_sequence=jnp.eye(21)[aatype],
    )

@pytest.fixture
def mock_model():
    """Creates a mock model for testing."""
    model = MagicMock()
    model.return_value = (None, jnp.zeros((10, 21)))
    return model

@patch("prxteinmpnn.run.scoring.prep_protein_stream_and_model")
def test_score_without_streaming(mock_prep, mock_protein, mock_model):
    """Test the score function without streaming to an H5 file."""
    # Arrange
    mock_prep.return_value = ([mock_protein], mock_model)
    spec = ScoringSpecification(
        inputs=["dummy_path"],
        sequences_to_score=["G" * 10, "A" * 10],
    )

    # Act
    results = score(spec)

    # Assert
    chex.assert_shape(results["scores"], (10, 1, 2))
    chex.assert_shape(results["logits"], (10, 1, 2, 10, 21))
    chex.assert_tree_all_finite((results["scores"], results["logits"]))
    assert "scores" in results
    assert "logits" in results
    assert "metadata" in results

@patch("prxteinmpnn.run.scoring.prep_protein_stream_and_model")
def test_score_with_streaming(mock_prep, mock_protein, mock_model, tmp_path):
    """Test the score function with streaming to an H5 file."""
    # Arrange
    h5_path = tmp_path / "scores.h5"
    mock_prep.return_value = ([mock_protein], mock_model)
    spec = ScoringSpecification(
        inputs=["dummy_path"],
        sequences_to_score=["G" * 10, "A" * 10],
        output_h5_path=h5_path,
    )

    # Act
    results = score(spec)

    # Assert
    assert "output_h5_path" in results
    assert "metadata" in results
    assert h5_path.exists()

@patch("prxteinmpnn.run.scoring.prep_protein_stream_and_model")
def test_score_with_ar_mask(mock_prep, mock_protein, mock_model):
    """Test the score function with a custom ar_mask."""
    # Arrange
    mock_prep.return_value = ([mock_protein], mock_model)
    ar_mask = jnp.ones((10, 10))
    spec = ScoringSpecification(
        inputs=["dummy_path"],
        sequences_to_score=["G" * 10],
        ar_mask=ar_mask,
    )

    # Act
    results = score(spec)

    # Assert
    assert "scores" in results
    assert "logits" in results
    assert "metadata" in results

@patch("prxteinmpnn.run.scoring.prep_protein_stream_and_model")
def test_score_no_scores(mock_prep, mock_model):
    """Test that score returns an empty dict if no scores are generated."""
    # Arrange
    mock_prep.return_value = ([], mock_model)
    spec = ScoringSpecification(
        inputs=["dummy_path"],
        sequences_to_score=["G" * 10],
    )

    # Act
    results = score(spec)

    # Assert
    assert results == {}

def test_score_no_sequences():
    """Test that score raises a ValueError if no sequences are provided."""
    with pytest.raises(ValueError, match="No sequences provided for scoring"):
        score(ScoringSpecification(inputs=["dummy_path"], sequences_to_score=[]))

@patch("prxteinmpnn.run.scoring.prep_protein_stream_and_model")
def test_score_spec_none(mock_prep, mock_protein, mock_model):
    """Test the score function with spec=None."""
    # Arrange
    mock_prep.return_value = ([mock_protein], mock_model)

    # Act
    results = score(inputs=["dummy_path"], sequences_to_score=["G" * 10])

    # Assert
    assert "scores" in results
    assert "logits" in results
    assert "metadata" in results
