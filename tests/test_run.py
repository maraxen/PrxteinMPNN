"""Tests for the core user interface module."""

from unittest.mock import Mock, patch

import jax
import jax.numpy as jnp
import pytest

from prxteinmpnn.run import (
    categorical_jacobian,
    sample,
    score,
)
from prxteinmpnn.utils.data_structures import Protein, ProteinBatch


@pytest.fixture
def mock_protein_batch() -> ProteinBatch:
    """Create a mock ProteinBatch for testing."""
    protein = Protein(
        coordinates=jnp.ones((10, 37, 3)),
        aatype=jnp.zeros(10, dtype=jnp.int8),
        one_hot_sequence=jax.nn.one_hot(jnp.zeros(10, dtype=jnp.int8), 21),
        atom_mask=jnp.ones((10, 37)),
        residue_index=jnp.arange(10),
        chain_index=jnp.zeros(10),
        dihedrals=None,
        mapping=None,
    )
    return jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, axis=0), protein)


class TestScore:
    """Test the score function."""

    def test_score_single_structure_single_sequence(
        self, mock_protein_batch: ProteinBatch
    ) -> None:
        """Test scoring a single structure with a single sequence."""
        with patch(
            "prxteinmpnn.io.loaders.create_protein_dataset"
        ) as mock_create_dataset, patch(
            "prxteinmpnn.run.get_mpnn_model"
        ) as mock_get_model, patch(
            "prxteinmpnn.run.make_score_sequence"
        ) as mock_make_score, patch(
            "prxteinmpnn.utils.aa_convert.string_to_protein_sequence"
        ) as mock_string_to_seq:
            mock_create_dataset.return_value = [mock_protein_batch]
            mock_get_model.return_value = {"params": {}}
            mock_score_fn = Mock(return_value=(1.0, jnp.ones((10, 21)), {}))
            mock_make_score.return_value = mock_score_fn
            mock_string_to_seq.return_value = jnp.arange(10)

            result = score(
                inputs="test.pdb",
                sequences_to_score=["ACDEFGHIKL"],
                backbone_noise=0.1,
            )

            assert "scores" in result
            assert "logits" in result
            assert "metadata" in result
            assert isinstance(result["scores"], jax.Array)
            assert isinstance(result["logits"], jax.Array)


class TestSample:
    """Test the sample function."""

    def test_sample_basic(self, mock_protein_batch: ProteinBatch) -> None:
        """Test basic sampling."""
        with patch(
            "prxteinmpnn.io.loaders.create_protein_dataset"
        ) as mock_create_dataset, patch(
            "prxteinmpnn.run.get_mpnn_model"
        ) as mock_get_model, patch(
            "prxteinmpnn.run.make_sample_sequences"
        ) as mock_make_sample:
            mock_create_dataset.return_value = [mock_protein_batch]
            mock_get_model.return_value = {"params": {}}
            mock_sample_fn = Mock(
                return_value=(
                    jnp.ones((1, 10)),
                    jnp.ones((1, 10, 21)),
                    jnp.ones((1, 10)),
                )
            )
            mock_make_sample.return_value = mock_sample_fn

            result = sample(
                inputs="test.pdb",
                num_samples=1,
                temperature=0.1,
            )

            assert "sequences" in result
            assert "logits" in result
            assert "metadata" in result
            assert isinstance(result["sequences"], jax.Array)
            assert isinstance(result["logits"], jax.Array)


class TestCategoricalJacobian:
    """Test the categorical_jacobian function."""

    def test_categorical_jacobian_basic(self, mock_protein_batch: ProteinBatch) -> None:
        """Test basic categorical jacobian calculation."""
        with patch(
            "prxteinmpnn.io.loaders.create_protein_dataset"
        ) as mock_create_dataset, patch(
            "prxteinmpnn.run.get_mpnn_model"
        ) as mock_get_model, patch(
            "prxteinmpnn.run.make_conditional_logits_fn"
        ) as mock_make_logits_fn:
            mock_create_dataset.return_value = [mock_protein_batch]
            mock_get_model.return_value = {"params": {}}
            
            # Mock the logits function to return something with the correct shape
            mock_make_logits_fn.return_value = mock_make_logits_fn.return_value = lambda *args, **kwargs: (jnp.ones((10, 21)), None, None)

            result = categorical_jacobian(
                inputs="test.pdb",
                backbone_noise=0.1,
            )
            assert "categorical_jacobians" in result
            assert isinstance(result["categorical_jacobians"], jax.Array)
            # Shape: (batch, noise, L, C, L, C)
            assert result["categorical_jacobians"].shape == (1, 1, 10, 21, 10, 21)
