"""Tests for the sampling script with averaged node features."""
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import h5py
import jax
import jax.numpy as jnp
import pytest
from prxteinmpnn.run.sampling import sample, SamplingSpecification
from prxteinmpnn.utils.data_structures import Protein

@pytest.fixture
def mock_protein():
    """Fixture for a mock Protein object."""
    return Protein(
        coordinates=jnp.ones((1, 10, 4, 3)),
        aatype=jnp.ones((1, 10), dtype=jnp.int8),
        one_hot_sequence=jax.nn.one_hot(jnp.ones((1, 10), dtype=jnp.int8), 21),
        mask=jnp.ones((1, 10)),
        residue_index=jnp.arange(10)[None, :],
        chain_index=jnp.zeros((1, 10))
    )

@pytest.fixture
def mock_model():
    """Fixture for a mock PrxteinMPNN model."""
    model = MagicMock()
    # Mock the model's w_s_embed.num_embeddings for decode_fn
    model.w_s_embed.num_embeddings = 21
    model.features.return_value = (
        jnp.ones((10, 48, 128)),  # edge_features
        jnp.ones((10, 48), dtype=jnp.int32),  # neighbor_indices
        jnp.ones((10, 128)),  # initial_node_features
        None,
    )
    model.encoder.return_value = (
        jnp.ones((10, 128)),  # node_features
        jnp.ones((10, 48, 128)),  # processed_edge_features
    )
    model.decoder.call_conditional.return_value = jnp.ones((10, 128))
    model.w_out.return_value = jnp.ones((21,))
    return model

@pytest.mark.parametrize(
    "average_mode, expected_shape",
    [
        ("inputs_and_noise", (1, 3, 10)),
        ("noise_levels", (1, 3, 10)),
        ("inputs", (1, 6, 10)),
    ],
)
def test_sample_averaged_non_streaming(
    mock_protein, mock_model, average_mode, expected_shape
):
    """Test the sample function with averaged node features (non-streaming)."""
    # The new implementation is too complex to mock individual functions inside.
    # We rely on the mock_model to test the end-to-end integration.
    with patch(
        "prxteinmpnn.run.sampling.prep_protein_stream_and_model",
        return_value=([mock_protein], mock_model),
    ):
        spec = SamplingSpecification(
            inputs=["dummy.pdb"],
            num_samples=3,
            backbone_noise=[0.1, 0.2],
            average_node_features=True,
            average_encoding_mode=average_mode,
        )
        result = sample(spec)

        assert "sequences" in result
        assert "logits" in result
        assert result["sequences"].shape == expected_shape
        assert result["logits"].shape == expected_shape + (21,)

def test_sample_averaged_streaming(mock_protein, mock_model):
    """Test the sample function with averaged node features (streaming)."""
    with tempfile.TemporaryDirectory() as tempdir:
        output_h5_path = Path(tempdir) / "output_averaged.h5"
        with patch('prxteinmpnn.run.sampling.prep_protein_stream_and_model', return_value=([mock_protein], mock_model)):
            spec = SamplingSpecification(
                inputs=["dummy.pdb"],
                num_samples=2,
                backbone_noise=[0.1, 0.2],
                average_node_features=True,
                output_h5_path=output_h5_path,
                average_encoding_mode="noise_levels",
            )
            result = sample(spec)

            assert "output_h5_path" in result
            assert Path(result["output_h5_path"]).exists()

            with h5py.File(output_h5_path, "r") as f:
                assert "structure_0" in f
                assert "sequences" in f["structure_0"]
                assert "logits" in f["structure_0"]
                # Shape is num_samples * num_noise
                assert f["structure_0/sequences"].shape == (2, 10)
                assert f["structure_0/logits"].shape == (2, 10, 21)
                assert f["structure_0"].attrs["num_noise_levels"] == 1 # Averaged
