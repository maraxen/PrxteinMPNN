"""Tests for the sampling script with averaged node features."""
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import chex
import equinox as eqx
import h5py
import jax
import jax.numpy as jnp
import pytest

from prxteinmpnn.model.mpnn import PrxteinMPNN
from prxteinmpnn.run.sampling import SamplingSpecification, sample
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
        chain_index=jnp.zeros((1, 10)),
    )

@pytest.fixture
def mock_model():
    """Fixture for a small real PrxteinMPNN model."""
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
    # Set to inference mode to avoid dropout randomness requiring keys
    return eqx.tree_inference(model, value=True)

@pytest.mark.parametrize(
    "average_mode, expected_shape",
    [
        ("inputs_and_noise", (1, 3, 10)),
        ("noise_levels", (1, 3, 10)),
        ("inputs", (1, 6, 10)),
    ],
)
def test_sample_averaged_non_streaming(
    mock_protein, mock_model, average_mode, expected_shape,
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

        chex.assert_shape(result["sequences"], expected_shape)
        chex.assert_shape(result["logits"], expected_shape + (21,))
        chex.assert_tree_all_finite((result["sequences"], result["logits"]))
        assert "sequences" in result
        assert "logits" in result

def test_sample_averaged_streaming(mock_protein, mock_model):
    """Test the sample function with averaged node features (streaming)."""
    with tempfile.TemporaryDirectory() as tempdir:
        output_h5_path = Path(tempdir) / "output_averaged.h5"
        with patch("prxteinmpnn.run.sampling.prep_protein_stream_and_model", return_value=([mock_protein], mock_model)):
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
