"""Tests for the sampling script."""
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import chex
import h5py
import jax
import jax.numpy as jnp
import pytest

from prxteinmpnn.run.sampling import (
    SamplingSpecification,
    sample,
)
from prxteinmpnn.utils.data_structures import Protein


@pytest.mark.parametrize("use_spec", [True, False])
def test_sample_non_streaming(use_spec):
    """Test the sample function for non-streaming."""
    # Add batch dimension to mock protein
    mock_protein = Protein(
        coordinates=jnp.ones((1, 10, 4, 3)),
        aatype=jnp.ones((1, 10), dtype=jnp.int8),
        one_hot_sequence=jax.nn.one_hot(jnp.ones((1, 10), dtype=jnp.int8), 21),
        mask=jnp.ones((1, 10)),
        residue_index=jnp.arange(10)[None, :],
        chain_index=jnp.zeros((1, 10)),
    )

    # Mock the model object to be a callable
    mock_model = MagicMock()
    # It should return a single sequence and logits, as vmap handles batching
    mock_sampler_fn = MagicMock()
    mock_sampler_fn.return_value = (jnp.ones((10,), dtype=jnp.int8), jnp.ones((10, 21)), jnp.arange(10))

    # prep_protein_stream_and_model returns an iterator and the model
    with patch("prxteinmpnn.run.sampling.prep_protein_stream_and_model", return_value=([mock_protein], mock_model)):
        with patch("prxteinmpnn.run.sampling.make_sample_sequences", return_value=mock_sampler_fn):
            if use_spec:
                spec = SamplingSpecification(
                    inputs=["1ubq.pdb"],
                    num_samples=2,
                    backbone_noise=[0.1],
                )
                result = sample(spec)
            else:
                result = sample(
                    inputs=["1ubq.pdb"],
                    num_samples=2,
                    backbone_noise=[0.1],
                )

            chex.assert_shape(result["sequences"], (1, 2, 1, 10))
            chex.assert_shape(result["logits"], (1, 2, 1, 10, 21))
            chex.assert_tree_all_finite((result["sequences"], result["logits"]))
            assert "sequences" in result
            assert "logits" in result

def test_sample_streaming():
    """Test the streaming functionality of the sample function."""
    mock_protein = Protein(
        coordinates=jnp.ones((1, 10, 4, 3)),
        aatype=jnp.ones((1, 10), dtype=jnp.int8),
        one_hot_sequence=jax.nn.one_hot(jnp.ones((1, 10), dtype=jnp.int8), 21),
        mask=jnp.ones((1, 10)),
        residue_index=jnp.arange(10)[None, :],
        chain_index=jnp.zeros((1, 10)),
    )

    mock_model = MagicMock()
    mock_sampler_fn = MagicMock()
    mock_sampler_fn.return_value = (jnp.ones((10,), dtype=jnp.int8), jnp.ones((10, 21)), jnp.arange(10))


    with tempfile.TemporaryDirectory() as tempdir:
        output_h5_path = Path(tempdir) / "output.h5"
        with patch("prxteinmpnn.run.sampling.prep_protein_stream_and_model", return_value=([mock_protein], mock_model)):
            with patch("prxteinmpnn.run.sampling.make_sample_sequences", return_value=mock_sampler_fn):
                spec = SamplingSpecification(
                    inputs=["1ubq.pdb"],
                    num_samples=2,
                    backbone_noise=[0.1, 0.2],
                    output_h5_path=output_h5_path,
                )
                result = sample(spec)

                assert "output_h5_path" in result
                assert Path(result["output_h5_path"]).exists()

                with h5py.File(output_h5_path, "r") as f:
                    assert "structure_0" in f
                    assert "sequences" in f["structure_0"]
                    assert "logits" in f["structure_0"]
                    assert f["structure_0/sequences"].shape == (2, 2, 10)
                    assert f["structure_0/logits"].shape == (2, 2, 10, 21)




