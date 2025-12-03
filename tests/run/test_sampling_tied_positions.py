"""Tests for tied positions in sampling."""
import jax
import jax.numpy as jnp
import pytest
from unittest.mock import MagicMock, patch

from prxteinmpnn.run.sampling import SamplingSpecification, sample
from prxteinmpnn.utils.data_structures import Protein


@pytest.mark.parametrize("tied_mode", [[[(0, 1), (0, 2)], [(0, 3), (0, 4)]]])
def test_sample_with_tied_positions(tied_mode):
    """Test sampling with tied positions using explicit position tuples.
    
    This tests the fix for the vmap dimension mismatch error where
    tie_group_map had shape (n_residues,) but vmap expected (batch_size, n_residues).
    """
    # Create a mock protein with batch dimension
    mock_protein = Protein(
        coordinates=jnp.ones((1, 10, 4, 3)),
        aatype=jnp.ones((1, 10), dtype=jnp.int8),
        one_hot_sequence=jax.nn.one_hot(jnp.ones((1, 10), dtype=jnp.int8), 21),
        mask=jnp.ones((1, 10)),
        residue_index=jnp.arange(10)[None, :],
        chain_index=jnp.zeros((1, 10), dtype=jnp.int32),
    )

    mock_model = MagicMock()
    mock_sampler_fn = MagicMock()
    mock_sampler_fn.return_value = (
        jnp.ones((10,), dtype=jnp.int8),
        jnp.ones((10, 21)),
        jnp.arange(10),
    )

    with patch(
        "prxteinmpnn.run.sampling.prep_protein_stream_and_model",
        return_value=([mock_protein], mock_model),
    ):
        with patch(
            "prxteinmpnn.run.sampling.make_sample_sequences",
            return_value=mock_sampler_fn,
        ):
            spec = SamplingSpecification(
                inputs=["test.pdb"],
                num_samples=2,
                backbone_noise=[0.1],
                tied_positions=tied_mode,
                pass_mode="inter",
            )
            result = sample(spec)

            # Verify the result has the expected shape
            assert "sequences" in result
            assert "logits" in result
            assert result["sequences"].shape[0] == 1  # batch dimension
            assert result["sequences"].shape[1] == 2  # num_samples
            assert result["sequences"].shape[-1] == 10  # sequence length



def test_sample_with_tied_positions_batch_size_1():
    """Test that tied positions work correctly with batch_size=1.
    
    This specifically tests the fix for the dimension mismatch error:
    ValueError: vmap got inconsistent sizes for array axes to be mapped:
    * most axes (5 of them) had size 1, e.g. axis 0 of argument coords of type float32[1,182,37,3];
    * one axis had size 182: axis 0 of argument current_tie_map of type int32[182]
    """
    # Create a protein with batch_size=1 and 182 residues (matching the error case)
    n_residues = 182
    mock_protein = Protein(
        coordinates=jnp.ones((1, n_residues, 4, 3)),
        aatype=jnp.ones((1, n_residues), dtype=jnp.int8),
        one_hot_sequence=jax.nn.one_hot(jnp.ones((1, n_residues), dtype=jnp.int8), 21),
        mask=jnp.ones((1, n_residues)),
        residue_index=jnp.arange(n_residues)[None, :],
        chain_index=jnp.zeros((1, n_residues), dtype=jnp.int32),
        # Add mapping for "direct" mode - single structure
        mapping=jnp.zeros((1, n_residues), dtype=jnp.int32),
    )

    mock_model = MagicMock()
    mock_sampler_fn = MagicMock()
    mock_sampler_fn.return_value = (
        jnp.ones((n_residues,), dtype=jnp.int8),
        jnp.ones((n_residues, 21)),
        jnp.arange(n_residues),
    )

    with patch(
        "prxteinmpnn.run.sampling.prep_protein_stream_and_model",
        return_value=([mock_protein], mock_model),
    ):
        with patch(
            "prxteinmpnn.run.sampling.make_sample_sequences",
            return_value=mock_sampler_fn,
        ):
            # Use "direct" mode which triggers resolve_tie_groups
            spec = SamplingSpecification(
                inputs=["test.pdb"],
                num_samples=1,
                backbone_noise=[0.0],
                tied_positions="direct",
                pass_mode="inter",
                batch_size=1,
            )
            
            # This should not raise a ValueError about dimension mismatch
            result = sample(spec)
            
            assert "sequences" in result
            assert result["sequences"].shape[-1] == n_residues


def test_sample_with_tied_positions_and_mapping():
    """Test sampling with both tied positions and structure mapping.
    
    This tests that both tie_group_map and mapping arrays are properly
    reshaped to have batch dimensions when mapping is 1D.
    """
    n_residues = 20
    mock_protein = Protein(
        coordinates=jnp.ones((1, n_residues, 4, 3)),
        aatype=jnp.ones((1, n_residues), dtype=jnp.int8),
        one_hot_sequence=jax.nn.one_hot(jnp.ones((1, n_residues), dtype=jnp.int8), 21),
        mask=jnp.ones((1, n_residues)),
        residue_index=jnp.arange(n_residues)[None, :],
        chain_index=jnp.zeros((1, n_residues), dtype=jnp.int32),
        # Add a 2D mapping array for "direct" mode
        mapping=jnp.zeros((1, n_residues), dtype=jnp.int32),
    )

    mock_model = MagicMock()
    mock_sampler_fn = MagicMock()
    mock_sampler_fn.return_value = (
        jnp.ones((n_residues,), dtype=jnp.int8),
        jnp.ones((n_residues, 21)),
        jnp.arange(n_residues),
    )

    with patch(
        "prxteinmpnn.run.sampling.prep_protein_stream_and_model",
        return_value=([mock_protein], mock_model),
    ):
        with patch(
            "prxteinmpnn.run.sampling.make_sample_sequences",
            return_value=mock_sampler_fn,
        ):
            spec = SamplingSpecification(
                inputs=["test.pdb"],
                num_samples=1,
                backbone_noise=[0.0],
                tied_positions="direct",
                pass_mode="inter",
            )
            
            # This should handle both tie_group_map and mapping reshaping
            result = sample(spec)
            
            assert "sequences" in result
            assert result["sequences"].shape[-1] == n_residues


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_sample_tied_positions_various_batch_sizes(batch_size):
    """Test that tied positions work with various batch sizes.
    
    This ensures the atleast_2d fix works correctly for different batch sizes.
    """
    n_residues = 15
    mock_protein = Protein(
        coordinates=jnp.ones((batch_size, n_residues, 4, 3)),
        aatype=jnp.ones((batch_size, n_residues), dtype=jnp.int8),
        one_hot_sequence=jax.nn.one_hot(
            jnp.ones((batch_size, n_residues), dtype=jnp.int8), 21
        ),
        mask=jnp.ones((batch_size, n_residues)),
        residue_index=jnp.tile(jnp.arange(n_residues)[None, :], (batch_size, 1)),
        chain_index=jnp.zeros((batch_size, n_residues), dtype=jnp.int32),
        # Add mapping for "direct" mode
        mapping=jnp.zeros((batch_size, n_residues), dtype=jnp.int32),
    )

    mock_model = MagicMock()
    mock_sampler_fn = MagicMock()
    mock_sampler_fn.return_value = (
        jnp.ones((n_residues,), dtype=jnp.int8),
        jnp.ones((n_residues, 21)),
        jnp.arange(n_residues),
    )

    with patch(
        "prxteinmpnn.run.sampling.prep_protein_stream_and_model",
        return_value=([mock_protein], mock_model),
    ):
        with patch(
            "prxteinmpnn.run.sampling.make_sample_sequences",
            return_value=mock_sampler_fn,
        ):
            spec = SamplingSpecification(
                inputs=["test.pdb"],
                num_samples=1,
                backbone_noise=[0.0],
                tied_positions="direct",
                pass_mode="inter",
            )
            
            result = sample(spec)
            
            assert "sequences" in result
            # First dimension should match batch_size
            assert result["sequences"].shape[0] == batch_size
