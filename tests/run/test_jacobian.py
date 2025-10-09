"""Tests for the core user interface module."""
import pathlib
import tempfile
from unittest.mock import patch

import h5py
import jax
import jax.numpy as jnp
import pytest

from prxteinmpnn.run.jacobian import (
  categorical_jacobian,
  JacobianSpecification,
)
from prxteinmpnn.utils.data_structures import Protein, ProteinBatch


@pytest.fixture
def mock_protein_batch() -> ProteinBatch:
    """Create a mock ProteinBatch for testing."""
    protein = Protein(
        coordinates=jnp.ones((10, 37, 3)),
        aatype=jnp.zeros(10, dtype=jnp.int8),
        one_hot_sequence=jax.nn.one_hot(jnp.zeros(10, dtype=jnp.int8), 21),
        mask=jnp.ones((10,)),
        residue_index=jnp.arange(10),
        chain_index=jnp.zeros(10),
        dihedrals=None,
        mapping=None,
    )
    return jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, axis=0), protein)


@pytest.fixture
def mock_encoding_pytree(mock_protein_batch):
    """Create a mock encoding PyTree tuple."""
    L = mock_protein_batch.coordinates.shape[1]
    # (node_features, edge_features, neighbor_indices, mask, autoregressive_mask)
    return (
        jnp.ones((L, 128)),
        jnp.ones((L, 10, 128)),
        jnp.ones((L, 10), dtype=jnp.int32),
        jnp.ones((L,)),
        jnp.ones((L, L)),
    )


@pytest.fixture
def cif_file():
    """Create a temporary CIF file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cif", delete=False) as f:
        # A minimal CIF file content with required columns for Biotite
        f.write(
            """
data_test
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
ATOM 1 N N GLY A 1 -6.778 -1.424 4.200 1.00 0.00
"""
        )
        filepath = f.name
    yield filepath
    pathlib.Path(filepath).unlink()

def mock_iter(dataset):
    """Mock iterator that yields items from a list."""
    for item in dataset:
        yield item
class TestCategoricalJacobian:
    """Test the categorical_jacobian function."""

    @patch("prxteinmpnn.run.prep.prep_protein_stream_and_model")
    def test_in_memory_no_averaging(self, mock_prep, mock_protein_batch):
        """Test in-memory jacobian calculation without averaging."""
        mock_prep.return_value = (, {})
        spec = JacobianSpecification(inputs="dummy.pdb", backbone_noise=[0.1, 0.2])

        result = categorical_jacobian(spec=spec)
        

        assert "categorical_jacobians" in result
        assert isinstance(result["categorical_jacobians"], jax.Array)
        # Shape: (batch, noise, L, C, L, C)
        assert result["categorical_jacobians"].shape == (1, 2, 10, 21, 10, 21)

    @patch("prxteinmpnn.prep_protein_stream_and_model")
    @patch("prxteinmpnn.make_encoding_average_conditional_logits_fn")
    def test_in_memory_with_averaging(
        self, mock_make_fns, mock_prep, mock_protein_batch, mock_encoding_pytree
    ):
        """Test in-memory jacobian calculation with encoding averaging."""
        # Setup: Two batches, each with one protein
        mock_prep.return_value = ([mock_protein_batch, mock_protein_batch], {})

        # Mock the encode and decode functions
        mock_encode_fn = lambda *args, **kwargs: mock_encoding_pytree
        mock_logits_fn = lambda *args, **kwargs: (jnp.ones((10, 21)), None, None)
        mock_make_fns.return_value = (mock_encode_fn, mock_logits_fn)

        spec = JacobianSpecification(
            inputs="dummy.pdb", backbone_noise=[0.1, 0.2], average_encodings=True
        )

        result = categorical_jacobian(spec=spec)

        assert "categorical_jacobians" in result
        assert isinstance(result["categorical_jacobians"], jax.Array)
        # Shape: (1, 1, L, C, L, C) -> averaged over batch and noise, then dims added back
        assert result["categorical_jacobians"].shape == (1, 1, 10, 21, 10, 21)

    def test_streaming_no_averaging(self, tmp_path, cif_file):
        """Test streaming jacobian calculation without averaging."""
        h5_path = tmp_path / "test_streaming.h5"
        spec = JacobianSpecification(
            inputs=[cif_file],
            output_h5_path=h5_path,
            backbone_noise=[0.1, 0.2],
            batch_size=1,
            compute_apc=True,
        )

        result = categorical_jacobian(spec=spec)

        assert "output_h5_path" in result
        assert h5_path.exists()

        with h5py.File(h5_path, "r") as f:
            spec_hash = result["spec_hash"]
            assert spec_hash in f
            group = f[spec_hash]
            assert "categorical_jacobians" in group
            assert "one_hot_sequences" in group
            assert "apc_corrected_jacobians" in group
            # Shape: (batch, noise, L, C, L, C)
            assert group["categorical_jacobians"].shape[0] == 1  # 1 sample
            assert group["categorical_jacobians"].shape[1] == 2  # 2 noise levels

    def test_streaming_with_averaging(self, tmp_path, cif_file):
        """Test streaming jacobian calculation with encoding averaging."""
        h5_path = tmp_path / "test_streaming_avg.h5"
        spec = JacobianSpecification(
            inputs=[cif_file],
            output_h5_path=h5_path,
            backbone_noise=[0.1, 0.2],
            batch_size=1,
            average_encodings=True,
            compute_apc=True,
        )

        result = categorical_jacobian(spec=spec)

        assert "output_h5_path" in result
        assert h5_path.exists()

        with h5py.File(h5_path, "r") as f:
            spec_hash = result["spec_hash"]
            assert spec_hash in f
            group = f[spec_hash]
            assert "categorical_jacobians" in group
            assert "one_hot_sequences" in group
            assert "apc_corrected_jacobians" in group
            # Shape: (1, 1, L, C, L, C)
            assert group["categorical_jacobians"].shape[0] == 1
            assert group["categorical_jacobians"].shape[1] == 1
