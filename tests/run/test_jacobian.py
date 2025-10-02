"""Tests for the core user interface module."""
import pathlib
import tempfile
from unittest.mock import Mock, patch

import jax
import jax.numpy as jnp
import pytest
import h5py

from prxteinmpnn.run import (
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
_atom_site.pdbx_PDB_model_num
_atom_site.pdbx_PDB_ins_code
ATOM 1 N N GLY A 1 -6.778 -1.424 4.200 1.00 0.00 1 ?
"""
        )
        filepath = f.name
    yield filepath
    pathlib.Path(filepath).unlink()

class TestCategoricalJacobian:
    """Test the categorical_jacobian function."""

    def test_categorical_jacobian_basic(self, mock_protein_batch: ProteinBatch) -> None:
        """Test basic categorical jacobian calculation."""
        with patch(
            "prxteinmpnn.io.loaders.create_protein_dataset"
        ) as mock_create_dataset, patch(
            "prxteinmpnn.run.prep.get_mpnn_model"
        ) as mock_get_model, patch(
            "prxteinmpnn.run.jacobian.make_conditional_logits_fn"
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

    def test_categorical_jacobian_streaming(self, tmp_path, cif_file) -> None:
        """Test that categorical_jacobian correctly creates and populates an HDF5 file."""
        h5_path = tmp_path / "test_streaming.h5"

        config = JacobianSpecification(
            inputs=[cif_file],
            output_h5_path=h5_path,
            backbone_noise=[0.1],
            cache_path=tmp_path / "cache.h5",
        )
        result = categorical_jacobian(spec=config)

        assert "output_h5_path" in result
        assert h5_path.exists()

        with h5py.File(h5_path, "r") as f:
            assert "categorical_jacobians" in f
            assert "one_hot_sequences" in f
            assert "apc_corrected_jacobians" in f
            assert f["categorical_jacobians"].shape[0] > 0
