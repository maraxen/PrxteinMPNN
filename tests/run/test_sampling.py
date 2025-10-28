"""Tests for the core user interface module."""
import pathlib
import tempfile
from unittest.mock import Mock, patch

import h5py
import jax
import jax.numpy as jnp
import pytest

from prxteinmpnn.run import (
    SamplingSpecification,
    sample,
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
""",
        )
        filepath = f.name
    yield filepath
    pathlib.Path(filepath).unlink()


class TestSample:
    """Test the sample function."""

    def test_sample_basic(self, mock_protein_batch: ProteinBatch) -> None:
        """Test basic sampling."""
        with patch(
            "prxteinmpnn.io.loaders.create_protein_dataset",
        ) as mock_create_dataset, patch(
            "prxteinmpnn.run.prep.get_mpnn_model",
        ) as mock_get_model, patch(
            "prxteinmpnn.run.sampling.make_sample_sequences",
        ) as mock_make_sample:
            mock_create_dataset.return_value = [mock_protein_batch]
            mock_get_model.return_value = {"params": {}}
            mock_sample_fn = Mock(
                return_value=(
                    jnp.ones((1, 10)),
                    jnp.ones((1, 10, 21)),
                    jnp.ones((1, 10)),
                ),
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

    def test_sample_streaming(self, tmp_path, cif_file) -> None:
        """Test that sample correctly creates and populates an HDF5 file."""
        h5_path = tmp_path / "test_sampling_streaming.h5"

        spec = SamplingSpecification(
            inputs=[cif_file],
            output_h5_path=h5_path,
            backbone_noise=[0.1],
            cache_path=tmp_path / "cache.h5",
        )
        result = sample(spec=spec)

        assert "output_h5_path" in result
        assert h5_path.exists()

        with h5py.File(h5_path, "r") as f:
            # Check that at least one structure group exists
            structure_groups = [key for key in f if key.startswith("structure_")]
            assert len(structure_groups) > 0, "No structure groups found in HDF5 file"

            # Check the first structure group
            first_group = f[structure_groups[0]]
            assert "sequences" in first_group
            assert "logits" in first_group

            # Ensure all dimensions are properly set (not zero)
            sequences = first_group["sequences"]
            logits = first_group["logits"]

            assert all(dim > 0 for dim in sequences.shape), (
                f"Sequences shape has zero dimensions: {sequences.shape}"
            )
            assert all(dim > 0 for dim in logits.shape), (
                f"Logits shape has zero dimensions: {logits.shape}"
            )

            # Verify metadata attributes
            assert "structure_index" in first_group.attrs
            assert "num_samples" in first_group.attrs
            assert "num_noise_levels" in first_group.attrs
            assert "sequence_length" in first_group.attrs

