"""Tests for the core user interface module."""
import pathlib
import tempfile
from unittest.mock import Mock, patch
from typing import cast, Sequence

import jax
import jax.numpy as jnp
import pytest
import h5py

from prxteinmpnn.run import (
    ScoringSpecification,
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

class TestScore:
    """Test the score function."""

    def test_score_single_structure_single_sequence(
        self, mock_protein_batch: ProteinBatch
    ) -> None:
        """Test scoring a single structure with a single sequence."""
        with patch(
            "prxteinmpnn.io.loaders.create_protein_dataset"
        ) as mock_create_dataset, patch(
            "prxteinmpnn.run.prep.get_mpnn_model"
        ) as mock_get_model, patch(
            "prxteinmpnn.run.scoring.make_score_sequence"
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

    def test_score_streaming(self, tmp_path, cif_file) -> None:
        """Test that score correctly creates and populates an HDF5 file."""
        h5_path = tmp_path / "test_scoring_streaming.h5"

        spec = ScoringSpecification(
            inputs=[cif_file],
            sequences_to_score=["TESTSEQUENCE"],
            output_h5_path=h5_path,
            backbone_noise=[0.1],
        )
        result = score(spec=spec)

        assert "output_h5_path" in result
        assert h5_path.exists()

        with h5py.File(h5_path, "r") as f:
            assert "scores" in f
            assert "logits" in f
            assert f["scores"].shape[0] > 0
            assert f["logits"].shape[0] > 0

