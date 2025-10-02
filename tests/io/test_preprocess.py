"""Tests for the pre-processing module."""
import h5py
import pathlib
import pytest
from io import StringIO
from unittest.mock import patch, MagicMock

import numpy as np

from prxteinmpnn.io.preprocess import preprocess_inputs_to_hdf5
from prxteinmpnn.utils.data_structures import ProteinTuple
from conftest import PDB_STRING


@pytest.fixture
def pdb_file(tmp_path):
    """Create a temporary PDB file in a unique temp directory."""
    pdb_path = tmp_path / "test.pdb"
    pdb_path.write_text(PDB_STRING)
    yield pdb_path


class TestPreprocessToHDF5:
    def test_basic_preprocessing(self, pdb_file, tmp_path):
        """Test basic functionality with a single PDB file."""
        output_h5 = tmp_path / "output.h5"
        inputs = [pdb_file]

        preprocess_inputs_to_hdf5(inputs, output_h5)

        assert output_h5.exists()
        with h5py.File(output_h5, "r") as f:
            assert f.attrs["format"] == "prxteinmpnn_preprocessed"
            assert "coordinates" in f
            assert f["coordinates"].shape[0] == 4  # PDB_STRING has 4 models

    def test_empty_input(self, tmp_path):
        """Test behavior with no input sources."""
        output_h5 = tmp_path / "output.h5"

        preprocess_inputs_to_hdf5([], output_h5)

        assert output_h5.exists()
        with h5py.File(output_h5, "r") as f:
            assert f.attrs["format"] == "prxteinmpnn_preprocessed"
            assert f.attrs["status"] == "empty"

    def test_mixed_inputs(self, pdb_file, tmp_path):
        """Test with a mix of file paths and StringIO objects."""
        output_h5 = tmp_path / "output.h5"
        inputs = [pdb_file, StringIO(PDB_STRING)]

        preprocess_inputs_to_hdf5(inputs, output_h5)

        assert output_h5.exists()
        with h5py.File(output_h5, "r") as f:
            assert "coordinates" in f
            assert f["coordinates"].shape[0] == 8  # 4 from file + 4 from string

    @patch("prxteinmpnn.io.preprocess.parse_input")
    def test_with_parse_kwargs(self, mock_parse_input, pdb_file, tmp_path):
        """Test that parse_kwargs are passed correctly."""
        output_h5 = tmp_path / "output.h5"
        inputs = [pdb_file]
        parse_kwargs = {"chain_id": "A"}

        # Mock parse_input to return a valid ProteinTuple
        mock_frame = ProteinTuple(
            coordinates=np.zeros((10, 37, 3)),
            aatype=np.zeros(10),
            atom_mask=np.zeros((10, 37)),
            residue_index=np.zeros(10),
            chain_index=np.zeros(10),
            full_coordinates=np.zeros((10, 3)),
            dihedrals=None,
            source=str(pdb_file),
            mapping=None,
        )
        mock_parse_input.return_value = [mock_frame]

        preprocess_inputs_to_hdf5(inputs, output_h5, parse_kwargs=parse_kwargs)

        mock_parse_input.assert_called_once_with(pdb_file, **parse_kwargs)