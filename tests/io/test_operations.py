"""Tests for Grain operations for processing protein structures."""

import io
from unittest.mock import Mock, patch

import jax.numpy as jnp
import numpy as np
import pytest
import requests

from prxteinmpnn.io.operations import (
    ParseStructure,
    _fetch_pdb,
    pad_and_collate_proteins,
)
from prxteinmpnn.utils.data_structures import Protein, ProteinBatch, ProteinTuple


@patch("requests.get")
def test_fetch_pdb_success(mock_get: Mock) -> None:
    """Test successful fetching of a PDB file."""
    mock_response = Mock()
    mock_response.text = "PDB_CONTENT"
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    content = _fetch_pdb("1abc")
    assert content == "PDB_CONTENT"
    mock_get.assert_called_once_with(
        "https://files.rcsb.org/download/1abc.pdb", timeout=60
    )


@patch("requests.get")
def test_fetch_pdb_http_error(mock_get: Mock) -> None:
    """Test handling of an HTTP error when fetching a PDB file."""
    mock_response = Mock()
    mock_response.raise_for_status.side_effect = requests.HTTPError
    mock_get.return_value = mock_response

    with pytest.raises(requests.HTTPError):
        _fetch_pdb("1abc")


class TestParseStructure:
    """Tests for the ParseStructure Grain operation."""

    @pytest.fixture
    def mock_protein_tuple(self) -> ProteinTuple:
        """Fixture for a mock ProteinTuple."""
        return ProteinTuple(
            coordinates=np.ones((10, 37, 3)),
            aatype=np.ones(10, dtype=np.int8),
            atom_mask=np.ones((10, 37)),
            residue_index=np.arange(10),
            nitrogen_mask=np.ones((10, 37)),
            chain_index=np.zeros(10),
            dihedrals=None,
            source=None,
            mapping=None,
        )

    def test_map_file_path(self, mock_protein_tuple: ProteinTuple) -> None:
        """Test parsing from a file path."""
        op = ParseStructure()
        with patch("prxteinmpnn.io.operations.parse_input") as mock_parse:
            mock_parse.return_value = [mock_protein_tuple]
            result = op.flat_map(("file_path", "test.pdb"))
            assert result == [mock_protein_tuple]
            mock_parse.assert_called_once_with("test.pdb")

    def test_map_pdb_id(self, mock_protein_tuple: ProteinTuple) -> None:
        """Test parsing from a PDB ID."""
        op = ParseStructure()
        with patch("prxteinmpnn.io.operations._fetch_pdb") as mock_fetch, patch(
            "prxteinmpnn.io.operations.parse_input"
        ) as mock_parse:
            mock_fetch.return_value = "PDB_CONTENT"
            mock_parse.return_value = [mock_protein_tuple]

            result = op.flat_map(("pdb_id", "1abc"))

            assert result == [mock_protein_tuple]
            mock_fetch.assert_called_once_with("1abc")
            mock_parse.assert_called_once()
            # Check that the first argument to parse_input is a StringIO object
            assert isinstance(mock_parse.call_args[0][0], io.StringIO)

    def test_map_string_io(self, mock_protein_tuple: ProteinTuple) -> None:
        """Test parsing from a StringIO object."""
        op = ParseStructure()
        string_io = io.StringIO("PDB_CONTENT")
        with patch("prxteinmpnn.io.operations.parse_input") as mock_parse:
            mock_parse.return_value = [mock_protein_tuple]
            result = op.flat_map(("string_io", string_io))
            assert result == [mock_protein_tuple]
            mock_parse.assert_called_once_with(string_io)

    def test_map_foldcomp_ids(self, mock_protein_tuple: ProteinTuple) -> None:
        """Test parsing from FoldComp IDs."""
        op = ParseStructure()
        foldcomp_ids = ["ID1", "ID2"]
        with patch(
            "prxteinmpnn.io.operations.get_protein_structures"
        ) as mock_get_struct:
            mock_get_struct.return_value = [mock_protein_tuple]
            result = op.flat_map(("foldcomp_ids", foldcomp_ids))
            assert result == [mock_protein_tuple]
            mock_get_struct.assert_called_once_with(foldcomp_ids)

    def test_map_parsing_failure_warning(self) -> None:
        """Test that a warning is issued on parsing failure."""
        op = ParseStructure()
        with patch(
            "prxteinmpnn.io.operations.parse_input", side_effect=Exception("Parse error")
        ), pytest.warns(UserWarning, match="Failed to parse"):
            result = op.flat_map(("file_path", "bad.pdb"))



class TestPadAndCollate:
    """Tests for the pad_and_collate_proteins function."""

    def test_pad_and_collate(self) -> None:
        """Test correct batching and padding of proteins."""
        p1_tuple = ProteinTuple(
            coordinates=np.ones((10, 37, 3)),
            aatype=np.ones(10, dtype=np.int8),
            atom_mask=np.ones((10, 37)),
            residue_index=np.arange(10),
            chain_index=np.zeros(10, dtype=np.int32),
            nitrogen_mask=np.ones((10, 37)),
            dihedrals=None,
            source=None,
            mapping=None,
        )
        p2_tuple = ProteinTuple(
            coordinates=np.ones((15, 37, 3)),
            aatype=np.ones(15, dtype=np.int8),
            atom_mask=np.ones((15, 37)),
            residue_index=np.arange(15),
            nitrogen_mask=np.ones((15, 37)),
            chain_index=np.zeros(15, dtype=np.int32),
            dihedrals=None,
            source=None,
            mapping=None,
        )

        elements: list[ProteinTuple] = [p1_tuple, p2_tuple]
        batch: Protein = pad_and_collate_proteins(elements)

        assert isinstance(batch, Protein)
        assert batch.coordinates.shape == (2, 15, 37, 3)
        assert batch.aatype.shape == (2, 15)
        assert batch.atom_mask.shape == (2, 15, 37)
        assert batch.residue_index.shape == (2, 15)
        assert batch.chain_index.shape == (2, 15)

        # Check that the first protein is padded correctly
        assert jnp.all(batch.coordinates[0, 10:] == 0)
        assert jnp.all(batch.aatype[0, 10:] == 0)

    def test_collate_empty_list_raises_error(self) -> None:
        """Test that collating an empty list raises a ValueError."""
        with pytest.raises(ValueError, match="Cannot collate an empty list"):
            pad_and_collate_proteins([])
