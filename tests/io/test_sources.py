"""Tests for the Grain data sources."""

import io
import pathlib
from unittest.mock import MagicMock, patch

import pytest

from prxteinmpnn.io.sources import MixedInputDataSource
from prxteinmpnn.utils.foldcomp_utils import FoldCompDatabase


@pytest.fixture
def mock_foldcomp_db() -> FoldCompDatabase:
    """Fixture for a mock FoldCompDatabase."""
    return MagicMock(spec=FoldCompDatabase)


def test_initialization_and_len(mock_foldcomp_db: FoldCompDatabase) -> None:
    """Test that the data source is initialized correctly and __len__ works."""
    with patch("pathlib.Path.is_file", return_value=True):
        inputs = ["file1.pdb", "AF-Q5VSL9-F1-model_v4", "1t2t"]
        source = MixedInputDataSource(inputs, foldcomp_database=mock_foldcomp_db)
        # "file1.pdb" (file_path), "AF-Q5VSL9-F1-model_v4" (foldcomp_ids), "1t2t" (pdb_id)
        # The foldcomp_ids are grouped into one item.
        assert len(source) == 3


class TestInputCategorization:
    """Tests for the categorization logic in MixedInputDataSource."""

    def test_categorize_string_io(self) -> None:
        """Test that StringIO objects are categorized correctly."""
        string_io = io.StringIO("fake pdb content")
        source = MixedInputDataSource([string_io])
        assert len(source) == 1
        assert source[0] == ("string_io", string_io)

    def test_categorize_pdb_id(self) -> None:
        """Test that PDB IDs are categorized correctly."""
        with patch("pathlib.Path.exists", return_value=False):
            source = MixedInputDataSource(["1t2t"])
            assert len(source) == 1
            assert source[0] == ("pdb_id", "1t2t")

    def test_categorize_foldcomp_id(self, mock_foldcomp_db: FoldCompDatabase) -> None:
        """Test that FoldComp IDs are categorized and grouped correctly."""
        foldcomp_ids = ["AF-Q5VSL9-F1-model_v4", "MGYP000000000001_1"]
        source = MixedInputDataSource(foldcomp_ids, foldcomp_database=mock_foldcomp_db)
        assert len(source) == 1
        assert source[0] == ("foldcomp_ids", foldcomp_ids)

    def test_categorize_file_path(self) -> None:
        """Test that file paths are categorized correctly."""
        with patch("pathlib.Path.is_file", return_value=True):
            source = MixedInputDataSource(["/path/to/file.pdb"])
            assert len(source) == 1
            assert source[0] == ("file_path", "/path/to/file.pdb")

    def test_categorize_directory(self) -> None:
        """Test that directories are recursively searched for protein files."""
        with patch("pathlib.Path.is_dir", return_value=True), patch(
            "pathlib.Path.rglob"
        ) as mock_rglob:
            mock_rglob.return_value = [
                pathlib.Path("/dir/protein1.pdb"),
                pathlib.Path("/dir/protein2.cif"),
                pathlib.Path("/dir/notes.txt"),
            ]
            # Mock is_file for the globbed paths
            with patch("pathlib.Path.is_file", return_value=True):
                source = MixedInputDataSource(["/dir"])
                assert len(source) == 2
                assert source[0] == ("file_path", "/dir/protein1.pdb")
                assert source[1] == ("file_path", "/dir/protein2.cif")

    def test_unsupported_input_type_warning(self) -> None:
        """Test that a warning is issued for unsupported input types."""
        with pytest.warns(UserWarning, match="Unsupported input type"):
            MixedInputDataSource([12345])  # type: ignore

    def test_uncategorized_input_warning(self) -> None:
        """Test that a warning is issued for uncategorized string inputs."""
        with pytest.warns(UserWarning, match="could not be categorized"):
            with patch("pathlib.Path.exists", return_value=False):
                MixedInputDataSource(["not_a_pdb_or_foldcomp_id"])

    def test_foldcomp_id_without_db_warning(self) -> None:
        """Test warning when FoldComp IDs are provided without a database."""
        with pytest.warns(UserWarning, match="FoldComp IDs were provided but no database was configured"):
            MixedInputDataSource(["AF-Q5VSL9-F1-model_v4"], foldcomp_database=None)


class TestDataSourceMethods:
    """Tests for the core methods of MixedInputDataSource."""

    def test_getitem(self) -> None:
        """Test that __getitem__ returns the correct item."""
        inputs = [io.StringIO(""), "1t2t"]
        with patch("pathlib.Path.exists", return_value=False):
            source = MixedInputDataSource(inputs)
            assert source[1] == ("pdb_id", "1t2t")

    def test_len(self) -> None:
        """Test that __len__ returns the correct number of items."""
        inputs = ["1t2t", "2t2t", "3t3t"]
        with patch("pathlib.Path.exists", return_value=False):
            source = MixedInputDataSource(inputs)
            assert len(source) == 3
