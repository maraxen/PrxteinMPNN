"""Tests for prxteinmpnn.utils.foldcomp_utils."""

from unittest.mock import MagicMock, patch

import numpy as np

from prxteinmpnn.utils.data_structures import ProteinTuple
from prxteinmpnn.utils.foldcomp_utils import (
    _setup_foldcomp_database,
    get_protein_structures,
)


@patch("foldcomp.setup")
def test_setup_foldcomp_database_calls_setup(mock_setup: MagicMock):
    """Test _setup_foldcomp_database calls foldcomp.setup."""
    _setup_foldcomp_database.cache_clear()
    db = "afdb_swissprot_v4"
    _setup_foldcomp_database(db)
    mock_setup.assert_called_once_with(db)


@patch("foldcomp.setup")
def test_setup_foldcomp_database_cache(mock_setup: MagicMock):
    """Test _setup_foldcomp_database is cached."""
    _setup_foldcomp_database.cache_clear()
    db = "highquality_clust30"
    _setup_foldcomp_database(db)
    _setup_foldcomp_database(db)
    mock_setup.assert_called_once_with(db)


@patch("foldcomp.open")
@patch("prxteinmpnn.utils.foldcomp_utils._setup_foldcomp_database")
@patch("foldcomp.get_data")
def test_get_protein_structures_yields_structures(
    mock_get_data: MagicMock,
    mock_setup: MagicMock,
    mock_foldcomp_open: MagicMock,
):
    """Test get_protein_structures yields ProteinTuple objects."""
    protein_ids = ["P12345", "Q67890"]
    db = "afdb_rep_v4"

    mock_fcz_data = {
        "phi": np.zeros(10),
        "psi": np.zeros(10),
        "omega": np.zeros(10),
        "coordinates": np.zeros((10, 37, 3)),
        "residues": "A" * 10,
    }
    mock_get_data.return_value = mock_fcz_data

    mock_proteins_iter = iter([("P12345", b"fcz1"), ("Q67890", b"fcz2")])
    mock_foldcomp_open.return_value.__enter__.return_value = mock_proteins_iter

    result = list(get_protein_structures(protein_ids, database=db))

    mock_setup.assert_called_once_with(db)
    mock_foldcomp_open.assert_called_once_with(db, ids=protein_ids, decompress=False)
    assert mock_get_data.call_count == 2
    assert len(result) == 2
    assert all(isinstance(s, ProteinTuple) for s in result)
