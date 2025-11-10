"""Unit tests for PQR file parsing utilities (prxteinmpnn.io.parsing.pqr).
"""

import pathlib
import numpy as np
import pytest
from prxteinmpnn.io.parsing.pqr import _parse_pqr
from prxteinmpnn.utils.data_structures import EstatInfo

TEST_PQR_PATH = pathlib.Path(__file__).parent.parent.parent / "data" / "1a00.pqr"

def test_parse_pqr_basic():
    """Test parsing a standard PQR file."""
    temp_path, estat_info = _parse_pqr(TEST_PQR_PATH)
    assert temp_path.exists()
    assert isinstance(estat_info, EstatInfo)
    assert estat_info.charges.shape == estat_info.radii.shape
    assert estat_info.charges.dtype == np.float32
    assert estat_info.radii.dtype == np.float32
    assert estat_info.estat_backbone_mask.dtype == bool
    assert estat_info.estat_resid.dtype == np.int32
    assert estat_info.estat_chain_index.dtype == np.int32
    # Check at least one backbone atom is present
    assert estat_info.estat_backbone_mask.any()
    temp_path.unlink()

def test_parse_pqr_chain_selection():
    """Test parsing with chain selection (should only include chain A)."""
    temp_path, estat_info = _parse_pqr(TEST_PQR_PATH, chain_id="A")
    assert temp_path.exists()
    # All chain indices should correspond to 'A'
    assert np.all(estat_info.estat_chain_index == ord("A"))
    temp_path.unlink()

def test_parse_pqr_empty(tmp_path):
    """Test parsing an empty PQR file (should return empty arrays)."""
    empty_pqr = tmp_path / "empty.pqr"
    empty_pqr.write_text("")
    temp_path, estat_info = _parse_pqr(empty_pqr)
    assert temp_path.exists()
    assert estat_info.charges.size == 0
    temp_path.unlink()
