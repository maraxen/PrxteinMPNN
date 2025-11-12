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

def test_parse_pqr_insertion_codes(tmp_path):
    """Test parsing PQR file with residue insertion codes (e.g., '52A')."""
    pqr_with_insertion = tmp_path / "insertion.pqr"
    # Create a minimal PQR with insertion codes
    pqr_content = """\
ATOM      1  N   ALA A  50      10.000  20.000  30.000  -0.500   1.850
ATOM      2  CA  ALA A  50      11.000  21.000  31.000   0.100   1.700
ATOM      3  N   ALA A  52      12.000  22.000  32.000  -0.500   1.850
ATOM      4  CA  ALA A  52      13.000  23.000  33.000   0.100   1.700
ATOM      5  N   ALA A  52A     14.000  24.000  34.000  -0.500   1.850
ATOM      6  CA  ALA A  52A     15.000  25.000  35.000   0.100   1.700
ATOM      7  N   ALA A  52B     16.000  26.000  36.000  -0.500   1.850
ATOM      8  CA  ALA A  52B     17.000  27.000  37.000   0.100   1.700
ATOM      9  N   ALA A  53      18.000  28.000  38.000  -0.500   1.850
"""
    pqr_with_insertion.write_text(pqr_content)
    temp_path, estat_info = _parse_pqr(pqr_with_insertion)
    
    assert temp_path.exists()
    assert len(estat_info.charges) == 9
    
    # Check that residue IDs are extracted correctly (numeric part only)
    # 50, 50, 52, 52, 52, 52, 52, 52, 53
    expected_resids = np.array([50, 50, 52, 52, 52, 52, 52, 52, 53], dtype=np.int32)
    assert np.array_equal(estat_info.estat_resid, expected_resids)
    
    # Check that backbone atoms are identified correctly
    assert estat_info.estat_backbone_mask.sum() == 9  # All N and CA atoms
    
    temp_path.unlink()
