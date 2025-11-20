"""Unit tests for PQR file parsing utilities (prxteinmpnn.io.parsing.pqr).
"""

import pathlib

import numpy as np
import pytest

from prxteinmpnn.io.parsing.pqr import parse_pqr_to_processed_structure
from prxteinmpnn.io.parsing.structures import ProcessedStructure

TEST_PQR_PATH = pathlib.Path(__file__).parent.parent.parent / "data" / "1a00.pqr"

def test_parse_pqr_basic():
    """Test parsing a standard PQR file."""
    processed_structure = parse_pqr_to_processed_structure(TEST_PQR_PATH)
    assert isinstance(processed_structure, ProcessedStructure)
    assert processed_structure.charges.shape == processed_structure.radii.shape
    assert processed_structure.charges.dtype == np.float32
    assert processed_structure.radii.dtype == np.float32
    # Check that we have atoms
    assert processed_structure.atom_array.array_length() > 0

def test_parse_pqr_chain_selection():
    """Test parsing with chain selection (should only include chain A)."""
    processed_structure = parse_pqr_to_processed_structure(TEST_PQR_PATH, chain_id="A")
    # Check that all chain IDs in the atom array are 'A'
    assert np.all(processed_structure.atom_array.chain_id == "A")

def test_parse_pqr_empty(tmp_path):
    """Test parsing an empty PQR file (should raise ValueError)."""
    empty_pqr = tmp_path / "empty.pqr"
    empty_pqr.write_text("")
    with pytest.raises(ValueError, match="No atoms found"):
        parse_pqr_to_processed_structure(empty_pqr)

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
    processed_structure = parse_pqr_to_processed_structure(pqr_with_insertion)

    assert len(processed_structure.charges) == 9

    # Check that residue IDs are extracted correctly (numeric part only)
    # 50, 50, 52, 52, 52, 52, 52, 52, 53
    expected_resids = np.array([50, 50, 52, 52, 52, 52, 52, 52, 53], dtype=int)
    assert np.array_equal(processed_structure.r_indices, expected_resids)
