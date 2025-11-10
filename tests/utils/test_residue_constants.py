"""Unit tests for residue constants."""

from unittest.mock import patch, mock_open
import numpy as np
import pytest

from prxteinmpnn.utils import residue_constants

MOCK_STEREO_CHEMICAL_PROPS = """bonds header
N-CA ALA 1.458 0.019
CA-C ALA 1.526 0.019
-

angles header
N-CA-C ALA 111.2 2.0
-
"""
MOCK_STEREO_CHEMICAL_PROPS_EMPTY = """bonds header
-

angles header
-
"""


def test_process_lines_iter_chemical_props():
    """Test processing of stereo chemical properties from an iterator."""
    lines_iter = iter(MOCK_STEREO_CHEMICAL_PROPS.splitlines())
    residue_bonds, residue_bond_angles = residue_constants.process_lines_iter_chemical_props(
        lines_iter
    )

    assert "ALA" in residue_bonds
    assert len(residue_bonds["ALA"]) == 2
    bond = residue_bonds["ALA"][0]
    assert bond.atom1_name == "N"
    assert bond.atom2_name == "CA"
    assert bond.length == 1.458
    assert bond.stddev == 0.019

    assert "ALA" in residue_bond_angles
    assert len(residue_bond_angles["ALA"]) == 1
    angle = residue_bond_angles["ALA"][0]
    assert angle.atom1_name == "N"
    assert angle.atom2_name == "CA"
    assert angle.atom3name == "C"
    assert np.isclose(angle.angle_rad, np.deg2rad(111.2))
    assert np.isclose(angle.stddev, np.deg2rad(2.0))


def test_process_lines_iter_chemical_props_empty():
    """Test processing of empty stereo chemical properties."""
    lines_iter = iter(MOCK_STEREO_CHEMICAL_PROPS_EMPTY.splitlines())
    residue_bonds, residue_bond_angles = residue_constants.process_lines_iter_chemical_props(
        lines_iter
    )
    assert "UNK" in residue_bonds
    assert not residue_bonds["UNK"]
    assert "UNK" in residue_bond_angles
    assert not residue_bond_angles["UNK"]


@patch("pathlib.Path.open", new_callable=mock_open, read_data=MOCK_STEREO_CHEMICAL_PROPS)
def test_load_stereo_chemical_props(mock_file):
    """Test loading and processing of stereo chemical properties."""
    residue_constants.load_stereo_chemical_props.cache_clear()
    (
        residue_bonds,
        residue_virtual_bonds,
        residue_bond_angles,
    ) = residue_constants.load_stereo_chemical_props()

    assert "ALA" in residue_bonds
    assert "ALA" in residue_virtual_bonds
    assert "ALA" in residue_bond_angles

    # Check virtual bond calculation
    virtual_bond = residue_virtual_bonds["ALA"][0]
    assert virtual_bond.atom1_name == "N"
    assert virtual_bond.atom2_name == "C"
    expected_length = np.sqrt(
        1.458**2 + 1.526**2 - 2 * 1.458 * 1.526 * np.cos(np.deg2rad(111.2))
    )
    assert np.isclose(virtual_bond.length, expected_length)


@patch("prxteinmpnn.utils.residue_constants.load_stereo_chemical_props")
def test_make_atom14_dists_bounds(mock_load_stereo):
    """Test the creation of atom14 distance bounds."""
    mock_residue_bonds = {
        "ALA": [residue_constants.Bond("N", "CA", 1.458, 0.019)],
        "UNK": [],
    }
    mock_residue_virtual_bonds = {
        "ALA": [residue_constants.Bond("N", "C", 2.462, 0.038)],
        "UNK": [],
    }
    mock_residue_bond_angles = {"ALA": [], "UNK": []}

    # Add other restypes to avoid KeyError
    for resname in residue_constants.resnames:
        if resname not in mock_residue_bonds:
            mock_residue_bonds[resname] = []
            mock_residue_virtual_bonds[resname] = []
            mock_residue_bond_angles[resname] = []

    mock_load_stereo.return_value = (
        mock_residue_bonds,
        mock_residue_virtual_bonds,
        mock_residue_bond_angles,
    )

    bounds = residue_constants.make_atom14_dists_bounds()

    lower_bound = bounds["lower_bound"]
    upper_bound = bounds["upper_bound"]

    ala_idx = residue_constants.restype_order["A"]
    atom_list = residue_constants.restype_name_to_atom14_names["ALA"]
    n_idx = atom_list.index("N")
    ca_idx = atom_list.index("CA")
    c_idx = atom_list.index("C")
    o_idx = atom_list.index("O")

    # van der waals check for non-bonded atoms
    c_radius = residue_constants.van_der_waals_radius["C"]
    o_radius = residue_constants.van_der_waals_radius["O"]
    expected_lower_vdw = c_radius + o_radius - 1.5
    assert np.isclose(lower_bound[ala_idx, c_idx, o_idx], expected_lower_vdw)

    # bond check for N-CA
    expected_lower_bond = 1.458 - 15 * 0.019
    expected_upper_bond = 1.458 + 15 * 0.019
    assert np.isclose(lower_bound[ala_idx, n_idx, ca_idx], expected_lower_bond)
    assert np.isclose(upper_bound[ala_idx, n_idx, ca_idx], expected_upper_bond)

    # virtual bond check for N-C
    expected_lower_vb = 2.462 - 15 * 0.038
    expected_upper_vb = 2.462 + 15 * 0.038
    assert np.isclose(lower_bound[ala_idx, n_idx, c_idx], expected_lower_vb)
    assert np.isclose(upper_bound[ala_idx, n_idx, c_idx], expected_upper_vb)
