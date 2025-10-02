"""Tests for coordinate utilities."""

import numpy as np
import pytest
from chex import assert_trees_all_close
from prxteinmpnn.io.parsing.coordinates import (
    compute_cb_precise,
    extend_coordinate,
    process_coordinates,
)

def test_extend_coordinate():
    """Test the extend_coordinate function."""
    atom_a = np.array([0, 0, 0])
    atom_b = np.array([1, 0, 0])
    atom_c = np.array([1, 1, 0])
    bond_length = 1.5
    bond_angle = np.pi / 2
    dihedral_angle = np.pi / 2
    atom_d = extend_coordinate(atom_a, atom_b, atom_c, bond_length, bond_angle, dihedral_angle)
    assert_trees_all_close(atom_d, np.array([1., 1., 1.5]), atol=1e-6)


def test_compute_cb_precise():
    """Test the compute_cb_precise function."""
    n_coord = np.array([0, 0, 0])
    ca_coord = np.array([1.46, 0, 0])
    c_coord = np.array([1.46 + 1.52 * np.cos(111 * np.pi / 180), 1.52 * np.sin(111 * np.pi / 180), 0])
    cb_coord = compute_cb_precise(n_coord, ca_coord, c_coord)
    assert cb_coord.shape == (3,)
