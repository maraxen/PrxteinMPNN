"""Tests for Biotite parsing utilities."""

import tempfile

from biotite.structure.io import load_structure

from prxteinmpnn.io.parsing.utils import (
    atom_array_dihedrals,
)


def test_atom_array_dihedrals():
    """Test the atom_array_dihedrals function."""
    with open("tests/data/1ubq.pdb") as f:
        pdb_string = f.read()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as tmp:
        tmp.write(pdb_string)
        filepath = tmp.name
    atom_array = load_structure(filepath)
    dihedrals = atom_array_dihedrals(atom_array)
    assert dihedrals is not None
