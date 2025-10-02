"""Tests for Biotite parsing utilities."""

import numpy as np
import pytest
from biotite.structure import Atom, AtomArray, AtomArrayStack, array as strucarray
from chex import assert_trees_all_close
from io import StringIO
from biotite.structure.io import load_structure

from prxteinmpnn.io.parsing.biotite import (
    atom_array_dihedrals,
)
from conftest import pdb_file

def test_atom_array_dihedrals(pdb_file):
    """Test the atom_array_dihedrals function."""
    atom_array = load_structure(pdb_file)
    dihedrals = atom_array_dihedrals(atom_array)
    assert dihedrals is not None
