
"""Unit tests for the prxteinmpnn.io.parsing submodule."""

import pathlib
import tempfile
from io import StringIO

import h5py
import mdtraj as md
import numpy as np
import pytest
from biotite.structure import Atom, AtomArray, AtomArrayStack, array as strucarray
from chex import assert_trees_all_close

from prxteinmpnn.io.parsing.mappings import (
    _check_if_file_empty,
    af_to_mpnn,
    atom_names_to_index,
    mpnn_to_af,
    protein_sequence_to_string,
    residue_names_to_aatype,
    string_key_to_index,
    string_to_protein_sequence,
)
from prxteinmpnn.utils.data_structures import ProteinTuple
from prxteinmpnn.utils.residue_constants import resname_to_idx, restype_order, unk_restype_index
from conftest import PDB_1UBQ_STRING

def test_af_to_mpnn():
    """Test conversion from AlphaFold to ProteinMPNN alphabet."""
    af_sequence = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    mpnn_sequence = af_to_mpnn(af_sequence)
    assert mpnn_sequence.tolist() == [
        0,
        14,
        11,
        2,
        1,
        13,
        3,
        5,
        6,
        7,
        9,
        8,
        10,
        4,
        12,
        15,
        16,
        18,
        19,
        17,
        20,
    ]


def test_mpnn_to_af():
    """Test conversion from ProteinMPNN to AlphaFold alphabet."""
    mpnn_sequence = np.array(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    )
    af_sequence = mpnn_to_af(mpnn_sequence)
    print(af_sequence)
    assert af_sequence.tolist() == [
      0,
      4,
      3,
      6,
      13,
      7,
      8,
      9,
      11,
      10,
      12,
      2,
      14,
      5,
      1,
      15,
      16,
      19,
      17,
      18,
      20,
    ]



def test_string_key_to_index():
    """Test the string_key_to_index function."""
    key_map = {"A": 0, "B": 1, "C": 2}
    keys = np.array(["A", "C", "D"])
    indices = string_key_to_index(keys, key_map, unk_index=3)
    assert np.array_equal(indices, np.array([0, 2, 3]))


def test_string_to_protein_sequence():
    """Test the string_to_protein_sequence function."""
    sequence = "ARND"
    protein_seq = string_to_protein_sequence(sequence)
    expected = af_to_mpnn(np.array([0, 1, 2, 3]))
    assert np.array_equal(protein_seq, expected)


def test_protein_sequence_to_string():
    """Test the protein_sequence_to_string function."""
    protein_seq = af_to_mpnn(np.array([0, 1, 2, 3]))
    sequence = protein_sequence_to_string(protein_seq)
    assert sequence == "ARND"


def test_residue_names_to_aatype():
    """Test the residue_names_to_aatype function."""
    residue_names = np.array(["ALA", "ARG", "ASN", "ASP"])
    aatype = residue_names_to_aatype(residue_names)
    expected = af_to_mpnn(np.array([0, 1, 2, 3]))
    assert np.array_equal(aatype, expected)


def test_atom_names_to_index():
    """Test the atom_names_to_index function."""
    atom_names = np.array(["N", "CA", "C", "O", "CB"])
    indices = atom_names_to_index(atom_names)
    assert np.array_equal(indices, np.array([0, 1, 2, 4, 3]))



def test_check_if_file_empty(tmp_path):
    """Test the _check_if_file_empty utility."""
    empty_file = tmp_path / "empty.txt"
    empty_file.touch()
    assert _check_if_file_empty(str(empty_file))

    non_empty_file = tmp_path / "non_empty.txt"
    non_empty_file.write_text("hello")
    assert not _check_if_file_empty(str(non_empty_file))

    assert _check_if_file_empty("non_existent_file.txt")

