import pathlib
from io import StringIO
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
import pytest
from biotite import structure
from biotite.structure import Atom, AtomArray, AtomArrayStack, array

from prxteinmpnn.io.parsing import (
  _check_if_file_empty,
  _fill_in_cb_coordinates,
  atom_names_to_index,
  parse_input,
  process_atom_array,
  residue_names_to_aatype,
  string_key_to_index,
  string_to_protein_sequence,
)
from prxteinmpnn.utils.data_structures import ProteinTuple
from prxteinmpnn.utils.residue_constants import atom_order


# --- Fixtures ---
@pytest.fixture
def sample_pdb_string() -> str:
  """A simple PDB string for testing."""
  return """
ATOM      1  N   ALA A   1      27.230  36.324  24.562  1.00  0.00           N
ATOM      2  CA  ALA A   1      28.150  35.200  24.340  1.00  0.00           C
ATOM      3  C   ALA A   1      27.460  34.050  23.630  1.00  0.00           C
ATOM      4  O   ALA A   1      26.250  34.040  23.440  1.00  0.00           O
ATOM      5  CB  ALA A   1      29.300  35.550  25.250  1.00  0.00           C
ATOM      6  N   GLY B   2      28.200  33.050  23.250  1.00  0.00           N
ATOM      7  CA  GLY B   2      27.600  31.860  22.650  1.00  0.00           C
ATOM      8  C   GLY B   2      28.560  31.400  21.600  1.00  0.00           C
ATOM      9  O   GLY B   2      29.660  31.780  21.400  1.00  0.00           O
"""


@pytest.fixture
def sample_atom_array() -> AtomArray:
  """A simple AtomArray for testing."""
  atoms = [
    Atom(coord=[27.23, 36.32, 24.56], atom_name="N", res_name="ALA", chain_id="A", res_id=1),
    Atom(coord=[28.15, 35.20, 24.34], atom_name="CA", res_name="ALA", chain_id="A", res_id=1),
    Atom(coord=[27.46, 34.05, 23.63], atom_name="C", res_name="ALA", chain_id="A", res_id=1),
    Atom(coord=[29.30, 35.55, 25.25], atom_name="CB", res_name="ALA", chain_id="A", res_id=1),
    Atom(coord=[28.20, 33.05, 23.25], atom_name="N", res_name="GLY", chain_id="B", res_id=2),
    Atom(coord=[27.60, 31.86, 22.65], atom_name="CA", res_name="GLY", chain_id="B", res_id=2),
    Atom(coord=[28.56, 31.40, 21.60], atom_name="C", res_name="GLY", chain_id="B", res_id=2),
  ]
  return array(atoms)


# --- Test Helper Functions ---
def test_check_if_file_empty(tmp_path: pathlib.Path) -> None:
  """Test _check_if_file_empty with various file states."""
  empty_file = tmp_path / "empty.txt"
  empty_file.touch()
  assert _check_if_file_empty(str(empty_file)) is True

  non_empty_file = tmp_path / "non_empty.txt"
  non_empty_file.write_text("hello")
  assert _check_if_file_empty(str(non_empty_file)) is False

  non_existent_file = tmp_path / "i_dont_exist.txt"
  assert _check_if_file_empty(str(non_existent_file)) is True


# --- Test String/Key Conversion Functions ---
def test_string_key_to_index() -> None:
  """Test string_key_to_index for correct key-to-index conversion."""
  key_map = {"A": 0, "C": 1, "D": 2}
  string_keys = np.array(["A", "D", "X", "C"])

  # Test with default unknown index
  result = string_key_to_index(string_keys, key_map)
  expected = jnp.array([0, 2, 3, 1])
  assert jnp.array_equal(result, expected)

  # Test with custom unknown index
  result_custom_unk = string_key_to_index(string_keys, key_map, unk_index=99)
  expected_custom_unk = jnp.array([0, 2, 99, 1])
  assert jnp.array_equal(result_custom_unk, expected_custom_unk)


def test_string_to_protein_sequence() -> None:
  """Test conversion from a string sequence to ProteinSequence."""
  sequence = "AGX"
  result = string_to_protein_sequence(sequence)
  # AF: A=0, G=7, X=20 -> MPNN: 0, 5, 20
  expected = jnp.array([0, 5, 20], dtype=jnp.int8)
  assert jnp.array_equal(result, expected)


def test_residue_names_to_aatype() -> None:
  """Test conversion of 3-letter residue names to aatype."""
  residue_names = np.array(["ALA", "GLY", "INVALID"])
  result = residue_names_to_aatype(residue_names)
  # AF: ALA=0, GLY=7, INVALID=20 -> MPNN: 0, 5, 20
  expected = jnp.array([0, 5, 20], dtype=jnp.int8)
  assert jnp.array_equal(result, expected)


def test_atom_names_to_index() -> None:
  """Test conversion of atom names to indices."""
  atom_names = np.array(["N", "CA", "INVALID", "C"])
  result = atom_names_to_index(atom_names)
  expected = jnp.array([atom_order["N"], atom_order["CA"], -1, atom_order["C"]], dtype=jnp.int8)
  assert jnp.array_equal(result, expected)


# --- Test AtomArray/Structure Processing ---
def test_process_atom_array(sample_atom_array: AtomArray) -> None:
  """Test processing a valid AtomArray."""
  protein = process_atom_array(sample_atom_array)

  assert isinstance(protein, ProteinTuple)
  assert protein.coordinates.shape == (2, 37, 3)
  assert protein.aatype.shape == (2,)
  assert protein.atom_mask.shape == (2, 37)
  assert protein.residue_index.shape == (2,)
  assert protein.chain_index.shape == (2,)

  # Check aatype (AF: ALA=0, GLY=7 -> MPNN: 0, 5)
  assert jnp.array_equal(protein.aatype, jnp.array([0, 5], dtype=jnp.int8))

  # Check chain_index (A=0, B=1)
  assert jnp.array_equal(protein.chain_index, jnp.array([0, 1]))

  # Check coordinates for a known atom
  ala_ca_coords = protein.coordinates[0, atom_order["CA"]]
  assert jnp.allclose(ala_ca_coords, jnp.array([28.15, 35.20, 24.34]))

  # Check mask for a known atom and an unknown atom
  assert protein.atom_mask[0, atom_order["CA"]]
  assert not protein.atom_mask[0, atom_order["OXT"]]


def test_process_atom_array_with_chain_id(sample_atom_array: AtomArray) -> None:
  """Test processing with chain_id filtering."""
  protein = process_atom_array(sample_atom_array, chain_id="A")
  assert protein.aatype.shape == (1,)
  assert jnp.array_equal(protein.chain_index, jnp.array([0]))


def test_process_atom_array_empty() -> None:
  """Test processing an empty AtomArray."""
  with pytest.raises(ValueError, match="AtomArray is empty."):
    process_atom_array(AtomArray(0))


def test_process_atom_array_no_atom_names(sample_atom_array: AtomArray, monkeypatch) -> None:
  """Test processing an AtomArray with no atom names."""
  monkeypatch.setattr(structure.AtomArray, "atom_name", None, raising=False)
  with pytest.raises(ValueError, match="Atom names are not available in the structure."):
    process_atom_array(sample_atom_array)




@pytest.mark.asyncio
async def test_parse_input_single_model(sample_atom_array: AtomArray) -> None:
  """Test parse_input with a source returning a single AtomArray."""
  with patch("biotite.structure.io.load_structure", return_value=sample_atom_array):
    proteins = [p async for p in parse_input(StringIO("..."))]
    assert len(proteins) == 1
    assert isinstance(proteins[0][0], ProteinTuple)


@pytest.mark.asyncio
async def test_parse_input_multi_model(sample_atom_array: AtomArray) -> None:
  """Test parse_input with a source returning an AtomArrayStack."""
  stack = structure.stack([sample_atom_array, sample_atom_array])
  with patch("biotite.structure.io.load_structure", return_value=stack):
    proteins = [p async for p in parse_input(StringIO("..."))]
    assert len(proteins) == 2
    assert isinstance(proteins[0][0], ProteinTuple)


@pytest.mark.asyncio
async def test_parse_input_failure() -> None:
  """Test parse_input when loading fails."""
  with patch("biotite.structure.io.load_structure", side_effect=Exception("mock error")):
    with pytest.warns(UserWarning, match="Failed to parse structure from source"):
      with pytest.raises(RuntimeError):
        _ = [p async for p in parse_input(StringIO("..."))]


@pytest.mark.asyncio
async def test_parse_input_no_proteins() -> None:
  """Test parse_input when no valid structures are parsed."""
  with patch("biotite.structure.io.load_structure", return_value=AtomArrayStack(0, 0)):
    proteins = [p async for p in parse_input(StringIO("..."))]
    assert not proteins