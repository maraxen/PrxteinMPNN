"""Tests for the PDB processing utilities."""

import chex
import numpy as np
import jax.numpy as jnp
import pytest
from biotite.structure import Atom, AtomArray, AtomArrayStack, array

from prxteinmpnn import io
from prxteinmpnn.utils.residue_constants import resname_to_idx, unk_restype_index
from prxteinmpnn.utils.data_structures import (
  ProteinStructure,
)
  


@pytest.fixture
def mock_atom_array() -> AtomArray:
  """Create a mock AtomArray for testing.

  Represents a simple dipeptide (Ala-Cys) with some standard atoms.

  Returns:
    A sample AtomArray object.

  """
  atoms = [
    Atom(
      coord=[1.0, 2.0, 3.0],
      chain_id="A",
      res_id=1,
      res_name="ALA",
      atom_name="N",
      element="N",
      b_factor=10.0,
    ),
    Atom(
      coord=[2.0, 3.0, 4.0],
      chain_id="A",
      res_id=1,
      res_name="ALA",
      atom_name="CA",
      element="C",
      b_factor=11.0,
    ),
    Atom(
      coord=[3.0, 4.0, 5.0],
      chain_id="A",
      res_id=2,
      res_name="CYS",
      atom_name="N",
      element="N",
      b_factor=12.0,
    ),
    Atom(
      coord=[4.0, 5.0, 6.0],
      chain_id="A",
      res_id=2,
      res_name="CYS",
      atom_name="CA",
      element="C",
      b_factor=13.0,
    ),
    # Add a non-standard atom to be ignored
    Atom(
      coord=[5.0, 6.0, 7.0],
      chain_id="A",
      res_id=2,
      res_name="CYS",
      atom_name="HOH",
      element="O",
      b_factor=14.0,
    ),
  ]
  return array(atoms)


@pytest.fixture
def mock_pdb_file_content() -> str:
  """Provide content for a mock PDB file.

  Returns:
    A string containing PDB-formatted data for two residues.
  """
  return (
    "ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00 10.00           N\n"
    "ATOM      2  CA  ALA A   1       2.000   3.000   4.000  1.00 11.00           C\n"
    "ATOM      3  N   CYS A   2       3.000   4.000   5.000  1.00 12.00           N\n"
    "ATOM      4  CA  CYS A   2       4.000   5.000   6.000  1.00 13.00           C\n"
    "ATOM      5  HOH CYS A   2       5.000   6.000   7.000  1.00 14.00           O\n"
    "TER\n"
  )


@pytest.fixture
def mock_trajectory_file_content() -> str:
  """Provide content for a mock multi-model PDB file.

  Returns:
    A string containing a two-model PDB trajectory.
  """
  model1 = (
    "MODEL        1\n"
    "ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00 10.00           N\n"
    "ATOM      2  CA  ALA A   1       2.000   3.000   4.000  1.00 11.00           C\n"
    "TER\n"
    "ENDMDL\n"
  )
  model2 = (
    "MODEL        2\n"
    "ATOM      1  N   ALA A   1       1.100   2.100   3.100  1.00 10.50           N\n"
    "ATOM      2  CA  ALA A   1       2.100   3.100   4.100  1.00 11.50           C\n"
    "TER\n"
    "ENDMDL\n"
  )
  return model1 + model2


def test_string_key_to_index():
  """Test conversion of string keys to integer indices.

  Args:
    None

  Returns:
    None

  Raises:
    AssertionError: If the output does not match the expected value.
  """
  key_map = {"A": 0, "B": 1, "C": 2}
  string_keys = np.array(["A", "C", "D", "B"], dtype="U5")
  expected = jnp.array([0, 2, 3, 1])
  result = io.string_key_to_index(string_keys, key_map)
  chex.assert_trees_all_close(result, expected)

  # Test with custom unknown index
  expected_custom_unk = jnp.array([0, 2, -1, 1])
  result_custom_unk = io.string_key_to_index(string_keys, key_map, unk_index=-1)
  chex.assert_trees_all_close(result_custom_unk, expected_custom_unk)


def test_residue_names_to_aatype():
  """Test conversion of residue names to aatype indices.

  Args:
    None

  Returns:
    None

  Raises:
    AssertionError: If the output does not match the expected value.
  """
  residue_names = np.array(["ALA", "UNKNOWN", "CYS"], dtype="U5")
  expected = jnp.array(
    [resname_to_idx["ALA"], unk_restype_index, resname_to_idx["CYS"]],
    dtype=jnp.int8,
  )
  result = io.residue_names_to_aatype(residue_names)
  chex.assert_trees_all_close(result, expected)


def test_atom_names_to_index():
  """Test conversion of atom names to atom indices.

  Args:
    None

  Returns:
    None

  Raises:
    AssertionError: If the output does not match the expected value.
  """
  atom_names = np.array(["N", "CA", "HOH", "C"], dtype="U5")
  expected = jnp.array([0, 1, -1, 2], dtype=np.int8)  # N=0, CA=1, C=2
  result = io.atom_names_to_index(atom_names)
  chex.assert_trees_all_close(result, expected)


def test_process_atom_array(mock_atom_array: AtomArray):
  """Test the processing of a biotite AtomArray.

  Args:
    mock_atom_array: A pytest fixture providing a sample AtomArray.

  Returns:
    None

  Raises:
    AssertionError: If the processed structure does not match expectations.
  """
  protein_structure = io.process_atom_array(mock_atom_array, chain_id="A")

  # Expected shapes
  num_residues = 2
  num_atoms = 37
  chex.assert_shape(protein_structure.coordinates, (num_residues, num_atoms, 3))
  chex.assert_shape(protein_structure.aatype, (num_residues,))
  chex.assert_shape(protein_structure.atom_mask, (num_residues, num_atoms))
  chex.assert_shape(protein_structure.residue_index, (num_residues,))
  chex.assert_shape(protein_structure.b_factors, (num_residues, num_atoms))

  # Expected aatype
  expected_aatype = jnp.array([resname_to_idx["ALA"], resname_to_idx["CYS"]], dtype=jnp.int8)
  chex.assert_trees_all_close(protein_structure.aatype, expected_aatype)

  # Expected residue index
  expected_res_index = jnp.array([1, 2], dtype=jnp.int32)
  chex.assert_trees_all_close(protein_structure.residue_index, expected_res_index)

  # Check coordinates, mask, and b-factors for a specific atom (ALA, CA)
  ala_ca_coord = jnp.array([2.0, 3.0, 4.0])
  chex.assert_trees_all_close(protein_structure.coordinates[0, 1, :], ala_ca_coord)
  assert protein_structure.atom_mask[0, 1] == 1.0
  assert protein_structure.b_factors[0, 1] == 11.0

  # Check that a non-standard atom was ignored
  # HOH would not have a valid index, so we check that all masks for residue 2 are 0 except N and CA
  cys_mask = protein_structure.atom_mask[1]
  assert cys_mask[0] == 1.0  # N
  assert cys_mask[1] == 1.0  # CA
  assert jnp.sum(cys_mask) == 2.0


def test_process_empty_atom_array():
  """Test processing an empty AtomArray.

  Args:
    None

  Returns:
    None

  Raises:
    ValueError: If an empty AtomArray is processed.
  """
  empty_array = AtomArray(0)
  with pytest.raises(ValueError, match="No atoms found in the structure."):
    io.process_atom_array(empty_array)


def test_from_structure_file(tmp_path, mock_pdb_file_content):
  """Test loading a ProteinStructure from a PDB file.

  Args:
    tmp_path: Pytest fixture for a temporary directory.
    mock_pdb_file_content: Pytest fixture with PDB file content.

  Returns:
    None

  Raises:
    AssertionError: If the loaded structure is incorrect.
  """
  pdb_file = tmp_path / "test.pdb"
  pdb_file.write_text(mock_pdb_file_content)

  protein_structure = io.from_structure_file(str(pdb_file), chain_id="A")

  assert protein_structure.residue_index.shape[0] == 2
  expected_aatype = jnp.array([resname_to_idx["ALA"], resname_to_idx["CYS"]], dtype=jnp.int8)
  chex.assert_trees_all_close(protein_structure.aatype, expected_aatype)


def test_from_structure_file_errors(tmp_path):
  """Test error handling in from_structure_file.

  Args:
    tmp_path: Pytest fixture for a temporary directory.

  Returns:
    None

  Raises:
    FileNotFoundError: If the PDB file does not exist.
    TypeError: If chain_id is not a string.
  """
  with pytest.raises(FileNotFoundError):
    io.from_structure_file("non_existent_file.pdb")

  pdb_file = tmp_path / "test.pdb"
  pdb_file.write_text(
    "ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00 10.00           N\n"
  )
  with pytest.raises(TypeError, match="Expected chain_id to be a string"):
    io.from_structure_file(str(pdb_file), chain_id=123)  # type: ignore[call-arg]


def test_from_trajectory(tmp_path, mock_trajectory_file_content):
  """Test loading a trajectory from a multi-model PDB file.

  Args:
    tmp_path: Pytest fixture for a temporary directory.
    mock_trajectory_file_content: Pytest fixture with trajectory content.

  Returns:
    None

  Raises:
    AssertionError: If the trajectory is not processed correctly.
  """
  traj_file = tmp_path / "traj.pdb"
  traj_file.write_text(mock_trajectory_file_content)

  frames_iterator = io.from_trajectory(str(traj_file))
  frames = list(frames_iterator)

  assert len(frames) == 2
  assert isinstance(frames[0], ProteinStructure)
  assert isinstance(frames[1], ProteinStructure)

  # Check that coordinates are different between frames
  coords_frame1 = frames[0].coordinates
  coords_frame2 = frames[1].coordinates
  assert not jnp.array_equal(coords_frame1, coords_frame2)

  # Check a specific coordinate in frame 2
  expected_coord_f2 = jnp.array([1.100, 2.100, 3.100])
  chex.assert_trees_all_close(coords_frame2[0, 0, :], expected_coord_f2, atol=1e-5)


def test_from_trajectory_errors(tmp_path, mock_pdb_file_content):
  """Test error handling in from_trajectory.

  Args:
    tmp_path: Pytest fixture for a temporary directory.
    mock_pdb_file_content: Pytest fixture with single-model PDB content.

  Returns:
    None

  Raises:
    TypeError: If a single-model PDB is passed instead of a trajectory.
    ValueError: If the trajectory file is empty.
  """
  # Test passing a single-frame file, which should raise a TypeError
  single_frame_file = tmp_path / "single.pdb"
  single_frame_file.write_text(mock_pdb_file_content)
  with pytest.raises(TypeError, match="Expected a trajectory"):
    list(io.from_trajectory(str(single_frame_file)))

  # Test with an empty file
  empty_file = tmp_path / "empty.pdb"
  empty_file.write_text("\n")
  with pytest.raises(FileNotFoundError, match="The file '.*' is empty or does not exist"):
    list(io.from_trajectory(str(empty_file)))


def test_from_string(mock_pdb_file_content):
  """Test loading a ProteinStructure from a PDB string.

  Args:
    mock_pdb_file_content: Pytest fixture with PDB file content.

  Returns:
    None

  Raises:
    AssertionError: If the loaded structure is incorrect.
  """
  protein_structure = io.from_string(mock_pdb_file_content, chain_id="A")

  assert protein_structure.residue_index.shape[0] == 2
  expected_aatype = jnp.array([resname_to_idx["ALA"], resname_to_idx["CYS"]], dtype=jnp.int8)
  chex.assert_trees_all_close(protein_structure.aatype, expected_aatype)

  # Check coordinates for ALA CA atom
  ala_ca_coord = jnp.array([2.0, 3.0, 4.0])
  chex.assert_trees_all_close(protein_structure.coordinates[0, 1, :], ala_ca_coord)
  assert protein_structure.atom_mask[0, 1] == 1.0
  assert protein_structure.b_factors[0, 1] == 11.0


def test_from_string_no_chain_filter(mock_pdb_file_content):
  """Test loading a ProteinStructure from a PDB string without chain filtering.

  Args:
    mock_pdb_file_content: Pytest fixture with PDB file content.

  Returns:
    None

  Raises:
    AssertionError: If the loaded structure is incorrect.
  """
  protein_structure = io.from_string(mock_pdb_file_content)

  assert protein_structure.residue_index.shape[0] == 2
  expected_aatype = jnp.array([resname_to_idx["ALA"], resname_to_idx["CYS"]], dtype=jnp.int8)
  chex.assert_trees_all_close(protein_structure.aatype, expected_aatype)


def test_from_string_multimodel():
  """Test loading from a multi-model PDB string should process only the first model.

  Args:
    None

  Returns:
    None

  Raises:
    AssertionError: If the structure is not processed correctly.
  """
  multimodel_pdb = (
    "MODEL        1\n"
    "ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00 10.00           N\n"
    "ATOM      2  CA  ALA A   1       2.000   3.000   4.000  1.00 11.00           C\n"
    "TER\n"
    "ENDMDL\n"
    "MODEL        2\n"
    "ATOM      1  N   ALA A   1       5.000   6.000   7.000  1.00 15.00           N\n"
    "ATOM      2  CA  ALA A   1       6.000   7.000   8.000  1.00 16.00           C\n"
    "TER\n"
    "ENDMDL\n"
  )
  
  protein_structure = io.from_string(multimodel_pdb, model=1, chain_id="A")
  
  # Should only have processed model 1
  expected_coord = jnp.array([1.0, 2.0, 3.0])
  chex.assert_trees_all_close(protein_structure.coordinates[0, 0, :], expected_coord)
  
  # Test loading model 2
  protein_structure_model2 = io.from_string(multimodel_pdb, model=2, chain_id="A")
  expected_coord_model2 = jnp.array([5.0, 6.0, 7.0])
  chex.assert_trees_all_close(protein_structure_model2.coordinates[0, 0, :], expected_coord_model2)


def test_from_string_errors():
  """Test error handling in from_string.

  Args:
    None

  Returns:
    None

  Raises:
    TypeError: If chain_id is not a string or unexpected structure type.
  """
  valid_pdb = (
    "ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00 10.00           N\n"
    "ATOM      2  CA  ALA A   1       2.000   3.000   4.000  1.00 11.00           C\n"
    "TER\n"
  )
  
  # Test invalid chain_id type
  with pytest.raises(TypeError, match="Expected chain_id to be a string"):
    io.from_string(valid_pdb, chain_id=123)  # type: ignore[call-arg]


def test_from_string_empty():
  """Test loading from an empty PDB string.

  Args:
    None

  Returns:
    None

  Raises:
    ValueError: If the PDB string is empty or contains no atoms.
  """
  empty_pdb = ""
  with pytest.raises(ValueError, match="No atoms found in the structure"):
    io.from_string(empty_pdb)

  # Test with only whitespace
  whitespace_pdb = "   \n  \t  \n"
  with pytest.raises(ValueError, match="No atoms found in the structure"):
    io.from_string(whitespace_pdb)


def test_from_string_nonexistent_chain():
  """Test loading from a PDB string with a nonexistent chain.

  Args:
    None

  Returns:
    None

  Raises:
    ValueError: If the specified chain is not found.
  """
  pdb_string = (
    "ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00 10.00           N\n"
    "ATOM      2  CA  ALA A   1       2.000   3.000   4.000  1.00 11.00           C\n"
    "TER\n"
  )
  
  # Request chain B when only chain A exists
  with pytest.raises(ValueError, match="No atoms found in the structure for chain 'B'"):
    io.from_string(pdb_string, chain_id="B")

