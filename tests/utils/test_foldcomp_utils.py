"""Tests for prxteinmpnn.utils.foldcomp."""

import unittest.mock
from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import pytest

from prxteinmpnn.mpnn import ModelWeights, ProteinMPNNModelVersion
from prxteinmpnn.utils.data_structures import ModelInputs, ProteinStructure
from prxteinmpnn.utils.foldcomp_utils import (
  FoldCompDatabaseEnum,
  _get_protein_structures_from_database,
  get_protein_structures,
  model_from_id,
  _setup_foldcomp_database,
)


@pytest.fixture
def dummy_protein_structure() -> ProteinStructure:
  """Create a dummy ProteinStructure for testing.

  Returns:
    A ProteinStructure object with minimal valid data.

  """
  return ProteinStructure(
    coordinates=jnp.zeros((10, 37, 3)),
    aatype=jnp.zeros(10, dtype=jnp.int8),
    atom_mask=jnp.ones((10, 37)),
    residue_index=jnp.arange(10),
    chain_index=jnp.zeros(10, dtype=jnp.int32),
    b_factors=jnp.zeros((10, 37)),
  )


@pytest.fixture
def dummy_model_inputs() -> ModelInputs:
  """Create a dummy ModelInputs for testing.

  Returns:
    A ModelInputs object with minimal valid data.

  """
  return ModelInputs(
    structure_coordinates=jnp.zeros((10, 37, 3)),
    sequence=jnp.zeros(10, dtype=jnp.int8),
    mask=jnp.ones((10, 37)),
    residue_index=jnp.arange(10),
    chain_index=jnp.zeros(10, dtype=jnp.int32),
    lengths=jnp.array([10]),
    bias=jnp.zeros((10, 20)),
  )


def test_foldcomp_database_enum():
  """Test that FoldCompDatabase enum has expected members and values.

  Args:
    None

  Returns:
    None

  Raises:
    AssertionError: If enum members or values are not as expected.

  """
  assert FoldCompDatabaseEnum.ESMATLAS_FULL.value == "esmatlas"
  assert FoldCompDatabaseEnum.AFDB_H_SAPIENS.value == "afdb_h_sapiens"
  assert len(list(FoldCompDatabaseEnum)) > 0


@patch("foldcomp.setup")
def test_setup_foldcomp_database_calls_setup(mock_setup: MagicMock):
  """Test that setup_foldcomp_database calls foldcomp.setup correctly.

  Args:
    mock_setup: A mock object for the foldcomp.setup function.

  Returns:
    None

  Raises:
    AssertionError: If foldcomp.setup is not called with the correct arguments.

  """
  # Clear cache to ensure the function call is not skipped
  _setup_foldcomp_database.cache_clear()
  database = FoldCompDatabaseEnum.AFDB_SWISSPROT_V4
  _setup_foldcomp_database(database)
  mock_setup.assert_called_once_with(database.value)


@patch("foldcomp.setup")
def test_setup_foldcomp_database_is_cached(mock_setup: MagicMock):
  """Test that setup_foldcomp_database is cached.

  Args:
    mock_setup: A mock object for the foldcomp.setup function.

  Returns:
    None

  Raises:
    AssertionError: If foldcomp.setup is called more than once for the same database.

  """
  # Clear cache before the test
  _setup_foldcomp_database.cache_clear()
  database = FoldCompDatabaseEnum.ESMATLAS_HIGH_QUALITY

  # Call the function multiple times
  _setup_foldcomp_database(database)
  _setup_foldcomp_database(database)
  _setup_foldcomp_database(database)

  # Assert that the underlying foldcomp.setup was only called once
  mock_setup.assert_called_once_with(database.value)


@patch("prxteinmpnn.utils.foldcomp.from_string")
def test_get_protein_structures_from_database(
  mock_from_string: MagicMock,
  dummy_protein_structure: ProteinStructure,
):
  """Test the internal _get_protein_structures_from_database helper.

  Args:
    mock_from_string: A mock for the from_string function.
    dummy_protein_structure: A fixture providing a dummy ProteinStructure.

  Returns:
    None

  Raises:
    AssertionError: If the helper function does not yield the expected structures.

  """
  mock_from_string.return_value = dummy_protein_structure
  proteins_dict = {"P12345": "PDB_STRING_1", "Q67890": "PDB_STRING_2"}

  structures_iterator = _get_protein_structures_from_database(proteins_dict)
  structures_list = list(structures_iterator)

  assert len(structures_list) == 2
  assert all(isinstance(s, ProteinStructure) for s in structures_list)
  mock_from_string.assert_has_calls([
    unittest.mock.call("PDB_STRING_1"),
    unittest.mock.call("PDB_STRING_2"),
  ])


@patch("prxteinmpnn.utils.foldcomp.setup_foldcomp_database")
@patch("foldcomp.open")
@patch("prxteinmpnn.utils.foldcomp._get_protein_structures_from_database")
def test_get_protein_structures_happy_path(
  mock_get_from_db: MagicMock,
  mock_foldcomp_open: MagicMock,
  mock_setup: MagicMock,
  dummy_protein_structure: ProteinStructure,
):
  """Test get_protein_structures for the successful retrieval of structures.

  Args:
    mock_get_from_db: Mock for the internal helper function.
    mock_foldcomp_open: Mock for the foldcomp.open context manager.
    mock_setup: Mock for the database setup function.
    dummy_protein_structure: A fixture for a dummy ProteinStructure.

  Returns:
    None

  Raises:
    AssertionError: If dependencies are not called correctly or output is wrong.

  """
  protein_ids = ["P12345", "Q67890"]
  database = FoldCompDatabaseEnum.AFDB_REP_V4
  mock_proteins_dict = {"P12345": "pdb1", "Q67890": "pdb2"}
  mock_foldcomp_open.return_value.__enter__.return_value = mock_proteins_dict
  mock_get_from_db.return_value = iter([dummy_protein_structure] * 2)

  result_iterator = get_protein_structures(protein_ids, database=database)
  result_list = list(result_iterator)

  mock_setup.assert_called_once_with(database)
  mock_foldcomp_open.assert_called_once_with(database.value, ids=protein_ids)
  mock_get_from_db.assert_called_once_with(mock_proteins_dict)
  assert len(result_list) == 2
  assert all(isinstance(s, ProteinStructure) for s in result_list)


@patch("prxteinmpnn.utils.foldcomp.get_protein_structures")
@patch("prxteinmpnn.utils.foldcomp.get_mpnn_model")
@patch("prxteinmpnn.utils.foldcomp.protein_structure_to_model_inputs")
def test_model_from_id_single_id(
  mock_to_inputs: MagicMock,
  mock_get_model: MagicMock,
  mock_get_structures: MagicMock,
  dummy_protein_structure: ProteinStructure,
  dummy_model_inputs: ModelInputs,
):
  """Test model_from_id with a single protein ID string.

  Args:
    mock_to_inputs: Mock for protein_structure_to_model_inputs.
    mock_get_model: Mock for get_mpnn_model.
    mock_get_structures: Mock for get_protein_structures.
    dummy_protein_structure: Fixture for a dummy ProteinStructure.
    dummy_model_inputs: Fixture for a dummy ModelInputs.

  Returns:
    None

  Raises:
    AssertionError: If dependencies are not called correctly or output is wrong.

  """
  protein_id = "P12345"
  mock_model = {"params": "test_params"}
  mock_get_model.return_value = mock_model
  mock_get_structures.return_value = iter([dummy_protein_structure])
  mock_to_inputs.return_value = dummy_model_inputs

  model, inputs_iterator = model_from_id(protein_id)
  inputs_list = list(inputs_iterator)

  mock_get_model.assert_called_once_with(
    model_version=ProteinMPNNModelVersion.V_48_002,
    model_weights=ModelWeights.DEFAULT,
  )
  mock_get_structures.assert_called_once_with(protein_ids=[protein_id])
  mock_to_inputs.assert_called_once_with(dummy_protein_structure)
  assert model == mock_model
  assert len(inputs_list) == 1
  assert inputs_list[0] == dummy_model_inputs


@patch("prxteinmpnn.utils.foldcomp.get_protein_structures")
@patch("prxteinmpnn.utils.foldcomp.get_mpnn_model")
def test_model_from_id_custom_model(
  mock_get_model: MagicMock,
  mock_get_structures: MagicMock,
  dummy_protein_structure: ProteinStructure,
):
  """Test model_from_id with custom model weights and version.

  Args:
    mock_get_model: Mock for get_mpnn_model.
    mock_get_structures: Mock for get_protein_structures.
    dummy_protein_structure: Fixture for a dummy ProteinStructure.

  Returns:
    None

  Raises:
    AssertionError: If get_mpnn_model is not called with custom arguments.

  """
  protein_ids = ["P12345", "Q67890"]
  mock_get_structures.return_value = iter([dummy_protein_structure] * 2)

  model_version = ProteinMPNNModelVersion.V_48_020
  model_weights = ModelWeights.SOLUBLE

  model_from_id(
    protein_ids,
    model_weights=model_weights,
    model_version=model_version,
  )

  mock_get_model.assert_called_once_with(
    model_version=model_version,
    model_weights=model_weights,
  )


@patch("prxteinmpnn.utils.foldcomp.get_protein_structures")
def test_model_from_id_no_structures_found(mock_get_structures: MagicMock):
  """Test that model_from_id raises ValueError when no structures are found.

  Args:
    mock_get_structures: Mock for get_protein_structures.

  Returns:
    None

  Raises:
    AssertionError: If ValueError is not raised as expected.

  """
  protein_ids = ["UNKNOWN1"]
  mock_get_structures.return_value = iter([])  # Empty iterator

  import re
  with pytest.raises(
    ValueError,
    match=re.escape(f"No protein structures found for IDs: {protein_ids}"),
  ):
    model_from_id(protein_ids)
    