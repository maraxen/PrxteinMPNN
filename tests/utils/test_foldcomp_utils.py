"""Tests for prxteinmpnn.utils.foldcomp_utils."""

import pytest
import jax.numpy as jnp
from unittest.mock import MagicMock, patch

from prxteinmpnn.mpnn import ModelWeights, ProteinMPNNModelVersion
from prxteinmpnn.utils.data_structures import ModelInputs, ProteinStructure
from prxteinmpnn.utils.foldcomp_utils import (
  FoldCompDatabaseEnum,
  _setup_foldcomp_database,
  get_protein_structures,
  model_from_id,
)


@pytest.fixture
def dummy_protein_structure() -> ProteinStructure:
  """Create a dummy ProteinStructure for testing.

  Returns:
    ProteinStructure: Dummy structure.
  """
  return ProteinStructure(
    coordinates=jnp.zeros((10, 37, 3)),
    aatype=jnp.zeros(10, dtype=jnp.int8),
    atom_mask=jnp.ones((10, 37)),  # Changed from 1D to 2D shape
    residue_index=jnp.arange(10),
    chain_index=jnp.zeros(10, dtype=jnp.int32),
  )


@pytest.fixture
def dummy_model_inputs() -> ModelInputs:
  """Create a dummy ModelInputs for testing.

  Returns:
    ModelInputs: Dummy inputs.

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


def test_foldcomp_database_enum_members():
  """Test FoldCompDatabaseEnum members and values."""
  assert FoldCompDatabaseEnum.ESMATLAS_FULL.value == "esmatlas"
  assert FoldCompDatabaseEnum.AFDB_H_SAPIENS.value == "afdb_h_sapiens"
  assert len(list(FoldCompDatabaseEnum)) > 0


@patch("foldcomp.setup")
def test_setup_foldcomp_database_calls_setup(mock_setup: MagicMock):
  """Test _setup_foldcomp_database calls foldcomp.setup."""
  _setup_foldcomp_database.cache_clear()
  db = FoldCompDatabaseEnum.AFDB_SWISSPROT_V4
  _setup_foldcomp_database(db)
  mock_setup.assert_called_once_with(db.value)


@patch("foldcomp.setup")
def test_setup_foldcomp_database_cache(mock_setup: MagicMock):
  """Test _setup_foldcomp_database is cached."""
  _setup_foldcomp_database.cache_clear()
  db = FoldCompDatabaseEnum.ESMATLAS_HIGH_QUALITY
  _setup_foldcomp_database(db)
  _setup_foldcomp_database(db)
  mock_setup.assert_called_once_with(db.value)


@patch("foldcomp.open")
@patch("prxteinmpnn.utils.foldcomp_utils._setup_foldcomp_database")
@patch("prxteinmpnn.utils.foldcomp_utils._from_fcz")
def test_get_protein_structures_yields_structures(
  mock_from_fcz: MagicMock,
  mock_setup: MagicMock,
  mock_foldcomp_open: MagicMock,
  dummy_protein_structure: ProteinStructure,
):
  """Test get_protein_structures yields ProteinStructure objects."""
  protein_ids = ["P12345", "Q67890"]
  db = FoldCompDatabaseEnum.AFDB_REP_V4
  mock_proteins_iter = MagicMock()
  mock_foldcomp_open.return_value.__enter__.return_value = mock_proteins_iter
  mock_from_fcz.return_value = iter([dummy_protein_structure, dummy_protein_structure])
  result = list(get_protein_structures(protein_ids, database=db))
  mock_setup.assert_called_once_with(db)
  mock_foldcomp_open.assert_called_once_with(db.value, ids=protein_ids, decompress=False)
  mock_from_fcz.assert_called_once_with(mock_proteins_iter)
  assert len(result) == 2
  assert all(isinstance(s, ProteinStructure) for s in result)


@patch("prxteinmpnn.utils.foldcomp_utils.get_mpnn_model")
@patch("prxteinmpnn.utils.foldcomp_utils.get_protein_structures")
@patch("prxteinmpnn.utils.foldcomp_utils.protein_structure_to_model_inputs")
def test_model_from_id_single_id(
  mock_to_inputs: MagicMock,
  mock_get_structures: MagicMock,
  mock_get_model: MagicMock,
  dummy_protein_structure: ProteinStructure,
  dummy_model_inputs: ModelInputs,
):
  """Test model_from_id with a single protein ID.

  Args:
    mock_to_inputs: Mock for protein_structure_to_model_inputs.
    mock_get_structures: Mock for get_protein_structures.
    mock_get_model: Mock for get_mpnn_model.
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


@patch("prxteinmpnn.utils.foldcomp_utils.get_mpnn_model")
@patch("prxteinmpnn.utils.foldcomp_utils.get_protein_structures")
@patch("prxteinmpnn.utils.foldcomp_utils.protein_structure_to_model_inputs")
def test_model_from_id_multiple_ids(
  mock_to_inputs: MagicMock,
  mock_get_structures: MagicMock,
  mock_get_model: MagicMock,
  dummy_protein_structure: ProteinStructure,
  dummy_model_inputs: ModelInputs,
):
  """Test model_from_id with multiple protein IDs."""
  protein_ids = ["P12345", "Q67890"]
  mock_get_model.return_value = {"params": "test_params"}
  mock_get_structures.return_value = iter([dummy_protein_structure, dummy_protein_structure])
  mock_to_inputs.return_value = dummy_model_inputs
  model, inputs_iter = model_from_id(protein_ids)
  inputs = list(inputs_iter)
  mock_get_model.assert_called_once()
  mock_get_structures.assert_called_once_with(protein_ids=protein_ids)
  assert model == {"params": "test_params"}
  assert len(inputs) == 2
  assert all(i == dummy_model_inputs for i in inputs)


@patch("prxteinmpnn.utils.foldcomp_utils.get_protein_structures")
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


@patch("prxteinmpnn.utils.foldcomp_utils.get_mpnn_model")
@patch("prxteinmpnn.utils.foldcomp_utils.get_protein_structures")
def test_model_from_id_custom_model(
  mock_get_structures: MagicMock,
  mock_get_model: MagicMock,
  dummy_protein_structure: ProteinStructure,
):
  """Test model_from_id with custom model weights and version.

  Args:
    mock_get_structures: Mock for get_protein_structures.
    mock_get_model: Mock for get_mpnn_model.
    dummy_protein_structure: Fixture for a dummy ProteinStructure.

  Returns:
    None

  Raises:
    AssertionError: If get_mpnn_model is not called with custom arguments.
  """
  protein_ids = ["P12345", "Q67890"]
  mock_get_structures.return_value = iter([dummy_protein_structure] * 2)  # Changed back to iterator
  mock_get_model.return_value = {"params": "test_params"}

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


@patch("prxteinmpnn.utils.foldcomp_utils.get_protein_structures")
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
