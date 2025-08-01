# tests/test_mpnn.py

"""Tests for prxteinmpnn.mpnn."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import numpy as np
import pytest

from prxteinmpnn.mpnn import (
  ModelWeights,
  ProteinMPNNModelVersion,
  get_mpnn_model,
)


def test_protein_mpnn_model_version_enum():
  """Test that ProteinMPNNModelVersion enum has expected members and values.

  Args:
    None

  Returns:
    None

  Raises:
    AssertionError: If enum members or values are not as expected.

  """
  assert ProteinMPNNModelVersion.V_48_002.value == "v_48_002.pkl"
  assert ProteinMPNNModelVersion.V_48_010.value == "v_48_010.pkl"
  assert ProteinMPNNModelVersion.V_48_020.value == "v_48_020.pkl"
  assert ProteinMPNNModelVersion.V_48_030.value == "v_48_030.pkl"


def test_model_weights_enum():
  """Test that ModelWeights enum has expected members and values.

  Args:
    None

  Returns:
    None

  Raises:
    AssertionError: If enum members or values are not as expected.

  """
  assert ModelWeights.DEFAULT.value == "original"
  assert ModelWeights.SOLUBLE.value == "soluble"


@patch("prxteinmpnn.mpnn.joblib.load")
@patch("prxteinmpnn.mpnn.pathlib.Path")
def test_get_mpnn_model_defaults(mock_path: MagicMock, mock_load: MagicMock):
  """Test get_mpnn_model with default arguments.

  This test mocks file I/O to verify that the correct path is constructed
  and that the loaded data is correctly processed into JAX arrays.

  Args:
    mock_path: A mock for the pathlib.Path class.
    mock_load: A mock for the joblib.load function.

  Returns:
    None

  Raises:
    AssertionError: If dependencies are not called correctly or the output is invalid.

  """
  # Arrange: Set up the mock return value for joblib.load
  mock_checkpoint = {
    "layer1": {"weights": np.array([1.0, 2.0]), "bias": np.array([0.5])},
    "layer2": np.array([3.0, 4.0, 5.0]),
  }
  mock_load.return_value = mock_checkpoint

  # Arrange: Set up the mock for pathlib.Path to simulate path construction
  # The mock path object needs to handle chained calls to .parent and /
  mock_base_dir = MagicMock(spec=Path)
  mock_path.return_value.parent.parent = mock_base_dir

  # Act: Call the function with default arguments
  model_params = get_mpnn_model()

  # Assert: Verify the path construction
  expected_model_path = (
    mock_base_dir / "models" / ModelWeights.DEFAULT.value / ProteinMPNNModelVersion.V_48_002.value
  )
  mock_load.assert_called_once_with(expected_model_path)

  # Assert: Verify the output is a JAX PyTree with jnp.ndarray leaves
  assert isinstance(model_params, dict)
  assert isinstance(model_params["layer1"]["weights"], jnp.ndarray)
  assert isinstance(model_params["layer2"], jnp.ndarray)
  np.testing.assert_array_equal(model_params["layer1"]["weights"], np.array([1.0, 2.0]))


@patch("prxteinmpnn.mpnn.joblib.load")
@patch("prxteinmpnn.mpnn.pathlib.Path")
def test_get_mpnn_model_custom_args(mock_path: MagicMock, mock_load: MagicMock):
  """Test get_mpnn_model with custom arguments.

  This test ensures that custom model versions and weights are used to
  construct the correct file path for loading.

  Args:
    mock_path: A mock for the pathlib.Path class.
    mock_load: A mock for the joblib.load function.

  Returns:
    None

  Raises:
    AssertionError: If the file path is not constructed with the custom arguments.

  """
  # Arrange
  mock_load.return_value = {"params": np.array([1.0])}
  mock_base_dir = MagicMock(spec=Path)
  mock_path.return_value.parent.parent = mock_base_dir
  custom_version = ProteinMPNNModelVersion.V_48_030
  custom_weights = ModelWeights.SOLUBLE

  # Act
  get_mpnn_model(model_version=custom_version, model_weights=custom_weights)

  # Assert
  expected_model_path = (
    mock_base_dir / "models" / custom_weights.value / custom_version.value
  )
  mock_load.assert_called_once_with(expected_model_path)


@patch("prxteinmpnn.mpnn.joblib.load")
@patch("prxteinmpnn.mpnn.pathlib.Path")
def test_get_mpnn_model_file_not_found(mock_path: MagicMock, mock_load: MagicMock):
  """Test that get_mpnn_model propagates FileNotFoundError.

  Args:
    mock_path: A mock for the pathlib.Path class.
    mock_load: A mock for the joblib.load function.

  Returns:
    None

  Raises:
    AssertionError: If FileNotFoundError is not raised when the file is missing.

  """
  # Arrange: Configure the mock to raise a FileNotFoundError
  mock_load.side_effect = FileNotFoundError("Model file not found at path")
  mock_base_dir = MagicMock(spec=Path)
  mock_path.return_value.parent.parent = mock_base_dir

  # Act & Assert
  with pytest.raises(FileNotFoundError):
    get_mpnn_model()