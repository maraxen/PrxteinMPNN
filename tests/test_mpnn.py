# tests/test_mpnn.py

"""Tests for prxteinmpnn.mpnn."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import jax.numpy as jnp
import numpy as np
import pytest

from prxteinmpnn.functional import get_functional_model as get_mpnn_model
from prxteinmpnn.functional.model import ModelVersion, ModelWeights






@patch("prxteinmpnn.functional.model.joblib.load")
@patch("prxteinmpnn.functional.model.pathlib.Path")
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
    "model_state_dict": {
    "layer1": {"weights": np.array([1.0, 2.0]), "bias": np.array([0.5])},
    "layer2": np.array([3.0, 4.0, 5.0]),
    }
  }
  mock_load.return_value = mock_checkpoint

  # Arrange: Set up the mock for pathlib.Path to simulate path construction
  # The mock path object needs to handle chained calls to .parent.parent and /
  mock_base_dir = MagicMock(spec=Path)
  mock_path.return_value.parent.parent = mock_base_dir

  # Act: Call the function with default arguments
  model_params = get_mpnn_model()

  # Assert: Verify the path construction
  expected_model_path = (
    mock_base_dir / "model" / "original" / "v_48_020"
  )
  mock_load.assert_called_once_with(expected_model_path)

  # Assert: Verify the output is a JAX PyTree with jnp.ndarray leaves
  assert isinstance(model_params, dict)
  assert isinstance(model_params["layer1"]["weights"], jnp.ndarray)
  assert isinstance(model_params["layer2"], jnp.ndarray)
  np.testing.assert_array_equal(model_params["layer1"]["weights"], np.array([1.0, 2.0]))


@patch("prxteinmpnn.functional.model.joblib.load")
@patch("prxteinmpnn.functional.model.pathlib.Path")
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
  mock_load.return_value = {"model_state_dict": {"params": np.array([1.0])}}
  mock_base_dir = MagicMock(spec=Path)
  mock_path.return_value.parent.parent = mock_base_dir
  custom_version = "v_48_030"
  custom_weights = "soluble"

  # Act
  get_mpnn_model(model_version=custom_version, model_weights=custom_weights)

  # Assert
  expected_model_path = (
    mock_base_dir / "model" / custom_weights / custom_version
  )
  mock_load.assert_called_once_with(expected_model_path)


@patch("prxteinmpnn.functional.model.joblib.load")
@patch("prxteinmpnn.functional.model.pathlib.Path")
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