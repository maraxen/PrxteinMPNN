import jax.numpy as jnp
import pytest

from prxteinmpnn.model.projection import final_projection


@pytest.fixture
def mock_mpnn_parameters() -> dict[str, dict[str, jnp.ndarray]]:
  """Fixture for mock ModelParameters for final_projection.

  Returns:
    Dict[str, Dict[str, jnp.ndarray]]: Mock parameters with 'w' and 'b'.

  """
  # 3 input features, 2 output logits
  w = jnp.array([[1.0, 2.0], [0.5, -1.0], [0.0, 1.0]], dtype=jnp.float32)
  b = jnp.array([0.1, -0.2], dtype=jnp.float32)
  return {
    "protein_mpnn/~/W_out": {
      "w": w,
      "b": b,
    },
  }


@pytest.fixture
def mock_node_features() -> jnp.ndarray:
  """Fixture for mock NodeFeatures for final_projection.

  Returns:
    jnp.ndarray: Node features of shape (batch, features).

  """
  # 2 nodes, 3 features each
  return jnp.array([[1.0, 2.0, 3.0], [0.0, -1.0, 0.5]], dtype=jnp.float32)


def test_final_projection_shape(mock_mpnn_parameters, mock_node_features):
  """Test that final_projection returns correct shape.

  Args:
    mock_mpnn_parameters: Mock model parameters.
    mock_node_features: Mock node features.

  Returns:
    None

  Raises:
    AssertionError: If output shape is incorrect.

  Example:
    >>> test_final_projection_shape(mock_mpnn_parameters, mock_node_features)

  """
  logits = final_projection(mock_mpnn_parameters, mock_node_features)
  assert logits.shape == (2, 2), f"Expected shape (2, 2), got {logits.shape}"


def test_final_projection_values(mock_mpnn_parameters, mock_node_features):
  """Test that final_projection computes correct values.

  Args:
    mock_mpnn_parameters: Mock model parameters.
    mock_node_features: Mock node features.

  Returns:
    None

  Raises:
    AssertionError: If output values are incorrect.

  Example:
    >>> test_final_projection_values(mock_mpnn_parameters, mock_node_features)

  """
  logits = final_projection(mock_mpnn_parameters, mock_node_features)
  # Manually compute expected values
  w = mock_mpnn_parameters["protein_mpnn/~/W_out"]["w"]
  b = mock_mpnn_parameters["protein_mpnn/~/W_out"]["b"]
  expected = jnp.dot(mock_node_features, w) + b
  assert jnp.allclose(logits, expected), f"Expected {expected}, got {logits}"


def test_final_projection_batch_consistency(mock_mpnn_parameters):
  """Test that final_projection works for single and batched inputs.

  Args:
    mock_mpnn_parameters: Mock model parameters.

  Returns:
    None

  Raises:
    AssertionError: If output for batch size 1 does not match single input.

  Example:
    >>> test_final_projection_batch_consistency(mock_mpnn_parameters)

  """
  single_node = jnp.array([[1.0, 0.0, -1.0]], dtype=jnp.float32)
  batch_nodes = jnp.vstack([single_node, single_node])
  logits_single = final_projection(mock_mpnn_parameters, single_node)
  logits_batch = final_projection(mock_mpnn_parameters, batch_nodes)
  assert jnp.allclose(logits_batch[0], logits_single[0]), (
    f"Batch and single input results differ: {logits_batch[0]} vs {logits_single[0]}"
  )
