import jax
import jax.numpy as jnp

from prxteinmpnn.model.dense import dense_layer
from prxteinmpnn.utils.types import ModelParameters


def make_dense_params(
  in_dim: int,
  hidden_dim: int,
  out_dim: int,
  dtype=jnp.float32,
) -> ModelParameters:
  """Create mock parameters for a dense layer.

  Args:
      in_dim (int): Input feature dimension.
      hidden_dim (int): Hidden layer dimension.
      out_dim (int): Output feature dimension.
      dtype: JAX dtype for parameters.

  Returns:
      ModelParameters: Dictionary of dense layer parameters.

  """
  key = jax.random.PRNGKey(0)
  w_in = jax.random.normal(key, (in_dim, hidden_dim), dtype)
  b_in = jnp.zeros((hidden_dim,), dtype)
  w_out = jax.random.normal(key, (hidden_dim, out_dim), dtype)
  b_out = jnp.zeros((out_dim,), dtype)
  return {
    "dense_W_in": {"w": w_in, "b": b_in},
    "dense_W_out": {"w": w_out, "b": b_out},
  }


def test_dense_layer_output_shape():
  """Test that dense_layer returns correct output shape.

  Args:
      None

  Returns:
      None

  Raises:
      AssertionError: If output shape is incorrect.

  Example:
      >>> test_dense_layer_output_shape()

  """
  in_dim, hidden_dim, out_dim = 4, 8, 3
  batch_size = 5
  params = make_dense_params(in_dim, hidden_dim, out_dim)
  x = jnp.ones((batch_size, in_dim), dtype=jnp.float32)
  y = dense_layer(params, x)
  assert y.shape == (batch_size, out_dim), f"Expected {(batch_size, out_dim)}, got {y.shape}"


def test_dense_layer_deterministic():
  """Test that dense_layer is deterministic for same input.

  Args:
      None

  Returns:
      None

  Raises:
      AssertionError: If outputs differ for same input.

  Example:
      >>> test_dense_layer_deterministic()

  """
  in_dim, hidden_dim, out_dim = 2, 4, 2
  params = make_dense_params(in_dim, hidden_dim, out_dim)
  x = jnp.array([[1.0, -1.0], [0.5, 0.5]], dtype=jnp.float32)
  y1 = dense_layer(params, x)
  y2 = dense_layer(params, x)
  assert jnp.allclose(y1, y2), "dense_layer is not deterministic for same input"


def test_dense_layer_zero_input():
  """Test dense_layer with zero input.

  Args:
      None

  Returns:
      None

  Raises:
      AssertionError: If output is not finite.

  Example:
      >>> test_dense_layer_zero_input()

  """
  in_dim, hidden_dim, out_dim = 3, 6, 2
  params = make_dense_params(in_dim, hidden_dim, out_dim)
  x = jnp.zeros((7, in_dim), dtype=jnp.float32)
  y = dense_layer(params, x)
  assert jnp.all(jnp.isfinite(y)), "Output contains non-finite values"


def test_dense_layer_batching():
  """Test dense_layer with different batch sizes.

  Args:
      None

  Returns:
      None

  Raises:
      AssertionError: If output shape is incorrect for different batch sizes.

  Example:
      >>> test_dense_layer_batching()

  """
  in_dim, hidden_dim, out_dim = 5, 10, 4
  params = make_dense_params(in_dim, hidden_dim, out_dim)
  for batch_size in [1, 10, 32]:
    x = jnp.ones((batch_size, in_dim), dtype=jnp.float32)
    y = dense_layer(params, x)
    assert y.shape == (batch_size, out_dim), f"Batch {batch_size}: got {y.shape}"


def test_dense_layer_grad():
  """Test that dense_layer is differentiable.

  Args:
      None

  Returns:
      None

  Raises:
      AssertionError: If gradient computation fails.

  Example:
      >>> test_dense_layer_grad()

  """
  in_dim, hidden_dim, out_dim = 3, 5, 2
  params = make_dense_params(in_dim, hidden_dim, out_dim)
  x = jnp.ones((2, in_dim), dtype=jnp.float32)

  def loss_fn(x_):
    y = dense_layer(params, x_)
    return jnp.sum(y)

  grad_fn = jax.grad(loss_fn)
  grad = grad_fn(x)
  assert grad.shape == x.shape, f"Expected grad shape {x.shape}, got {grad.shape}"
