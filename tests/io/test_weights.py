"""Test suite for the io.weights module."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from prxteinmpnn.io.weights import load_model, load_weights


def test_load_weights_with_none_initializes_glorot_normal():
  """Test that load_weights with None initializes with Glorot normal.

  Args:
    None

  Returns:
    None

  Raises:
    AssertionError: If initialization fails or values are incorrect.

  Example:
    >>> test_load_weights_with_none_initializes_glorot_normal()
  """
  # Create a simple model skeleton
  class SimpleModel(eqx.Module):
    """Simple test model with weight and bias."""

    weight: jax.Array
    bias: jax.Array
    some_static: str

    def __init__(self, in_features: int, out_features: int):
      self.weight = jnp.zeros((out_features, in_features))
      self.bias = jnp.zeros((out_features,))
      self.some_static = "test"

  skeleton = SimpleModel(in_features=10, out_features=5)
  key = jax.random.PRNGKey(42)

  # Load with None to trigger Glorot normal initialization
  initialized_model = load_weights(
    model_weights=None,
    skeleton=skeleton,
    key=key,
  )

  # Check that weights were initialized (not zeros)
  assert not jnp.allclose(initialized_model.weight, jnp.zeros_like(initialized_model.weight))
  assert not jnp.allclose(initialized_model.bias, jnp.zeros_like(initialized_model.bias))

  # Check that static attributes are preserved
  assert initialized_model.some_static == "test"

  # Check that the initialization is deterministic given the same key
  initialized_model2 = load_weights(
    model_weights=None,
    skeleton=skeleton,
    key=key,
  )
  assert jnp.allclose(initialized_model.weight, initialized_model2.weight)
  assert jnp.allclose(initialized_model.bias, initialized_model2.bias)

  # Check that different keys produce different initializations
  initialized_model3 = load_weights(
    model_weights=None,
    skeleton=skeleton,
    key=jax.random.PRNGKey(123),
  )
  assert not jnp.allclose(initialized_model.weight, initialized_model3.weight)


def test_load_weights_none_requires_skeleton():
  """Test that load_weights with None requires skeleton.

  Args:
    None

  Returns:
    None

  Raises:
    ValueError: If skeleton is not provided.

  Example:
    >>> test_load_weights_none_requires_skeleton()
  """
  with pytest.raises(ValueError, match="skeleton is required when model_weights is None"):
    load_weights(model_weights=None, skeleton=None)


def test_load_weights_preserves_structure():
  """Test that load_weights preserves model structure.

  Args:
    None

  Returns:
    None

  Raises:
    AssertionError: If structure is not preserved.

  Example:
    >>> test_load_weights_preserves_structure()
  """
  # Create a nested model structure
  class InnerModule(eqx.Module):
    """Inner test module."""

    weight: jax.Array
    static_val: int

    def __init__(self):
      self.weight = jnp.zeros((3, 3))
      self.static_val = 42

  class OuterModule(eqx.Module):
    """Outer test module."""

    inner: InnerModule
    bias: jax.Array
    name: str

    def __init__(self):
      self.inner = InnerModule()
      self.bias = jnp.zeros((3,))
      self.name = "outer"

  skeleton = OuterModule()
  key = jax.random.PRNGKey(0)

  initialized = load_weights(
    model_weights=None,
    skeleton=skeleton,
    key=key,
  )

  # Check structure is preserved
  assert isinstance(initialized, OuterModule)
  assert isinstance(initialized.inner, InnerModule)
  assert initialized.name == "outer"
  assert initialized.inner.static_val == 42

  # Check arrays were initialized
  assert not jnp.allclose(initialized.inner.weight, jnp.zeros_like(initialized.inner.weight))
  assert not jnp.allclose(initialized.bias, jnp.zeros_like(initialized.bias))
