"""Test suite for the io.weights module."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from prxteinmpnn.io.weights import load_model, load_weights


def test_load_weights_reinitialization():
  """Test that load_weights with no ID initializes with Glorot normal."""

  class SimpleModel(eqx.Module):
    weight: jax.Array
    bias: jax.Array
    some_static: str

    def __init__(self, in_features: int, out_features: int):
      self.weight = jnp.zeros((out_features, in_features))
      self.bias = jnp.zeros((out_features,))
      self.some_static = "test"

  skeleton = SimpleModel(in_features=10, out_features=5)
  key = jax.random.PRNGKey(42)

  # Load with no checkpoint_id to trigger reinitialization
  initialized_model = load_weights(
    skeleton=skeleton,
    key=key,
  )

  # Check that weights were initialized (not zeros)
  assert not jnp.allclose(initialized_model.weight, jnp.zeros_like(initialized_model.weight))
  assert not jnp.allclose(initialized_model.bias, jnp.zeros_like(initialized_model.bias))
  assert initialized_model.some_static == "test"


def test_load_weights_requires_skeleton():
  """Test that load_weights for reinitialization requires skeleton."""
  with pytest.raises(ValueError, match="skeleton is required for reinitialization"):
    load_weights(checkpoint_id=None, skeleton=None)


@pytest.mark.parametrize(
  "checkpoint_id",
  [
    "proteinmpnn_v_48_020",
    "ligandmpnn_v_32_010_25",
    "ligandmpnn_sc_v_32_002_16",
    "global_label_membrane_mpnn_v_48_020",
  ],
)
def test_smart_factory_model_loading(checkpoint_id: str):
  """Test that load_model correctly dispatches and loads all model types."""
  # This test verifies the resources are found and the skeletons are correctly built
  model = load_model(checkpoint_id)
  assert isinstance(model, eqx.Module)

  # Check topology inference
  if "v_32" in checkpoint_id:
    feat = model.features
    if hasattr(feat, "k_neighbors"):
      assert feat.k_neighbors == 32
    elif hasattr(feat, "top_k"):
      assert feat.top_k == 32
  else:
    assert model.features.k_neighbors == 48

  # Verify weights aren't all zero
  leaves = jax.tree_util.tree_leaves(model)
  for leaf in leaves:
    if hasattr(leaf, "shape") and len(leaf.shape) > 0:
      assert not jnp.all(leaf == 0)
      break


def test_load_model_legacy_compatibility():
  """Test that the factory still supports legacy model_weights/version args."""
  model = load_model(model_weights="original", model_version="v_48_020")
  assert model.features.k_neighbors == 48


def test_load_model_membrane_detection():
  """Test that membrane models are detected and physics_feature_dim is set."""
  model = load_model("global_label_membrane_mpnn_v_48_020")
  # Membrane models use 3 physics features by default (from topo parser)
  # Check encoder for physics projection if it's a PhysicsEncoder
  if hasattr(model.encoder, "physics_projection"):
    assert model.encoder.physics_projection.in_features == 3
  else:
    # If not PhysicsEncoder, weight loading would have failed if mismatched,
    # but we can check node_features_dim
    assert model.node_features_dim == 128
