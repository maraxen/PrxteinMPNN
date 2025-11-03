"""Tests for loading models from HuggingFace Hub."""

import tempfile
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from prxteinmpnn.eqx_new import PrxteinMPNN
from prxteinmpnn.io.weights import (
  ALL_MODEL_VERSIONS,
  ALL_MODEL_WEIGHTS,
  load_model,
  load_weights,
)


class TestLoadModel:
  """Tests for the high-level load_model API."""

  def test_load_model_default(self):
    """Test loading the default model (original_v_48_020)."""
    model = load_model()
    assert isinstance(model, PrxteinMPNN)
    assert model.node_features_dim == 128
    assert model.edge_features_dim == 128
    # hidden_features is used in MLPs, not stored as a direct attribute

  def test_load_model_with_params(self):
    """Test loading a specific model variant."""
    model = load_model(model_version="v_48_010", model_weights="soluble")
    assert isinstance(model, PrxteinMPNN)

  def test_load_model_with_custom_key(self):
    """Test that custom JAX key is properly handled."""
    key = jax.random.PRNGKey(42)
    model = load_model(key=key)
    assert isinstance(model, PrxteinMPNN)

  def test_load_model_forward_pass(self):
    """Test that a loaded model can perform forward pass."""
    model = load_model(model_version="v_48_020", model_weights="original")

    # Create dummy inputs (single batch, no batch dimension for _call_unconditional)
    seq_len = 25
    k = 48

    key = jax.random.PRNGKey(0)

    # Edge features [L, K, edge_features]
    edge_features = jax.random.normal(key, (seq_len, k, 128))

    # Neighbor indices [L, K]
    neighbor_indices = jnp.tile(jnp.arange(seq_len)[:, None], (1, k))

    # Mask [L]
    mask = jnp.ones(seq_len)

    # Forward pass
    _, logits = model._call_unconditional(
      edge_features,
      neighbor_indices,
      mask,
    )

    # Check output shape
    assert logits.shape == (seq_len, 21)

    # Check for NaN/Inf
    assert not jnp.isnan(logits).any()
    assert not jnp.isinf(logits).any()


class TestLoadWeights:
  """Tests for the low-level load_weights API."""

  def test_load_weights_eqx_format(self):
    """Test loading weights in .eqx format."""
    key = jax.random.PRNGKey(0)
    skeleton = PrxteinMPNN(
      node_features=128,
      edge_features=128,
      hidden_features=512,
      num_encoder_layers=3,
      num_decoder_layers=3,
      vocab_size=21,
      k_neighbors=48,
      key=key,
    )

    model = load_weights(
      model_version="v_48_020",
      model_weights="original",
      skeleton=skeleton,
      use_eqx_format=True,
    )

    assert isinstance(model, PrxteinMPNN)

  def test_load_weights_requires_skeleton(self):
    """Test that loading .eqx format requires a skeleton."""
    with pytest.raises(ValueError, match="skeleton must be provided"):
      load_weights(
        model_version="v_48_020",
        model_weights="original",
        skeleton=None,
        use_eqx_format=True,
      )


class TestSaveLoadRoundtrip:
  """Tests for save/load roundtrip consistency."""

  def test_save_load_roundtrip(self):
    """Test that saving and loading preserves model exactly."""
    # Load original model
    model = load_model(model_version="v_48_020", model_weights="original")

    # Create test input (no batch dimension)
    seq_len = 10
    k = 48

    key = jax.random.PRNGKey(123)

    edge_features = jax.random.normal(key, (seq_len, k, 128))
    neighbor_indices = jnp.tile(jnp.arange(seq_len)[:, None], (1, k))
    mask = jnp.ones(seq_len)

    # Get output from original model
    _, logits_original = model._call_unconditional(
      edge_features,
      neighbor_indices,
      mask,
    )

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".eqx", delete=False) as tmp_file:
      tmp_path = Path(tmp_file.name)
      eqx.tree_serialise_leaves(tmp_path, model)

    # Load from temp file
    reloaded_model = load_model(local_path=str(tmp_path))

    # Get output from reloaded model
    _, logits_reloaded = reloaded_model._call_unconditional(
      edge_features,
      neighbor_indices,
      mask,
    )

    # Check that outputs are exactly the same (bit-perfect)
    assert jnp.allclose(logits_original, logits_reloaded, rtol=0, atol=0)

    # Cleanup
    tmp_path.unlink()


class TestAllModels:
  """Tests that all model variants are downloadable."""

  @pytest.mark.parametrize("model_version", ALL_MODEL_VERSIONS)
  @pytest.mark.parametrize("model_weights", ALL_MODEL_WEIGHTS)
  def test_all_models_loadable(self, model_version, model_weights):
    """Test that all model variants can be loaded."""
    model = load_model(model_version=model_version, model_weights=model_weights)
    assert isinstance(model, PrxteinMPNN)
    # Quick sanity check - model can be called
    # (Don't actually run forward pass to keep tests fast)


@pytest.mark.slow
class TestDownloadPerformance:
  """Performance tests for model downloading (marked slow)."""

  def test_caching_works(self):
    """Test that HuggingFace Hub caching works (second load is fast)."""
    import time

    # First load (should download)
    start = time.time()
    model1 = load_model(model_version="v_48_020", model_weights="original")
    first_time = time.time() - start

    # Second load (should use cache)
    start = time.time()
    model2 = load_model(model_version="v_48_020", model_weights="original")
    second_time = time.time() - start

    # Second load should be much faster (at least 10x)
    assert second_time < first_time / 10

    # Models should be equivalent
    assert isinstance(model1, PrxteinMPNN)
    assert isinstance(model2, PrxteinMPNN)
