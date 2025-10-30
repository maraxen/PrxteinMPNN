"""Tests for Equinox-based functional wrappers.

These tests verify that the new Equinox-based functional wrappers provide
the same interface and numerical equivalence as the legacy functional implementation.
"""

import jax
import jax.numpy as jnp
import pytest

from prxteinmpnn import conversion
from prxteinmpnn.functional import (
  final_projection,
  get_functional_model,
  make_decoder,
  make_decoder_eqx,
  make_encoder,
  make_encoder_eqx,
  make_model_eqx,
)


class TestEquinoxWrappers:
  """Test Equinox-based functional wrappers."""

  def test_make_encoder_eqx_signature(self) -> None:
    """Equinox encoder wrapper should have same signature as legacy."""
    params = get_functional_model()
    key = jax.random.PRNGKey(0)

    # Create both encoders
    legacy_encoder = make_encoder(params, num_encoder_layers=3, scale=30.0)
    eqx_encoder = make_encoder_eqx(params, num_encoder_layers=3, scale=30.0, key=key)

    # Create test input
    edge_features = jax.random.normal(key, (20, 15, 128))
    neighbor_indices = jnp.arange(20)[:, None].repeat(15, axis=1)
    mask = jnp.ones(20)

    # Both should work with same inputs
    legacy_nodes, legacy_edges = legacy_encoder(edge_features, neighbor_indices, mask)
    eqx_nodes, eqx_edges = eqx_encoder(edge_features, neighbor_indices, mask)

    # Check shapes match
    assert legacy_nodes.shape == eqx_nodes.shape
    assert legacy_edges.shape == eqx_edges.shape

  def test_make_encoder_eqx_equivalence(self) -> None:
    """Equinox encoder wrapper should produce equivalent outputs."""
    params = get_functional_model()
    key = jax.random.PRNGKey(42)

    # Create both encoders
    legacy_encoder = make_encoder(params, num_encoder_layers=3, scale=30.0)
    eqx_encoder = make_encoder_eqx(params, num_encoder_layers=3, scale=30.0, key=key)

    # Create test input
    edge_features = jax.random.normal(key, (20, 15, 128))
    neighbor_indices = jnp.arange(20)[:, None].repeat(15, axis=1)
    mask = jnp.ones(20)

    # Run both
    legacy_nodes, legacy_edges = legacy_encoder(edge_features, neighbor_indices, mask)
    eqx_nodes, eqx_edges = eqx_encoder(edge_features, neighbor_indices, mask)

    # Check numerical equivalence (use slightly looser tolerance for edges)
    assert jnp.allclose(legacy_nodes, eqx_nodes, rtol=1e-5, atol=1e-5)
    assert jnp.allclose(legacy_edges, eqx_edges, rtol=1e-4, atol=1e-5)

  def test_make_decoder_eqx_signature(self) -> None:
    """Equinox decoder wrapper should have same signature as legacy."""
    params = get_functional_model()
    key = jax.random.PRNGKey(0)

    # Create both decoders
    legacy_decoder = make_decoder(
      params,
      attention_mask_type=None,
      num_decoder_layers=3,
      scale=30.0,
    )
    eqx_decoder = make_decoder_eqx(params, num_decoder_layers=3, scale=30.0, key=key)

    # Create test input
    node_features = jax.random.normal(key, (20, 128))
    edge_features = jax.random.normal(jax.random.PRNGKey(1), (20, 15, 128))
    mask = jnp.ones(20)

    # Both should work with same inputs
    legacy_output = legacy_decoder(node_features, edge_features, mask)
    eqx_output = eqx_decoder(node_features, edge_features, mask)

    # Check shapes match
    assert legacy_output.shape == eqx_output.shape

  def test_make_decoder_eqx_equivalence(self) -> None:
    """Equinox decoder wrapper should produce equivalent outputs."""
    params = get_functional_model()
    key = jax.random.PRNGKey(42)

    # Create both decoders
    legacy_decoder = make_decoder(
      params,
      attention_mask_type=None,
      num_decoder_layers=3,
      scale=30.0,
    )
    eqx_decoder = make_decoder_eqx(params, num_decoder_layers=3, scale=30.0, key=key)

    # Create test input
    node_features = jax.random.normal(key, (20, 128))
    edge_features = jax.random.normal(jax.random.PRNGKey(1), (20, 15, 128))
    mask = jnp.ones(20)

    # Run both
    legacy_output = legacy_decoder(node_features, edge_features, mask)
    eqx_output = eqx_decoder(node_features, edge_features, mask)

    # Check numerical equivalence
    assert jnp.allclose(legacy_output, eqx_output, rtol=1e-5, atol=1e-5)

  def test_make_model_eqx_full_pipeline(self) -> None:
    """Equinox model wrapper should work end-to-end."""
    params = get_functional_model()
    key = jax.random.PRNGKey(42)

    # Create Equinox-wrapped model
    model_fn = make_model_eqx(
      params,
      num_encoder_layers=3,
      num_decoder_layers=3,
      scale=30.0,
      key=key,
    )

    # Create test input
    edge_features = jax.random.normal(key, (20, 15, 128))
    neighbor_indices = jnp.arange(20)[:, None].repeat(15, axis=1)
    mask = jnp.ones(20)

    # Should produce logits
    logits = model_fn(edge_features, neighbor_indices, mask)

    # Check output shape
    assert logits.shape == (20, 21)  # 20 atoms, 21 amino acids

  def test_make_model_eqx_equivalence_with_legacy(self) -> None:
    """Equinox model wrapper should match legacy pipeline."""
    params = get_functional_model()
    key = jax.random.PRNGKey(42)

    # Create Equinox-wrapped model
    eqx_model_fn = make_model_eqx(
      params,
      num_encoder_layers=3,
      num_decoder_layers=3,
      scale=30.0,
      key=key,
    )

    # Create legacy pipeline
    legacy_encoder = make_encoder(params, num_encoder_layers=3, scale=30.0)
    legacy_decoder = make_decoder(
      params,
      attention_mask_type=None,
      num_decoder_layers=3,
      scale=30.0,
    )

    # Create test input
    edge_features = jax.random.normal(key, (20, 15, 128))
    neighbor_indices = jnp.arange(20)[:, None].repeat(15, axis=1)
    mask = jnp.ones(20)

    # Run Equinox model
    eqx_logits = eqx_model_fn(edge_features, neighbor_indices, mask)

    # Run legacy pipeline
    legacy_nodes, legacy_edges = legacy_encoder(edge_features, neighbor_indices, mask)
    legacy_nodes = legacy_decoder(legacy_nodes, legacy_edges, mask)
    legacy_logits = final_projection(params, legacy_nodes)

    # Check numerical equivalence
    assert jnp.allclose(eqx_logits, legacy_logits, rtol=1e-5, atol=1e-5)

  def test_make_model_eqx_equivalence_with_equinox_module(self) -> None:
    """Equinox wrapper should match direct Equinox module."""
    params = get_functional_model()
    key = jax.random.PRNGKey(42)

    # Create Equinox-wrapped model
    eqx_wrapped_fn = make_model_eqx(
      params,
      num_encoder_layers=3,
      num_decoder_layers=3,
      scale=30.0,
      key=key,
    )

    # Create direct Equinox model
    eqx_model = conversion.create_prxteinmpnn(
      params,
      num_encoder_layers=3,
      num_decoder_layers=3,
      scale=30.0,
      key=key,
    )

    # Create test input
    edge_features = jax.random.normal(key, (20, 15, 128))
    neighbor_indices = jnp.arange(20)[:, None].repeat(15, axis=1)
    mask = jnp.ones(20)

    # Run both
    wrapped_logits = eqx_wrapped_fn(edge_features, neighbor_indices, mask)
    direct_logits = eqx_model(edge_features, neighbor_indices, mask)

    # Should be very close (both use same Equinox model internally)
    # Note: Slight numerical differences due to separate model instances
    assert jnp.allclose(wrapped_logits, direct_logits, rtol=1e-5, atol=1e-5)

  def test_eqx_wrappers_jit_compatible(self) -> None:
    """Equinox wrappers should be JIT-compilable."""
    params = get_functional_model()
    key = jax.random.PRNGKey(42)

    # Create wrapped functions
    encoder_fn = make_encoder_eqx(params, num_encoder_layers=3, key=key)
    decoder_fn = make_decoder_eqx(params, num_decoder_layers=3, key=key)
    model_fn = make_model_eqx(params, key=key)

    # Create test inputs
    edge_features = jax.random.normal(key, (20, 15, 128))
    neighbor_indices = jnp.arange(20)[:, None].repeat(15, axis=1)
    mask = jnp.ones(20)
    node_features = jax.random.normal(jax.random.PRNGKey(1), (20, 128))

    # All should be JIT-compilable (functions are already @jax.jit decorated)
    # Just verify they run without errors
    nodes, edges = encoder_fn(edge_features, neighbor_indices, mask)
    decoded_nodes = decoder_fn(node_features, edges, mask)
    logits = model_fn(edge_features, neighbor_indices, mask)

    # Verify outputs have correct shapes
    assert nodes.shape == (20, 128)
    assert edges.shape == (20, 15, 128)
    assert decoded_nodes.shape == (20, 128)
    assert logits.shape == (20, 21)

  def test_eqx_wrappers_default_key(self) -> None:
    """Equinox wrappers should work without explicit key."""
    params = get_functional_model()

    # Should work without providing key (uses default PRNGKey(0))
    encoder_fn = make_encoder_eqx(params, num_encoder_layers=3)
    decoder_fn = make_decoder_eqx(params, num_decoder_layers=3)
    model_fn = make_model_eqx(params)

    # Create test inputs
    key = jax.random.PRNGKey(42)
    edge_features = jax.random.normal(key, (20, 15, 128))
    neighbor_indices = jnp.arange(20)[:, None].repeat(15, axis=1)
    mask = jnp.ones(20)
    node_features = jax.random.normal(jax.random.PRNGKey(1), (20, 128))

    # Should run without errors
    _ = encoder_fn(edge_features, neighbor_indices, mask)
    _ = decoder_fn(node_features, edge_features, mask)
    _ = model_fn(edge_features, neighbor_indices, mask)
