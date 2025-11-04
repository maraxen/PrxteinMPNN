"""Tests for the architecture adapter in functional/model.py.

This test validates Phase 1 of the migration: the adapter layer that allows
both legacy functional and new Equinox architectures to coexist.
"""

import jax

from prxteinmpnn.eqx_new import PrxteinMPNN
from prxteinmpnn.functional.model import get_functional_model


class TestArchitectureAdapter:
  """Test the get_functional_model adapter with use_new_architecture flag."""

  def test_load_new_architecture(self):
    """Test that use_new_architecture=True returns a PrxteinMPNN instance."""
    model = get_functional_model(
      model_version="v_48_020",
      model_weights="original",
      use_new_architecture=True,
    )

    # Verify it's an Equinox PrxteinMPNN instance
    assert isinstance(model, PrxteinMPNN), f"Expected PrxteinMPNN, got {type(model)}"
    print(f"✓ New architecture loaded: {type(model)}")

  def test_load_new_architecture_all_versions(self):
    """Test that all model versions can be loaded with the new architecture."""
    versions = ["v_48_002", "v_48_010", "v_48_020", "v_48_030"]
    weights = ["original", "soluble"]

    for version in versions:
      for weight in weights:
        model = get_functional_model(
          model_version=version,
          model_weights=weight,
          use_new_architecture=True,
        )
        assert isinstance(
          model, PrxteinMPNN,
        ), f"Failed for {weight}/{version}: got {type(model)}"
        print(f"✓ Loaded {weight}/{version}")

  def test_new_architecture_forward_pass(self):
    """Test that models loaded via new architecture can run forward passes."""
    model = get_functional_model(
      model_version="v_48_020",
      model_weights="original",
      use_new_architecture=True,
    )

    # Create dummy inputs (no batch dimension for single example)
    seq_len = 10
    k_neighbors = 48

    key = jax.random.PRNGKey(42)
    dummy_edge_features = jax.random.normal(key, (seq_len, k_neighbors, 128))
    dummy_neighbors = jax.random.randint(key, (seq_len, k_neighbors), 0, seq_len)
    dummy_mask = jax.numpy.ones((seq_len,), dtype=jax.numpy.int32)

    # Run a forward pass (unconditional)
    dummy_seq, logits = model._call_unconditional(
      dummy_edge_features, dummy_neighbors, dummy_mask,
    )

    # Verify shapes (unconditional returns dummy sequence and logits)
    assert dummy_seq.shape == (
      seq_len,
      21,
    ), f"Expected ({seq_len}, 21), got {dummy_seq.shape}"
    assert logits.shape == (seq_len, 21), f"Expected ({seq_len}, 21), got {logits.shape}"
    print(f"✓ Forward pass successful: dummy_seq={dummy_seq.shape}, logits={logits.shape}")

  def test_default_is_new_architecture(self):
    """Test that use_new_architecture defaults to True (new Equinox architecture)."""
    # After Phase 5 migration, the default is now the new Equinox architecture
    # This ensures we're using the modern, efficient implementation by default
    import inspect

    sig = inspect.signature(get_functional_model)
    use_new_arch_param = sig.parameters["use_new_architecture"]

    assert use_new_arch_param.default is True, "Default should be True for new architecture"
    print("✓ Default use_new_architecture=True confirmed (Equinox by default)")
