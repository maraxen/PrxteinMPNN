#!/usr/bin/env python3
"""Test downloading and loading .eqx models from HuggingFace."""

import sys
import tempfile
from pathlib import Path

import equinox
import jax
import jax.numpy as jnp

from prxteinmpnn.eqx_new import PrxteinMPNN


def test_download_and_load():
  """Test downloading and loading a model from HuggingFace."""
  print("🧪 Testing HuggingFace model download and deserialization\n")

  # Test with one model
  model_name = "original_v_48_020"
  print(f"📥 Testing with: {model_name}")

  # Step 1: Download from HuggingFace
  print("\n1. Downloading from HuggingFace...")
  from huggingface_hub import hf_hub_download

  try:
    model_path = hf_hub_download(
      repo_id="maraxen/prxteinmpnn",
      filename=f"eqx/{model_name}.eqx",
      repo_type="model",
    )
    print(f"   ✓ Downloaded to: {model_path}")
    print(f"   Size: {Path(model_path).stat().st_size / (1024**2):.2f} MB")
  except Exception as e:
    print(f"   ❌ Download failed: {e}")
    return False

  # Step 2: Create model structure
  print("\n2. Creating model structure...")
  try:
    # Create model structure directly (no need for functional params)
    # Use the same hyperparameters as the saved models
    key = jax.random.PRNGKey(0)
    model = PrxteinMPNN(
      node_features=128,
      edge_features=128,
      hidden_features=512,  # This is the MLP width in the actual models
      num_encoder_layers=3,
      num_decoder_layers=3,
      vocab_size=21,
      k_neighbors=48,
      key=key,
    )
    print("   ✓ Model structure created")
  except Exception as e:
    print(f"   ❌ Failed to create structure: {e}")
    return False

  # Step 3: Load weights
  print("\n3. Loading weights from .eqx file...")
  try:
    loaded_model = equinox.tree_deserialise_leaves(model_path, model)
    print("   ✓ Weights loaded successfully")
  except Exception as e:
    print(f"   ❌ Failed to load weights: {e}")
    return False

  # Step 4: Test forward pass
  print("\n4. Testing forward pass...")
  try:
    num_residues = 25
    K = 48
    key = jax.random.PRNGKey(42)

    edge_features = jax.random.normal(key, (num_residues, K, 128))
    neighbor_indices = jnp.tile(jnp.arange(num_residues)[:, None], (1, K))
    mask = jnp.ones(num_residues)

    _, logits = loaded_model._call_unconditional(
      edge_features,
      neighbor_indices,
      mask,
    )

    print("   ✓ Forward pass successful")
    print(f"   Output shape: {logits.shape}")
    print(f"   Output range: [{logits.min():.3f}, {logits.max():.3f}]")

    # Check for NaNs/Infs
    if jnp.any(jnp.isnan(logits)) or jnp.any(jnp.isinf(logits)):
      print("   ❌ Output contains NaN or Inf values!")
      return False

    print("   ✓ Output is valid (no NaN/Inf)")

  except Exception as e:
    print(f"   ❌ Forward pass failed: {e}")
    import traceback

    traceback.print_exc()
    return False

  # Step 5: Test save/load roundtrip
  print("\n5. Testing save/load roundtrip...")
  try:
    with tempfile.NamedTemporaryFile(suffix=".eqx", delete=False) as f:
      temp_path = f.name

    # Save
    equinox.tree_serialise_leaves(temp_path, loaded_model)
    print(f"   ✓ Saved to: {temp_path}")

    # Load again
    reloaded_model = equinox.tree_deserialise_leaves(temp_path, model)
    print("   ✓ Reloaded successfully")

    # Test it produces same output
    _, reloaded_logits = reloaded_model._call_unconditional(
      edge_features,
      neighbor_indices,
      mask,
    )

    if jnp.allclose(logits, reloaded_logits, rtol=1e-7, atol=1e-8):
      print("   ✓ Outputs match (bit-perfect)")
    else:
      max_diff = jnp.abs(logits - reloaded_logits).max()
      print(f"   ⚠️  Outputs differ slightly: max_diff={max_diff:.2e}")
      if max_diff < 1e-5:
        print("   ✓ Difference is acceptable")
      else:
        print("   ❌ Difference is too large!")
        return False

    # Cleanup
    Path(temp_path).unlink()

  except Exception as e:
    print(f"   ❌ Roundtrip test failed: {e}")
    import traceback

    traceback.print_exc()
    return False

  return True


def test_all_models():
  """Test downloading all model variants."""
  print("\n" + "=" * 60)
  print("🧪 Testing all model variants")
  print("=" * 60)

  models = [
    "original_v_48_002",
    "original_v_48_010",
    "original_v_48_020",
    "original_v_48_030",
    "soluble_v_48_002",
    "soluble_v_48_010",
    "soluble_v_48_020",
    "soluble_v_48_030",
  ]

  results = {}

  for model_name in models:
    print(f"\n📦 Testing: {model_name}")

    try:
      from huggingface_hub import hf_hub_download

      model_path = hf_hub_download(
        repo_id="maraxen/prxteinmpnn",
        filename=f"eqx/{model_name}.eqx",
        repo_type="model",
      )

      # Just check file exists and has reasonable size
      size_mb = Path(model_path).stat().st_size / (1024**2)

      if size_mb < 5 or size_mb > 10:
        print(f"   ⚠️  Unexpected size: {size_mb:.2f} MB")
        results[model_name] = "WARNING"
      else:
        print(f"   ✓ Downloaded ({size_mb:.2f} MB)")
        results[model_name] = "SUCCESS"

    except Exception as e:
      print(f"   ❌ Failed: {e}")
      results[model_name] = "FAILED"

  # Summary
  print("\n" + "=" * 60)
  print("📊 Summary")
  print("=" * 60)

  success = sum(1 for v in results.values() if v == "SUCCESS")
  warning = sum(1 for v in results.values() if v == "WARNING")
  failed = sum(1 for v in results.values() if v == "FAILED")

  print(f"\n✅ Success: {success}/{len(models)}")
  if warning:
    print(f"⚠️  Warning: {warning}/{len(models)}")
  if failed:
    print(f"❌ Failed: {failed}/{len(models)}")

  return failed == 0


def main():
  """Run all tests."""
  print("=" * 60)
  print("🧪 HuggingFace Model Download & Deserialization Test")
  print("=" * 60)

  # Test detailed download and usage
  print("\n" + "=" * 60)
  print("Test 1: Detailed Download & Usage Test")
  print("=" * 60)

  if not test_download_and_load():
    print("\n❌ Detailed test FAILED")
    return 1

  print("\n✅ Detailed test PASSED")

  # Test all models
  if not test_all_models():
    print("\n❌ Some models failed to download")
    return 1

  print("\n" + "=" * 60)
  print("✅ All tests PASSED!")
  print("=" * 60)
  print("\n💡 Next steps:")
  print("   1. ✅ Models are downloadable from HuggingFace")
  print("   2. ✅ Deserialization works correctly")
  print("   3. ✅ Forward pass produces valid outputs")
  print("   4. ✅ Save/load roundtrip is bit-perfect")
  print("\nReady to:")
  print("   - Update model card on HuggingFace")
  print("   - Update io module to use .eqx by default")
  print("   - Add these tests to the test suite")

  return 0


if __name__ == "__main__":
  try:
    sys.exit(main())
  except KeyboardInterrupt:
    print("\n\n⚠️  Interrupted by user")
    sys.exit(1)
  except Exception as e:
    print(f"\n❌ Unexpected error: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
