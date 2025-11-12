"""Test electrostatic feature computation with PQR data."""

import sys
from pathlib import Path

import jax.numpy as jnp

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from prxteinmpnn.io.parsing.dispatch import parse_input
from prxteinmpnn.physics.features import compute_electrostatic_node_features


def test_electrostatic_features() -> None:
  """Test that electrostatic features can be computed from PQR files."""
  # Find a PQR file
  pqr_dir = Path(__file__).parent / "scripts" / "overfit" / "data"
  pqr_files = list(pqr_dir.glob("*.pqr"))

  if not pqr_files:
    print(f"No PQR files found in {pqr_dir}")
    return

  pqr_file = pqr_files[0]
  print(f"\nTesting electrostatic features for: {pqr_file.name}")

  # Parse the file
  for protein in parse_input(pqr_file):
    print(f"\n{'=' * 60}")
    print(f"Protein: {protein.source}")
    print(f"Number of residues: {protein.coordinates.shape[0]}")
    print(f"{'=' * 60}")

    # Compute electrostatic features
    try:
      features = compute_electrostatic_node_features(protein)
      print("\n✓ Successfully computed electrostatic features!")
      print(f"  - Feature shape: {features.shape}")
      print(f"  - Expected shape: ({protein.coordinates.shape[0]}, 5)")
      print(f"  - Feature range: [{jnp.min(features):.6f}, {jnp.max(features):.6f}]")
      print(f"  - Feature mean: {jnp.mean(features):.6f}")
      print(f"  - Feature std: {jnp.std(features):.6f}")

      # Check that features are reasonable
      assert features.shape == (protein.coordinates.shape[0], 5), "Wrong shape!"
      assert jnp.all(jnp.isfinite(features)), "Features contain NaN or Inf!"

      # Show first few residues' features
      print("\n  First 3 residues' features:")
      for i in range(min(3, features.shape[0])):
        print(f"    Residue {i}: {features[i]}")

    except Exception as e:
      print("\n✗ Error computing electrostatic features:")
      print(f"  {type(e).__name__}: {e}")
      raise

    break  # Only test first structure


if __name__ == "__main__":
  test_electrostatic_features()
