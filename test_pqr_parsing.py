"""Test script to verify PQR parsing populates full_coordinates and charges correctly."""

import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from prxteinmpnn.io.parsing.dispatch import parse_input


def test_pqr_parsing():
  """Test that PQR files are parsed with full_coordinates and charges."""
  # Find a PQR file
  pqr_dir = Path(__file__).parent / "scripts" / "overfit" / "data"
  pqr_files = list(pqr_dir.glob("*.pqr"))

  if not pqr_files:
    print(f"No PQR files found in {pqr_dir}")
    return

  pqr_file = pqr_files[0]
  print(f"\nTesting PQR file: {pqr_file.name}")

  # Parse the file
  for protein in parse_input(pqr_file):
    print(f"\n{'=' * 60}")
    print(f"Source: {protein.source}")
    print(f"{'=' * 60}")

    # Check basic structure
    print("\nBasic structure info:")
    print(f"  - Number of residues: {protein.coordinates.shape[0]}")
    print(f"  - Backbone coordinates shape: {protein.coordinates.shape}")
    print(f"  - Atom mask shape: {protein.atom_mask.shape}")

    # Check full_coordinates
    if protein.full_coordinates is not None:
      print("\n✓ full_coordinates populated!")
      print(f"  - Shape: {protein.full_coordinates.shape}")
      print(f"  - Number of all atoms: {protein.full_coordinates.shape[0]}")
    else:
      print("\n✗ full_coordinates is None")

    # Check charges
    if protein.charges is not None:
      print("\n✓ charges populated!")
      print(f"  - Shape: {protein.charges.shape}")
      print(f"  - Number of charged atoms: {protein.charges.shape[0]}")
      print(f"  - Charge range: [{protein.charges.min():.3f}, {protein.charges.max():.3f}]")
    else:
      print("\n✗ charges is None")

    # Check radii
    if protein.radii is not None:
      print("\n✓ radii populated!")
      print(f"  - Shape: {protein.radii.shape}")
    else:
      print("\n✗ radii is None")

    # Verify alignment
    if protein.full_coordinates is not None and protein.charges is not None:
      n_coords = protein.full_coordinates.shape[0]
      n_charges = protein.charges.shape[0]
      if n_coords == n_charges:
        print(f"\n✓ Alignment verified: {n_coords} atoms with charges match coordinates")
      else:
        print(f"\n✗ Mismatch: {n_coords} atoms in full_coordinates vs {n_charges} charges")

    # Check electrostatics masks
    if protein.estat_backbone_mask is not None:
      print("\nestat_backbone_mask:")
      print(f"  - Shape: {protein.estat_backbone_mask.shape}")
      print(f"  - Backbone atoms: {protein.estat_backbone_mask.sum()}")
      print(f"  - Total atoms: {len(protein.estat_backbone_mask)}")

      # Show first residue worth of atoms to understand structure
      import numpy as np

      backbone_indices = np.where(protein.estat_backbone_mask)[0]
      print(f"  - First 10 backbone atom indices: {backbone_indices[:10]}")
      print(
        f"  - Expected: ~{protein.coordinates.shape[0] * 5} backbone atoms for {protein.coordinates.shape[0]} residues"
      )

    break  # Only test first structure


if __name__ == "__main__":
  test_pqr_parsing()
