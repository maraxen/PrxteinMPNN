"""Debug script to compare backbone coordinates between implementations."""

import jax
import jax.numpy as jnp
import numpy as np
from colabdesign.mpnn import mk_mpnn_model
from prxteinmpnn.io.parsing import parse_input
from prxteinmpnn.utils.data_structures import Protein
from prxteinmpnn.utils.coordinates import compute_backbone_coordinates

pdb_path = "tests/data/1ubq.pdb"

# 1. ColabDesign
print("=" * 80)
print("1. ColabDesign backbone coordinates")
print("=" * 80)
mpnn_model = mk_mpnn_model()
mpnn_model.prep_inputs(pdb_filename=pdb_path)
X_colab = mpnn_model._inputs["X"]  # Shape: (L, 4, 3) - N, CA, C, O
print(f"Input X shape: {X_colab.shape}")

# Compute CB like ColabDesign does
Y = jnp.array(X_colab).swapaxes(0, 1)  # (4, L, 3)
b, c = (Y[1] - Y[0]), (Y[2] - Y[1])
Cb = -0.58273431 * jnp.cross(b, c) + 0.56802827 * b - 0.54067466 * c + Y[1]
Y_with_cb = jnp.concatenate([Y, Cb[None]], 0)  # (5, L, 3)
colab_backbone = Y_with_cb.swapaxes(0, 1)  # (L, 5, 3)
print(f"Backbone shape: {colab_backbone.shape}")
print(f"\nFirst residue:")
print(f"  N  (idx 0): {colab_backbone[0, 0]}")
print(f"  CA (idx 1): {colab_backbone[0, 1]}")
print(f"  C  (idx 2): {colab_backbone[0, 2]}")
print(f"  O  (idx 3): {colab_backbone[0, 3]}")
print(f"  CB (idx 4): {colab_backbone[0, 4]}")

# 2. PrxteinMPNN
print("\n" + "=" * 80)
print("2. PrxteinMPNN backbone coordinates")
print("=" * 80)
protein_tuple = next(parse_input(pdb_path))
protein = Protein.from_tuple(protein_tuple)
prx_backbone = compute_backbone_coordinates(protein.coordinates)
print(f"Backbone shape: {prx_backbone.shape}")
print(f"\nFirst residue:")
print(f"  N  (idx 0): {prx_backbone[0, 0]}")
print(f"  CA (idx 1): {prx_backbone[0, 1]}")
print(f"  C  (idx 2): {prx_backbone[0, 2]}")
print(f"  O  (idx 3): {prx_backbone[0, 3]}")
print(f"  CB (idx 4): {prx_backbone[0, 4]}")

# 3. Compare
print("\n" + "=" * 80)
print("3. Comparison")
print("=" * 80)
diff = np.abs(np.array(colab_backbone) - np.array(prx_backbone))
print(f"Max difference: {diff.max():.9f}")
print(f"Mean difference: {diff.mean():.9f}")
print(f"\nDifferences per atom (first residue):")
for i, name in enumerate(["N", "CA", "C", "O", "CB"]):
    atom_diff = diff[0, i]
    print(f"  {name:3s}: {np.linalg.norm(atom_diff):.9f}")

# Check if they're essentially identical
if diff.max() < 1e-5:
    print("\n✅ Backbone coordinates match!")
else:
    print(f"\n❌ Backbone coordinates differ! Max diff: {diff.max()}")
    print("\nFirst few residues comparison:")
    for res_idx in range(min(3, colab_backbone.shape[0])):
        print(f"\nResidue {res_idx}:")
        for atom_idx, name in enumerate(["N", "CA", "C", "O", "CB"]):
            colab_atom = colab_backbone[res_idx, atom_idx]
            prx_atom = prx_backbone[res_idx, atom_idx]
            atom_diff = np.linalg.norm(colab_atom - prx_atom)
            print(f"  {name}: diff = {atom_diff:.6f}")
            if atom_diff > 0.01:
                print(f"    ColabDesign: {colab_atom}")
                print(f"    PrxteinMPNN: {prx_atom}")
