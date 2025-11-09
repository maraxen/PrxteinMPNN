"""Verify C-beta is used correctly in RBF computation."""

import jax.numpy as jnp
from prxteinmpnn.io.parsing import parse_input
from prxteinmpnn.utils.data_structures import Protein
from prxteinmpnn.utils.residue_constants import atom_order
from prxteinmpnn.utils.coordinates import compute_backbone_coordinates
from prxteinmpnn.utils.radial_basis import BACKBONE_PAIRS
from compare_pure_jax import colabdesign_get_cb, compare


def main():
    print("="*80)
    print("C-BETA USAGE IN RBF COMPUTATION")
    print("="*80)

    # Load protein
    pdb_path = "tests/data/1ubq.pdb"
    protein_tuple = next(parse_input(pdb_path))
    protein = Protein.from_tuple(protein_tuple)

    # Extract backbone atoms for ColabDesign
    backbone_indices = [atom_order["N"], atom_order["CA"], atom_order["C"], atom_order["O"]]
    X_backbone = protein.coordinates[:, backbone_indices, :]

    print("\n1. ColabDesign atom ordering...")
    Y = X_backbone.swapaxes(0, 1)  # (4, L, 3)
    print(f"   Input Y shape: {Y.shape}")
    print(f"   Y[0] = N (all nitrogen atoms)")
    print(f"   Y[1] = CA (all alpha carbons)")
    print(f"   Y[2] = C (all carbons)")
    print(f"   Y[3] = O (all oxygens)")

    # Compute and add CB
    Cb = colabdesign_get_cb(Y)
    Y = jnp.concatenate([Y, Cb[None]], 0)
    print(f"\n   After adding computed CB:")
    print(f"   Y shape: {Y.shape}")
    print(f"   Y[0] = N")
    print(f"   Y[1] = CA")
    print(f"   Y[2] = C")
    print(f"   Y[3] = O")
    print(f"   Y[4] = CB (computed)")

    print("\n2. PrxteinMPNN atom ordering...")
    backbone_prx = compute_backbone_coordinates(protein.coordinates)
    print(f"   Backbone shape: {backbone_prx.shape}")
    print(f"   backbone[:, 0, :] = N")
    print(f"   backbone[:, 1, :] = CA")
    print(f"   backbone[:, 2, :] = C")
    print(f"   backbone[:, 3, :] = O")
    print(f"   backbone[:, 4, :] = CB (computed)")

    print("\n3. Check if CB values match...")
    colab_cb = Y[4]  # (L, 3)
    prx_cb = backbone_prx[:, 4, :]  # (L, 3)

    compare("C-beta values", colab_cb, prx_cb)

    print("\n4. Check BACKBONE_PAIRS usage...")
    print(f"   BACKBONE_PAIRS has {len(BACKBONE_PAIRS)} atom pairs")
    print(f"   First 5 pairs:")
    for i in range(5):
        pair = BACKBONE_PAIRS[i]
        atoms = ["N", "CA", "C", "O", "CB"]
        print(f"     Pair {i}: [{pair[0]}, {pair[1]}] = {atoms[pair[0]]}-{atoms[pair[1]]}")

    # Check specifically for CB-CB pair
    cb_pairs = BACKBONE_PAIRS[BACKBONE_PAIRS[:, 0] == 4]
    print(f"\n   Pairs involving CB (index 4):")
    for pair in cb_pairs:
        atoms = ["N", "CA", "C", "O", "CB"]
        print(f"     [{pair[0]}, {pair[1]}] = {atoms[pair[0]]}-{atoms[pair[1]]}")

    # Check if CB-CB is at the expected position
    cb_cb_idx = jnp.where((BACKBONE_PAIRS[:, 0] == 4) & (BACKBONE_PAIRS[:, 1] == 4))[0]
    if len(cb_cb_idx) > 0:
        print(f"\n   ✅ CB-CB pair found at index {cb_cb_idx[0]}")
    else:
        print(f"\n   ❌ CB-CB pair NOT found!")

    # Compare with ColabDesign's edges_pairs
    print("\n5. ColabDesign edges_pairs (first 5)...")
    edges_pairs = jnp.array([[1,1],[0,0],[2,2],[3,3],[4,4],
                              [1,0],[1,2],[1,3],[1,4],[0,2],
                              [0,3],[0,4],[4,2],[4,3],[3,2],
                              [0,1],[2,1],[3,1],[4,1],[2,0],
                              [3,0],[4,0],[2,4],[3,4],[2,3]])

    for i in range(5):
        pair = edges_pairs[i]
        atoms = ["N", "CA", "C", "O", "CB"]
        print(f"     Pair {i}: [{pair[0]}, {pair[1]}] = {atoms[pair[0]]}-{atoms[pair[1]]}")

    print("\n6. Check if BACKBONE_PAIRS matches ColabDesign edges_pairs...")
    match = jnp.array_equal(BACKBONE_PAIRS, edges_pairs)
    print(f"   Exact match: {match}")

    if not match:
        print("\n   ⚠️  BACKBONE_PAIRS does NOT match edges_pairs!")
        print("   This could explain the 0.971 correlation!")

        # Find differences
        for i in range(min(len(BACKBONE_PAIRS), len(edges_pairs))):
            if not jnp.array_equal(BACKBONE_PAIRS[i], edges_pairs[i]):
                atoms = ["N", "CA", "C", "O", "CB"]
                print(f"     Pair {i} differs:")
                print(f"       PrxteinMPNN: [{BACKBONE_PAIRS[i][0]}, {BACKBONE_PAIRS[i][1]}] = {atoms[BACKBONE_PAIRS[i][0]]}-{atoms[BACKBONE_PAIRS[i][1]]}")
                print(f"       ColabDesign: [{edges_pairs[i][0]}, {edges_pairs[i][1]}] = {atoms[edges_pairs[i][0]]}-{atoms[edges_pairs[i][1]]}")
    else:
        print("   ✅ BACKBONE_PAIRS matches ColabDesign exactly!")


if __name__ == "__main__":
    main()
