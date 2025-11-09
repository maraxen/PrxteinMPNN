"""Test if approx_min_k vs top_k give different neighbor orderings."""

import jax
import jax.numpy as jnp
from prxteinmpnn.io.parsing import parse_input
from prxteinmpnn.utils.data_structures import Protein
from prxteinmpnn.utils.residue_constants import atom_order
from prxteinmpnn.utils.coordinates import compute_backbone_coordinates, compute_backbone_distance
from compare_pure_jax import colabdesign_get_edge_idx, colabdesign_get_cb, compare


def main():
    print("="*80)
    print("NEIGHBOR ORDERING: approx_min_k vs top_k")
    print("="*80)

    # Load protein
    pdb_path = "tests/data/1ubq.pdb"
    protein_tuple = next(parse_input(pdb_path))
    protein = Protein.from_tuple(protein_tuple)

    # Extract backbone
    backbone_indices = [atom_order["N"], atom_order["CA"], atom_order["C"], atom_order["O"]]
    X_backbone = protein.coordinates[:, backbone_indices, :]

    print("\n1. ColabDesign neighbor indices (approx_min_k)...")
    # ColabDesign method
    Y = X_backbone.swapaxes(0, 1)
    Cb = colabdesign_get_cb(Y)
    Y = jnp.concatenate([Y, Cb[None]], 0)

    colab_neighbors = colabdesign_get_edge_idx(Y[1], protein.mask, 48)
    print(f"   Shape: {colab_neighbors.shape}")
    print(f"   Sample (residue 0, first 10): {colab_neighbors[0, :10]}")

    print("\n2. PrxteinMPNN neighbor indices (top_k)...")
    # PrxteinMPNN method
    backbone = compute_backbone_coordinates(protein.coordinates)
    distances = compute_backbone_distance(backbone)

    distances_masked = jnp.where(
        (protein.mask[:, None] * protein.mask[None, :]).astype(bool),
        distances, jnp.inf
    )

    top_k = jax.jit(jax.lax.top_k, static_argnames=("k",))
    _, prx_neighbors = top_k(-distances_masked, 48)
    prx_neighbors = jnp.array(prx_neighbors, dtype=jnp.int32)

    print(f"   Shape: {prx_neighbors.shape}")
    print(f"   Sample (residue 0, first 10): {prx_neighbors[0, :10]}")

    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)

    # Check if they're exactly equal
    exactly_equal = jnp.all(colab_neighbors == prx_neighbors)
    print(f"\nExactly equal: {exactly_equal}")

    if not exactly_equal:
        # Count differences
        different_positions = (colab_neighbors != prx_neighbors).sum()
        total_positions = colab_neighbors.size
        print(f"Different positions: {different_positions} / {total_positions} ({100*different_positions/total_positions:.2f}%)")

        # Check if they contain the same neighbors (just different order)
        same_neighbors = []
        for i in range(colab_neighbors.shape[0]):
            colab_set = set(colab_neighbors[i].tolist())
            prx_set = set(prx_neighbors[i].tolist())
            same_neighbors.append(colab_set == prx_set)

        all_same = all(same_neighbors)
        print(f"Same neighbors (different order): {all_same}")

        if not all_same:
            num_different = sum(1 for x in same_neighbors if not x)
            print(f"Residues with different neighbor sets: {num_different} / {len(same_neighbors)}")

    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)

    if exactly_equal:
        print("✅ Neighbor orderings are IDENTICAL!")
        print("   This is NOT the source of the 0.971 correlation.")
    else:
        print("❌ Neighbor orderings are DIFFERENT!")
        print("   This COULD explain the 0.971 correlation!")
        print("   approx_min_k and top_k give different results!")


if __name__ == "__main__":
    main()
