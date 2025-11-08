"""Test if neighbor indices match between PrxteinMPNN and ColabDesign."""

import jax
import jax.numpy as jnp
import numpy as np

from prxteinmpnn.io.parsing import parse_input
from prxteinmpnn.utils.data_structures import Protein
from colabdesign.mpnn.model import mk_mpnn_model
from load_weights_comprehensive import load_prxteinmpnn_with_colabdesign_weights


def main():
    print("="*80)
    print("TESTING NEIGHBOR INDICES")
    print("="*80)

    # Load test structure
    pdb_path = "tests/data/1ubq.pdb"
    protein_tuple = next(parse_input(pdb_path))
    protein = Protein.from_tuple(protein_tuple)

    # Load models
    key = jax.random.PRNGKey(42)
    colab_weights_path = "/tmp/ColabDesign/colabdesign/mpnn/weights/v_48_020.pkl"

    prx_model = load_prxteinmpnn_with_colabdesign_weights(colab_weights_path, key=key)
    colab_model = mk_mpnn_model(model_name="v_48_020", weights="original", seed=42)
    colab_model.prep_inputs(pdb_filename=pdb_path)

    print("\n1. Extract PrxteinMPNN neighbor indices...")
    prx_edge_features, prx_neighbor_indices, _ = prx_model.features(
        key,
        protein.coordinates,
        protein.mask,
        protein.residue_index,
        protein.chain_index,
        None,
    )
    print(f"   Neighbor indices shape: {prx_neighbor_indices.shape}")
    print(f"   First residue neighbors (first 10): {prx_neighbor_indices[0, :10]}")
    print(f"   Neighbor indices dtype: {prx_neighbor_indices.dtype}")

    print("\n2. Extract ColabDesign neighbor indices...")
    # ColabDesign stores this in _inputs
    colab_neighbor_indices = colab_model._inputs['E_idx']
    print(f"   Neighbor indices shape: {colab_neighbor_indices.shape}")
    print(f"   First residue neighbors (first 10): {colab_neighbor_indices[0, :10]}")
    print(f"   Neighbor indices dtype: {colab_neighbor_indices.dtype}")

    print("\n3. Compare neighbor indices...")
    # Check if shapes match
    if prx_neighbor_indices.shape != colab_neighbor_indices.shape:
        print(f"   ❌ SHAPES DIFFER!")
        print(f"      PrxteinMPNN: {prx_neighbor_indices.shape}")
        print(f"      ColabDesign: {colab_neighbor_indices.shape}")
    else:
        print(f"   ✅ Shapes match: {prx_neighbor_indices.shape}")

    # Check if values match
    matches = (prx_neighbor_indices == colab_neighbor_indices).all()
    if matches:
        print(f"   ✅ ALL neighbor indices match!")
    else:
        num_diff = (prx_neighbor_indices != colab_neighbor_indices).sum()
        total = prx_neighbor_indices.size
        pct_diff = 100 * num_diff / total
        print(f"   ❌ Neighbor indices differ!")
        print(f"      {num_diff}/{total} ({pct_diff:.2f}%) indices are different")

        # Find first difference
        diff_mask = prx_neighbor_indices != colab_neighbor_indices
        diff_pos = jnp.argwhere(diff_mask)[0]
        i, j = diff_pos
        print(f"\n      First difference at position [{i}, {j}]:")
        print(f"        PrxteinMPNN: {prx_neighbor_indices[i, j]}")
        print(f"        ColabDesign: {colab_neighbor_indices[i, j]}")

        # Show full neighbor list for that residue
        print(f"\n      Full neighbor list for residue {i}:")
        print(f"        PrxteinMPNN: {prx_neighbor_indices[i]}")
        print(f"        ColabDesign: {colab_neighbor_indices[i]}")

    # Check K value
    print(f"\n4. K-neighbors value:")
    print(f"   PrxteinMPNN: {prx_neighbor_indices.shape[1]}")
    print(f"   ColabDesign: {colab_neighbor_indices.shape[1]}")


if __name__ == "__main__":
    main()
