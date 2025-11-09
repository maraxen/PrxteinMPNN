"""Test with EXACT same input (4 backbone atoms) for both implementations."""

import jax
import jax.numpy as jnp
import joblib
from prxteinmpnn.io.parsing import parse_input
from prxteinmpnn.utils.data_structures import Protein
from prxteinmpnn.utils.residue_constants import atom_order
from load_weights_comprehensive import load_prxteinmpnn_with_colabdesign_weights
from compare_pure_jax import colabdesign_features, compare


def main():
    print("="*80)
    print("TEST WITH SAME INPUT (4 backbone atoms)")
    print("="*80)

    # Load protein
    pdb_path = "tests/data/1ubq.pdb"
    protein_tuple = next(parse_input(pdb_path))
    protein = Protein.from_tuple(protein_tuple)

    # Load model and params
    key = jax.random.PRNGKey(42)
    colab_weights_path = "/tmp/ColabDesign/colabdesign/mpnn/weights/v_48_020.pkl"
    model = load_prxteinmpnn_with_colabdesign_weights(colab_weights_path, key=key)
    params = joblib.load(colab_weights_path)['model_state_dict']

    # Extract ONLY backbone atoms (N, CA, C, O)
    backbone_indices = [atom_order["N"], atom_order["CA"], atom_order["C"], atom_order["O"]]
    X_backbone_only = protein.coordinates[:, backbone_indices, :]  # (76, 4, 3)

    # Create atom37 array with ONLY backbone atoms (zeros for all others)
    backbone_only_atom37 = jnp.zeros_like(protein.coordinates)  # (76, 37, 3)
    for i, idx in enumerate(backbone_indices):
        backbone_only_atom37 = backbone_only_atom37.at[:, idx, :].set(X_backbone_only[:, i, :])

    print(f"\nInput shapes:")
    print(f"  ColabDesign input (4 atoms):     {X_backbone_only.shape}")
    print(f"  PrxteinMPNN input (atom37 format): {backbone_only_atom37.shape}")
    print(f"  Non-zero atoms in atom37: {(backbone_only_atom37.sum(axis=(1,2)) != 0).sum()}")

    print("\n1. ColabDesign with 4-atom input...")
    colab_E, colab_E_idx = colabdesign_features(
        X_backbone_only, protein.mask, protein.residue_index,
        protein.chain_index, params, k_neighbors=48
    )
    w_e = params['protein_mpnn/~/W_e']['w']
    b_e = params['protein_mpnn/~/W_e']['b']
    colab_E_final = colab_E @ w_e + b_e

    print("\n2. PrxteinMPNN with backbone-only atom37...")
    prx_edge_backbone, prx_idx_backbone, _ = model.features(
        key, backbone_only_atom37, protein.mask,
        protein.residue_index, protein.chain_index, None
    )

    print("\n3. PrxteinMPNN with FULL atom37 (for comparison)...")
    prx_edge_full, prx_idx_full, _ = model.features(
        key, protein.coordinates, protein.mask,
        protein.residue_index, protein.chain_index, None
    )

    print("\n" + "="*80)
    print("COMPARISONS")
    print("="*80)

    corr1 = compare("\n1. ColabDesign vs PrxteinMPNN (backbone-only input)", colab_E_final, prx_edge_backbone)
    corr2 = compare("\n2. ColabDesign vs PrxteinMPNN (full atom37 input)", colab_E_final, prx_edge_full)
    corr3 = compare("\n3. PrxteinMPNN backbone-only vs full atom37", prx_edge_backbone, prx_edge_full)

    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)

    if corr1 > 0.99 and corr2 < 0.98:
        print("✅ FOUND IT! Using backbone-only input gives perfect correlation!")
        print("   The issue is that full atom37 uses different atoms somewhere.")
        return True
    elif corr1 > 0.99 and corr2 > 0.99:
        print("✅ Both inputs give perfect correlation!")
        print("   The 0.971 issue must be elsewhere (bug in test code).")
        return True
    elif corr3 > 0.99:
        print("❓ PrxteinMPNN gives same output for both inputs")
        print("   But neither matches ColabDesign well.")
        return False
    else:
        print("❌ Inputs give different results - something fundamental is wrong!")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
