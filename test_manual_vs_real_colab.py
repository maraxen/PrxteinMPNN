"""Compare our manual colabdesign_forward() vs REAL ColabDesign library."""

import jax
import jax.numpy as jnp
import joblib
import sys
sys.path.insert(0, "/tmp/ColabDesign")

from colabdesign.mpnn.model import mk_mpnn_model

from prxteinmpnn.io.parsing import parse_input
from prxteinmpnn.utils.data_structures import Protein
from prxteinmpnn.utils.residue_constants import atom_order
from compare_pure_jax import colabdesign_forward, compare


def main():
    print("="*80)
    print("MANUAL COLABDESIGN IMPLEMENTATION vs REAL COLABDESIGN")
    print("="*80)

    # Load protein
    pdb_path = "tests/data/1ubq.pdb"
    protein_tuple = next(parse_input(pdb_path))
    protein = Protein.from_tuple(protein_tuple)

    # Extract backbone
    backbone_indices = [atom_order["N"], atom_order["CA"], atom_order["C"], atom_order["O"]]
    X_backbone = protein.coordinates[:, backbone_indices, :]

    print("\n1. Run REAL ColabDesign...")
    mpnn_model = mk_mpnn_model()
    mpnn_model.prep_inputs(pdb_filename=pdb_path)
    real_colab_logits = mpnn_model.get_unconditional_logits()

    print(f"   Real ColabDesign logits shape: {real_colab_logits.shape}")
    print(f"   Sample (residue 0): {real_colab_logits[0, :5]}")

    print("\n2. Run our MANUAL colabdesign_forward()...")
    colab_weights_path = "/tmp/ColabDesign/colabdesign/mpnn/weights/v_48_020.pkl"
    params = joblib.load(colab_weights_path)['model_state_dict']

    manual_colab_logits, _, _, _, _ = colabdesign_forward(
        X_backbone, protein.mask, protein.residue_index,
        protein.chain_index, params, k_neighbors=48, return_intermediates=True
    )

    print(f"   Manual ColabDesign logits shape: {manual_colab_logits.shape}")
    print(f"   Sample (residue 0): {manual_colab_logits[0, :5]}")

    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)

    corr = compare(
        "\nREAL ColabDesign vs MANUAL colabdesign_forward()",
        real_colab_logits,
        manual_colab_logits
    )

    print("\n" + "="*80)
    print("CRITICAL FINDING")
    print("="*80)

    if corr > 0.99:
        print(f"✅ Our manual implementation is CORRECT ({corr:.6f})")
        print("   It matches the real ColabDesign perfectly!")
    elif corr > 0.85:
        print(f"🟡 Our manual implementation is CLOSE ({corr:.6f})")
        print("   But there are some differences from real ColabDesign")
    else:
        print(f"❌ Our manual implementation is WRONG! ({corr:.6f})")
        print("   We've been comparing against a BUGGY reimplementation!")
        print("\n   This means:")
        print("   - PrxteinMPNN might actually be more correct than we thought")
        print("   - Our 0.871 correlation was against a buggy reference")
        print("   - The REAL target should be the actual ColabDesign library")


if __name__ == "__main__":
    main()
