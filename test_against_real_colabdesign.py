"""Test against the ACTUAL ColabDesign library, not our reimplementation."""

import jax
import jax.numpy as jnp
import sys
sys.path.insert(0, "/tmp/ColabDesign")

# Import actual ColabDesign
from colabdesign.mpnn.model import mk_mpnn_model

from prxteinmpnn.io.parsing import parse_input
from prxteinmpnn.utils.data_structures import Protein
from prxteinmpnn.utils.residue_constants import atom_order
from load_weights_comprehensive import load_prxteinmpnn_with_colabdesign_weights
from compare_pure_jax import compare


def main():
    print("="*80)
    print("TEST AGAINST REAL COLABDESIGN LIBRARY")
    print("="*80)

    # Load protein
    pdb_path = "tests/data/1ubq.pdb"
    protein_tuple = next(parse_input(pdb_path))
    protein = Protein.from_tuple(protein_tuple)

    # Setup for ColabDesign
    backbone_indices = [atom_order["N"], atom_order["CA"], atom_order["C"], atom_order["O"]]
    X_backbone = protein.coordinates[:, backbone_indices, :]

    print("\n1. Create REAL ColabDesign model...")
    # Initialize ColabDesign model
    mpnn_model = mk_mpnn_model()
    mpnn_model.prep_inputs(pdb_filename=pdb_path)

    print(f"   ColabDesign model initialized")
    print(f"   Batch keys: {mpnn_model._inputs.keys()}")

    # Get the actual inputs ColabDesign prepared
    colab_X = mpnn_model._inputs["X"]
    colab_mask = mpnn_model._inputs["mask"]
    colab_residue_idx = mpnn_model._inputs["residue_idx"]
    colab_chain_idx = mpnn_model._inputs["chain_idx"]

    print(f"\n   ColabDesign inputs:")
    print(f"   X shape: {colab_X.shape}")
    print(f"   mask shape: {colab_mask.shape}")

    # Run actual ColabDesign forward pass
    print("\n2. Run REAL ColabDesign forward pass...")
    try:
        # Get the actual ColabDesign output
        # The model might have different methods, let me try to call it
        colab_output = mpnn_model.model.apply(
            mpnn_model.model_params,
            colab_X,
            colab_mask,
            colab_residue_idx,
            colab_chain_idx
        )
        print(f"   ColabDesign output shape: {colab_output.shape}")

    except Exception as e:
        print(f"   Error running ColabDesign: {e}")
        print("\n   Trying alternative method...")

        # Try getting just the logits
        try:
            logits = mpnn_model.get_unconditional_logits()
            print(f"   Got unconditional logits shape: {logits.shape}")
            colab_output = logits
        except Exception as e2:
            print(f"   Also failed: {e2}")
            return

    # Run PrxteinMPNN
    print("\n3. Run PrxteinMPNN...")
    key = jax.random.PRNGKey(42)
    colab_weights_path = "/tmp/ColabDesign/colabdesign/mpnn/weights/v_48_020.pkl"
    prx_model = load_prxteinmpnn_with_colabdesign_weights(colab_weights_path, key=key)

    _, prx_output = prx_model(
        protein.coordinates, protein.mask, protein.residue_index,
        protein.chain_index, "unconditional", prng_key=key
    )

    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)

    corr = compare("\nREAL ColabDesign vs PrxteinMPNN", colab_output, prx_output)

    print("\n" + "="*80)
    print("RESULT")
    print("="*80)

    if corr > 0.99:
        print(f"✅ Excellent correlation ({corr:.6f})!")
        print("   Our implementation matches the real ColabDesign!")
    elif corr > 0.97:
        print(f"🟡 Good correlation ({corr:.6f})")
        print("   Close to our manual reimplementation (0.971)")
    elif corr > 0.87:
        print(f"🟡 Similar to current result ({corr:.6f})")
        print(f"   This matches our existing 0.871 correlation")
    else:
        print(f"❌ Lower correlation ({corr:.6f})")
        print("   Something is different!")


if __name__ == "__main__":
    main()
