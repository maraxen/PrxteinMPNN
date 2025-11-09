"""Test hypothesis: Use ColabDesign edge features with PrxteinMPNN encoder/decoder.

If PrxteinMPNN's encoder and decoder are perfect, using ColabDesign's edge features
should result in much better final logits correlation (hopefully >0.90).
"""

import jax
import jax.numpy as jnp
import joblib
from prxteinmpnn.io.parsing import parse_input
from prxteinmpnn.utils.data_structures import Protein
from prxteinmpnn.utils.residue_constants import atom_order
from prxteinmpnn.utils.concatenate import concatenate_neighbor_nodes
from load_weights_comprehensive import load_prxteinmpnn_with_colabdesign_weights
from compare_pure_jax import colabdesign_forward, compare


def main():
    print("="*80)
    print("CROSS-IMPLEMENTATION TEST: ColabDesign Edges → PrxteinMPNN Encoder/Decoder")
    print("="*80)

    # Load protein
    pdb_path = "tests/data/1ubq.pdb"
    protein_tuple = next(parse_input(pdb_path))
    protein = Protein.from_tuple(protein_tuple)

    # Load model
    key = jax.random.PRNGKey(42)
    colab_weights_path = "/tmp/ColabDesign/colabdesign/mpnn/weights/v_48_020.pkl"
    model = load_prxteinmpnn_with_colabdesign_weights(colab_weights_path, key=key)
    params = joblib.load(colab_weights_path)['model_state_dict']

    # Extract backbone
    backbone_indices = [atom_order["N"], atom_order["CA"], atom_order["C"], atom_order["O"]]
    X_backbone = protein.coordinates[:, backbone_indices, :]

    print("\n1. Get ColabDesign edge features and final logits (ground truth)...")
    colab_logits, colab_E, colab_E_idx, colab_context, colab_inter = colabdesign_forward(
        X_backbone, protein.mask, protein.residue_index,
        protein.chain_index, params, k_neighbors=48, return_intermediates=True
    )

    # Get edge features after W_e projection (this is what PrxteinMPNN features returns)
    w_e = params['protein_mpnn/~/W_e']['w']
    b_e = params['protein_mpnn/~/W_e']['b']
    colab_edge_features_projected = colab_E @ w_e + b_e

    print(f"   ColabDesign edge features shape: {colab_edge_features_projected.shape}")
    print(f"   ColabDesign neighbor indices shape: {colab_E_idx.shape}")
    print(f"   ColabDesign final logits shape: {colab_logits.shape}")

    print("\n2. Use ColabDesign edge features with PrxteinMPNN encoder/decoder...")

    # Run PrxteinMPNN encoder with ColabDesign's edge features
    prx_node_features = jnp.zeros((colab_edge_features_projected.shape[0], model.encoder.node_feature_dim))
    mask_2d = protein.mask[:, None] * protein.mask[None, :]
    prx_mask_attend = jnp.take_along_axis(mask_2d, colab_E_idx, axis=1)

    prx_edge_features = colab_edge_features_projected  # Use ColabDesign's edge features!
    prx_neighbor_indices = colab_E_idx  # Use ColabDesign's neighbor indices!

    # Run encoder
    for layer in model.encoder.layers:
        prx_node_features, prx_edge_features = layer(
            prx_node_features, prx_edge_features, prx_neighbor_indices,
            protein.mask, prx_mask_attend
        )

    # Build decoder context
    zeros_with_edges = concatenate_neighbor_nodes(
        jnp.zeros_like(prx_node_features), prx_edge_features, prx_neighbor_indices
    )
    prx_context = concatenate_neighbor_nodes(
        prx_node_features, zeros_with_edges, prx_neighbor_indices
    )

    # Run decoder
    prx_node_features_dec = prx_node_features
    for layer in model.decoder.layers:
        prx_node_features_dec = layer(prx_node_features_dec, prx_context, protein.mask)

    # Get final logits
    prx_logits_from_colab_edges = jax.vmap(model.w_out)(prx_node_features_dec)

    print("\n3. Also run fully PrxteinMPNN for comparison...")
    _, prx_logits_full = model(
        protein.coordinates, protein.mask, protein.residue_index,
        protein.chain_index, "unconditional", prng_key=key
    )

    print("\n" + "="*80)
    print("COMPARISONS")
    print("="*80)

    corr_hybrid = compare(
        "\n1. ColabDesign vs PrxteinMPNN (using ColabDesign edges)",
        colab_logits,
        prx_logits_from_colab_edges
    )

    corr_original = compare(
        "\n2. ColabDesign vs PrxteinMPNN (fully PrxteinMPNN)",
        colab_logits,
        prx_logits_full
    )

    print("\n" + "="*80)
    print("HYPOTHESIS TEST RESULTS")
    print("="*80)

    improvement = corr_hybrid - corr_original
    print(f"Using ColabDesign edges:  {corr_hybrid:.6f}")
    print(f"Using PrxteinMPNN edges:  {corr_original:.6f}")
    print(f"Improvement:               {improvement:+.6f}")

    if corr_hybrid > 0.90:
        print("\n✅ SUCCESS! Using ColabDesign edges gives >0.90 correlation!")
        print("   This PROVES that PrxteinMPNN's encoder and decoder are PERFECT.")
        print("   The ONLY issue is the initial edge features (0.971 correlation).")
        return True
    elif improvement > 0.05:
        print(f"\n🟡 SIGNIFICANT IMPROVEMENT (+{improvement:.6f})!")
        print("   This confirms encoder/decoder are mostly correct.")
        print("   The edge features are the main issue.")
        return False
    else:
        print(f"\n❌ NO IMPROVEMENT! (only +{improvement:.6f})")
        print("   This suggests encoder or decoder also have issues,")
        print("   not just edge features.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
