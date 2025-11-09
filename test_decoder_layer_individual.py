"""Test each decoder layer individually to isolate layer 2 issue."""

import jax
import jax.numpy as jnp
import joblib
from prxteinmpnn.io.parsing import parse_input
from prxteinmpnn.utils.data_structures import Protein
from prxteinmpnn.utils.residue_constants import atom_order
from prxteinmpnn.utils.concatenate import concatenate_neighbor_nodes
from load_weights_comprehensive import load_prxteinmpnn_with_colabdesign_weights
from compare_pure_jax import colabdesign_forward, colabdesign_decoder_layer, compare


def main():
    print("="*80)
    print("INDIVIDUAL DECODER LAYER TEST")
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

    print("\n1. Get encoder outputs and decoder context...")
    # Run ColabDesign forward to get encoder outputs
    _, colab_E, colab_E_idx, colab_context, colab_inter = colabdesign_forward(
        X_backbone, protein.mask, protein.residue_index,
        protein.chain_index, params, k_neighbors=48, return_intermediates=True
    )

    # Get PrxteinMPNN encoder outputs
    prx_edge, prx_idx, _ = model.features(
        key, protein.coordinates, protein.mask,
        protein.residue_index, protein.chain_index, None
    )

    prx_node_features = jnp.zeros((prx_edge.shape[0], model.encoder.node_feature_dim))
    mask_2d = protein.mask[:, None] * protein.mask[None, :]
    prx_mask_attend = jnp.take_along_axis(mask_2d, prx_idx, axis=1)

    for layer in model.encoder.layers:
        prx_node_features, prx_edge = layer(
            prx_node_features, prx_edge, prx_idx,
            protein.mask, prx_mask_attend
        )

    # Build decoder context
    zeros_with_edges = concatenate_neighbor_nodes(
        jnp.zeros_like(prx_node_features), prx_edge, prx_idx
    )
    prx_context = concatenate_neighbor_nodes(
        prx_node_features, zeros_with_edges, prx_idx
    )

    # Get encoder output node features from ColabDesign
    colab_h_V_encoder = colab_inter['encoder_2_h_V']

    print("\n2. Test each decoder layer individually with SAME input...")

    for layer_idx in range(3):
        print(f"\n{'='*80}")
        print(f"TESTING DECODER LAYER {layer_idx}")
        print(f"{'='*80}")

        # Use the encoder output as fresh input for each layer test
        colab_h_V_test = colab_h_V_encoder.copy()
        prx_h_V_test = prx_node_features.copy()

        # Run ColabDesign layer
        colab_h_V_output = colabdesign_decoder_layer(
            colab_h_V_test, colab_context, protein.mask, params, layer_idx
        )

        # Run PrxteinMPNN layer
        prx_h_V_output = model.decoder.layers[layer_idx](
            prx_h_V_test, prx_context, protein.mask
        )

        # Compare
        corr = compare(
            f"  Layer {layer_idx} output (fresh encoder input)",
            colab_h_V_output,
            prx_h_V_output
        )

        if corr < 0.90:
            print(f"  ⚠️  Layer {layer_idx} has LOW correlation with fresh input!")
            print(f"     This suggests a bug in layer {layer_idx} itself.")
        elif corr > 0.99:
            print(f"  ✅ Layer {layer_idx} is PERFECT with fresh input!")
        else:
            print(f"  🟡 Layer {layer_idx} is OK but not perfect.")

    print("\n" + "="*80)
    print("3. Now test with ACCUMULATED inputs (layer by layer)...")
    print("="*80)

    # Start from encoder output
    colab_h_V_accum = colab_h_V_encoder.copy()
    prx_h_V_accum = prx_node_features.copy()

    for layer_idx in range(3):
        print(f"\nDecoder Layer {layer_idx} (accumulated):")

        # Run layers
        colab_h_V_accum = colabdesign_decoder_layer(
            colab_h_V_accum, colab_context, protein.mask, params, layer_idx
        )
        prx_h_V_accum = model.decoder.layers[layer_idx](
            prx_h_V_accum, prx_context, protein.mask
        )

        # Compare
        corr = compare(
            f"  Layer {layer_idx} accumulated output",
            colab_h_V_accum,
            prx_h_V_accum
        )

        if layer_idx == 2 and corr < 0.80:
            print(f"  ⚠️  Layer 2 accumulated correlation dropped to {corr:.6f}!")
            print(f"     This confirms layer 2 has the biggest divergence.")

    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    print("If layer 2 with FRESH input has good correlation but ACCUMULATED has bad,")
    print("then the issue is accumulated error, not a bug in layer 2.")
    print("\nIf layer 2 with FRESH input also has bad correlation,")
    print("then layer 2 itself has a bug!")


if __name__ == "__main__":
    main()
