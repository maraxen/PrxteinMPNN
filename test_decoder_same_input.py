"""Test decoder layers with IDENTICAL inputs to isolate layer bugs."""

import jax
import jax.numpy as jnp
import joblib
from prxteinmpnn.io.parsing import parse_input
from prxteinmpnn.utils.data_structures import Protein
from prxteinmpnn.utils.residue_constants import atom_order
from load_weights_comprehensive import load_prxteinmpnn_with_colabdesign_weights
from compare_pure_jax import colabdesign_decoder_layer, compare


def main():
    print("="*80)
    print("DECODER LAYERS WITH IDENTICAL INPUTS")
    print("="*80)

    # Load protein for dimensions
    pdb_path = "tests/data/1ubq.pdb"
    protein_tuple = next(parse_input(pdb_path))
    protein = Protein.from_tuple(protein_tuple)

    # Load model
    key = jax.random.PRNGKey(42)
    colab_weights_path = "/tmp/ColabDesign/colabdesign/mpnn/weights/v_48_020.pkl"
    model = load_prxteinmpnn_with_colabdesign_weights(colab_weights_path, key=key)
    params = joblib.load(colab_weights_path)['model_state_dict']

    # Create IDENTICAL random inputs for both implementations
    print("\n1. Creating identical random inputs...")
    key_input, key = jax.random.split(key)

    h_V_shared = jax.random.normal(key_input, (76, 128))  # Node features
    h_E_context_shared = jax.random.normal(jax.random.PRNGKey(43), (76, 48, 384))  # Edge context

    print(f"   Shared h_V shape: {h_V_shared.shape}")
    print(f"   Shared context shape: {h_E_context_shared.shape}")

    print("\n2. Test each decoder layer with IDENTICAL inputs...")

    for layer_idx in range(3):
        print(f"\n{'='*80}")
        print(f"DECODER LAYER {layer_idx}")
        print(f"{'='*80}")

        # ColabDesign forward
        colab_output = colabdesign_decoder_layer(
            h_V_shared, h_E_context_shared, protein.mask, params, layer_idx
        )

        # PrxteinMPNN forward
        prx_output = model.decoder.layers[layer_idx](
            h_V_shared, h_E_context_shared, protein.mask
        )

        # Compare
        corr = compare(
            f"  Layer {layer_idx} (identical inputs)",
            colab_output,
            prx_output
        )

        if corr > 0.99:
            print(f"  ✅ Layer {layer_idx} is PERFECT!")
        elif corr > 0.90:
            print(f"  🟡 Layer {layer_idx} is good but not perfect")
        else:
            print(f"  ❌ Layer {layer_idx} has divergence even with identical inputs!")
            print(f"     This indicates a bug in the layer implementation!")

    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    print("If layers show >0.99 correlation with identical inputs,")
    print("then the implementation is correct and divergence is from input differences.")
    print("\nIf layers show <0.99 correlation with identical inputs,")
    print("then there's a bug in how the layer is implemented!")


if __name__ == "__main__":
    main()
