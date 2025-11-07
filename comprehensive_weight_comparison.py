"""
Comprehensive weight comparison between PrxteinMPNN and ColabDesign.

Compare ALL weights systematically:
1. Encoder layers (3 layers)
2. Decoder layers (3 layers)
3. Output projection (W_out)
4. Embedding layers
"""

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
from prxteinmpnn.io.weights import load_model as load_prxteinmpnn
from colabdesign.mpnn.model import mk_mpnn_model

def compare_weights(prx_w, colab_w, name, transpose=False):
    """Compare two weight matrices."""
    prx_w = np.array(prx_w)
    colab_w = np.array(colab_w)

    if transpose:
        colab_w = colab_w.T

    print(f"\n{name}:")
    print(f"  PrxteinMPNN shape: {prx_w.shape}")
    print(f"  ColabDesign shape: {colab_w.shape}")

    if prx_w.shape != colab_w.shape:
        print(f"  ❌ Shape mismatch!")
        return False, 0.0

    diff = np.abs(prx_w - colab_w)
    max_diff = diff.max()
    mean_diff = diff.mean()
    corr = np.corrcoef(prx_w.flatten(), colab_w.flatten())[0, 1]

    print(f"  Max diff: {max_diff:.6f}")
    print(f"  Mean diff: {mean_diff:.6f}")
    print(f"  Correlation: {corr:.6f}")

    if max_diff < 1e-5:
        print(f"  ✅ MATCH")
        return True, corr
    else:
        print(f"  ❌ DIFFER")
        return False, corr

print("="*80)
print("Loading Models")
print("="*80)

prx_model = load_prxteinmpnn(local_path="src/prxteinmpnn/io/weights/original_v_48_020.eqx")
colab_model = mk_mpnn_model(model_name="v_48_020", weights="original", seed=42)
colab_params = colab_model._model.params

print("\n" + "="*80)
print("1. OUTPUT PROJECTION (W_out)")
print("="*80)

prx_w_out = np.array(prx_model.w_out.weight)
prx_b_out = np.array(prx_model.w_out.bias)

colab_w_out = np.array(colab_params['protein_mpnn/~/W_out']['w']).T
colab_b_out = np.array(colab_params['protein_mpnn/~/W_out']['b'])

compare_weights(prx_w_out, colab_w_out, "W_out weights")
compare_weights(prx_b_out, colab_b_out, "W_out bias")

print("\n" + "="*80)
print("2. SEQUENCE EMBEDDING (W_s)")
print("="*80)

prx_w_s = np.array(prx_model.w_s_embed.weight)
colab_w_s = np.array(colab_params['protein_mpnn/~/W_e']['w']).T

compare_weights(prx_w_s, colab_w_s, "W_s (sequence embedding)")

print("\n" + "="*80)
print("3. ENCODER LAYERS")
print("="*80)

for layer_idx in range(3):
    print(f"\n--- Encoder Layer {layer_idx} ---")

    # Access PrxteinMPNN encoder layer
    prx_enc_layer = prx_model.encoder.layers[layer_idx]

    # Access ColabDesign encoder layer params
    if layer_idx == 0:
        prefix = "protein_mpnn/~/enc_layer"
    else:
        prefix = f"protein_mpnn/~/enc_layer_{layer_idx}"

    # Compare encoder layer weights
    # Need to map the weight names correctly

    # Edge message MLP
    if hasattr(prx_enc_layer, 'edge_message_mlp'):
        mlp = prx_enc_layer.edge_message_mlp
        if hasattr(mlp, 'layers') and len(mlp.layers) > 0:
            # First linear layer
            prx_w = np.array(mlp.layers[0].weight)
            prx_b = np.array(mlp.layers[0].bias)

            colab_key_w = f"{prefix}/~/edge_message_mlp/~/enc{layer_idx}_edge_dense_W_in"
            colab_key_b = f"{prefix}/~/edge_message_mlp/~/enc{layer_idx}_edge_dense_b_in"

            if colab_key_w in colab_params:
                colab_w = np.array(colab_params[colab_key_w]['w']).T
                colab_b = np.array(colab_params[colab_key_b]['b'])

                compare_weights(prx_w, colab_w, f"Enc{layer_idx} edge_message_mlp.0.weight", transpose=False)
                compare_weights(prx_b, colab_b, f"Enc{layer_idx} edge_message_mlp.0.bias")
            else:
                print(f"  ⚠️  Could not find {colab_key_w}")
                print(f"  Available keys containing 'edge': {[k for k in colab_params.keys() if 'edge' in k.lower() and f'enc{layer_idx}' in k]}")

print("\n" + "="*80)
print("4. DECODER LAYERS")
print("="*80)

for layer_idx in range(3):
    print(f"\n--- Decoder Layer {layer_idx} ---")

    # Access PrxteinMPNN decoder layer
    prx_dec_layer = prx_model.decoder.layers[layer_idx]

    # Access ColabDesign decoder layer params
    if layer_idx == 0:
        prefix = "protein_mpnn/~/dec_layer"
    else:
        prefix = f"protein_mpnn/~/dec_layer_{layer_idx}"

    # Compare W1, W2, W3 (attention weights)
    if hasattr(prx_dec_layer, 'W_Q'):
        prx_wq = np.array(prx_dec_layer.W_Q)
        colab_key = f"{prefix}/~/dec{layer_idx}_W1"

        if colab_key in colab_params:
            colab_wq = np.array(colab_params[colab_key]['w']).T

            compare_weights(prx_wq, colab_wq, f"Dec{layer_idx} W_Q (W1)")
        else:
            print(f"  ⚠️  Could not find {colab_key}")

    if hasattr(prx_dec_layer, 'W_K'):
        prx_wk = np.array(prx_dec_layer.W_K)
        colab_key = f"{prefix}/~/dec{layer_idx}_W2"

        if colab_key in colab_params:
            colab_wk = np.array(colab_params[colab_key]['w']).T

            compare_weights(prx_wk, colab_wk, f"Dec{layer_idx} W_K (W2)")

    if hasattr(prx_dec_layer, 'W_V'):
        prx_wv = np.array(prx_dec_layer.W_V)
        colab_key = f"{prefix}/~/dec{layer_idx}_W3"

        if colab_key in colab_params:
            colab_wv = np.array(colab_params[colab_key]['w']).T

            compare_weights(prx_wv, colab_wv, f"Dec{layer_idx} W_V (W3)")

    # Dense layers in decoder
    if hasattr(prx_dec_layer, 'dense'):
        dense = prx_dec_layer.dense
        if hasattr(dense, 'weight'):
            prx_dense_w = np.array(dense.weight)
            prx_dense_b = np.array(dense.bias)

            colab_key_w = f"{prefix}/~/position_wise_feed_forward/~/dec{layer_idx}_dense_W_in"
            colab_key_b_out = f"{prefix}/~/position_wise_feed_forward/~/dec{layer_idx}_dense_W_out"

            if colab_key_w in colab_params:
                # Note: ColabDesign might have separate W_in and W_out
                print(f"  Found ColabDesign dense keys: {colab_key_w}")
            else:
                print(f"  ⚠️  Could not find dense layer weights")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print("""
This comparison should reveal if there are any weight mismatches.
If ALL weights match exactly, then the issue is in the forward pass computation itself
(e.g., different order of operations, different numerical precision, etc.)
""")
