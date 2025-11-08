"""
Comprehensive layer-by-layer debugging to find where PrxteinMPNN and ColabDesign diverge.

This script compares outputs at every stage:
1. Feature extraction (RBF, positional encoding, edge features)
2. After each encoder layer (node_features, edge_features)
3. Context tensor for decoder
4. After each decoder layer (node_features)
5. Final logits

For each comparison, we compute:
- Pearson correlation
- Max absolute difference
- Mean absolute difference
"""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import pearsonr
from prxteinmpnn.io.parsing import parse_input
from prxteinmpnn.utils.data_structures import Protein
from colabdesign.mpnn.model import mk_mpnn_model
from load_weights_comprehensive import load_prxteinmpnn_with_colabdesign_weights


def compare_arrays(name, prx_array, colab_array, verbose=True):
    """Compare two arrays and print statistics."""
    prx_flat = np.array(prx_array).flatten()
    colab_flat = np.array(colab_array).flatten()

    # Handle NaN/Inf
    valid_mask = np.isfinite(prx_flat) & np.isfinite(colab_flat)
    if not valid_mask.all():
        print(f"  ⚠️  {name}: Contains {(~valid_mask).sum()} NaN/Inf values")
        prx_flat = prx_flat[valid_mask]
        colab_flat = colab_flat[valid_mask]

    if len(prx_flat) == 0:
        print(f"  ❌ {name}: No valid values to compare!")
        return 0.0

    corr = pearsonr(prx_flat, colab_flat)[0] if len(prx_flat) > 1 else 0.0
    max_diff = np.abs(prx_flat - colab_flat).max()
    mean_diff = np.abs(prx_flat - colab_flat).mean()

    # Determine status
    if corr > 0.99 and max_diff < 0.001:
        status = "✅"
    elif corr > 0.90:
        status = "🟡"
    else:
        status = "❌"

    if verbose:
        print(f"  {status} {name}:")
        print(f"     Correlation: {corr:.6f}")
        print(f"     Max diff: {max_diff:.6f}")
        print(f"     Mean diff: {mean_diff:.6f}")
        print(f"     Shapes: PrxteinMPNN {np.array(prx_array).shape}, ColabDesign {np.array(colab_array).shape}")

    return corr


def extract_colabdesign_intermediates(colab_model, key):
    """
    Extract intermediate values from ColabDesign by manually running forward pass.

    This is necessary because Haiku doesn't expose intermediate values easily.
    We'll manually run the forward pass step by step.
    """
    # Get inputs
    I = colab_model._inputs

    # Run features extraction
    # This is inside the Haiku model, so we need to use the applied function
    params = colab_model._model_params

    # Create a modified forward pass that returns intermediates
    def forward_with_intermediates(params, key, I):
        """Modified forward pass that returns all intermediate values."""
        import haiku as hk

        def _model_fn(I):
            # Get the model instance
            from colabdesign.mpnn.modules import RunModel
            model = RunModel(d_hidden=128, d_in=128, n_layers=3)

            # Features
            E, E_idx = model.features(I)
            h_V = jnp.zeros((E.shape[0], E.shape[-1]))
            h_E = model.W_e(E)

            intermediates = {
                'features_E': E,
                'features_E_idx': E_idx,
                'h_E_initial': h_E,
                'h_V_initial': h_V,
            }

            # Encoder
            mask_attend = jnp.take_along_axis(I["mask"][:,None] * I["mask"][None,:], E_idx, 1)

            for i, layer in enumerate(model.encoder_layers):
                h_V, h_E = layer(h_V, h_E, E_idx, I["mask"], mask_attend)
                intermediates[f'encoder_{i}_h_V'] = h_V
                intermediates[f'encoder_{i}_h_E'] = h_E

            # Build decoder context
            from colabdesign.mpnn.utils import cat_neighbors_nodes
            h_EX_encoder = cat_neighbors_nodes(jnp.zeros_like(h_V), h_E, E_idx)
            h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)

            intermediates['h_EX_encoder'] = h_EX_encoder
            intermediates['h_EXV_encoder'] = h_EXV_encoder

            # Decoder
            h_V_dec = h_V
            for i, layer in enumerate(model.decoder_layers):
                h_V_dec = layer(h_V_dec, h_EXV_encoder, I["mask"])
                intermediates[f'decoder_{i}_h_V'] = h_V_dec

            # Logits
            logits = model.W_out(h_V_dec)
            intermediates['logits'] = logits

            return intermediates

        # Transform and apply
        transformed = hk.transform(_model_fn)
        return transformed.apply(params, key, I)

    # Run the modified forward pass
    intermediates = forward_with_intermediates(params, key, I)
    return intermediates


def main():
    print("=" * 80)
    print("COMPREHENSIVE LAYER-BY-LAYER DEBUGGING")
    print("=" * 80)

    # Load test structure
    print("\n1. Loading test structure...")
    pdb_path = "tests/data/1ubq.pdb"
    protein_tuple = next(parse_input(pdb_path))
    protein = Protein.from_tuple(protein_tuple)
    seq_len = int(protein.mask.sum())
    print(f"   Loaded {seq_len} residues")

    # Load models
    print("\n2. Loading models...")
    key = jax.random.PRNGKey(42)
    colab_weights_path = "/tmp/ColabDesign/colabdesign/mpnn/weights/v_48_020.pkl"

    prx_model = load_prxteinmpnn_with_colabdesign_weights(colab_weights_path, key=key)
    colab_model = mk_mpnn_model(model_name="v_48_020", weights="original", seed=42)
    colab_model.prep_inputs(pdb_filename=pdb_path)
    print("   ✅ Models loaded")

    # Extract PrxteinMPNN intermediates
    print("\n3. Running PrxteinMPNN forward pass...")

    # Feature extraction
    prx_edge_features, prx_neighbor_indices, _ = prx_model.features(
        key,
        protein.coordinates,
        protein.mask,
        protein.residue_index,
        protein.chain_index,
        None,
    )
    print(f"   Features extracted: edge_features {prx_edge_features.shape}")

    # Encoder
    prx_node_features, prx_edge_features_enc = prx_model.encoder(
        prx_edge_features,
        prx_neighbor_indices,
        protein.mask,
    )
    print(f"   Encoder done: node_features {prx_node_features.shape}")

    # Decoder (unconditional)
    prx_node_features_dec = prx_model.decoder(
        prx_node_features,
        prx_edge_features_enc,
        prx_neighbor_indices,
        protein.mask,
    )
    print(f"   Decoder done: node_features {prx_node_features_dec.shape}")

    # Logits
    prx_logits = jax.vmap(prx_model.w_out)(prx_node_features_dec)
    print(f"   Logits: {prx_logits.shape}")

    # Extract ColabDesign intermediates
    print("\n4. Running ColabDesign forward pass...")
    try:
        colab_intermediates = extract_colabdesign_intermediates(colab_model, key)
        print("   ✅ ColabDesign intermediates extracted")
    except Exception as e:
        print(f"   ❌ Failed to extract ColabDesign intermediates: {e}")
        print("   Falling back to final logits only...")

        # Just get final logits
        colab_logits_af = colab_model.get_unconditional_logits(key=key)

        # Convert to MPNN order
        MPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
        AF_ALPHABET = "ARNDCQEGHILKMFPSTWYVX"
        perm = np.array([AF_ALPHABET.index(aa) for aa in MPNN_ALPHABET])
        colab_logits = np.array(colab_logits_af)[..., perm]

        print("\n5. FINAL LOGITS COMPARISON:")
        compare_arrays("Final logits", prx_logits, colab_logits)
        return

    # Compare all intermediates
    print("\n" + "=" * 80)
    print("5. DETAILED COMPARISONS")
    print("=" * 80)

    # Feature extraction
    print("\n📊 FEATURE EXTRACTION:")
    compare_arrays("Edge features (after embedding)", prx_edge_features, colab_intermediates['h_E_initial'])

    # Encoder outputs
    print("\n📊 ENCODER:")
    # Note: We need to extract per-layer outputs from PrxteinMPNN encoder
    # For now, just compare final outputs
    compare_arrays("Encoder final node features", prx_node_features, colab_intermediates['encoder_2_h_V'])
    compare_arrays("Encoder final edge features", prx_edge_features_enc, colab_intermediates['encoder_2_h_E'])

    # Decoder context
    print("\n📊 DECODER CONTEXT:")
    # Need to reconstruct context from PrxteinMPNN
    from prxteinmpnn.model.features import concatenate_neighbor_nodes
    prx_zeros_with_edges = concatenate_neighbor_nodes(
        jnp.zeros_like(prx_node_features),
        prx_edge_features_enc,
        prx_neighbor_indices,
    )
    prx_context = concatenate_neighbor_nodes(
        prx_node_features,
        prx_zeros_with_edges,
        prx_neighbor_indices,
    )
    compare_arrays("Decoder context tensor", prx_context, colab_intermediates['h_EXV_encoder'])

    # Decoder outputs
    print("\n📊 DECODER:")
    compare_arrays("Decoder final node features", prx_node_features_dec, colab_intermediates['decoder_2_h_V'])

    # Final logits
    print("\n📊 FINAL LOGITS:")
    # Convert ColabDesign logits to MPNN order
    colab_logits_af = colab_intermediates['logits']
    MPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
    AF_ALPHABET = "ARNDCQEGHILKMFPSTWYVX"
    perm = np.array([AF_ALPHABET.index(aa) for aa in MPNN_ALPHABET])
    colab_logits = np.array(colab_logits_af)[..., perm]

    final_corr = compare_arrays("Final logits", prx_logits, colab_logits)

    # Summary
    print("\n" + "=" * 80)
    print("6. SUMMARY")
    print("=" * 80)
    print(f"Final logits correlation: {final_corr:.4f}")
    if final_corr < 0.90:
        print("❌ FAIL: Need to investigate divergence points above")
        print("\nNext steps:")
        print("  1. Check which stage has the lowest correlation")
        print("  2. Add per-layer debugging for that stage")
        print("  3. Check scaling factors, normalizations, activations")
    else:
        print("✅ SUCCESS: Correlation >= 0.90!")


if __name__ == "__main__":
    main()
