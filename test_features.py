"""Test if feature extraction produces identical outputs."""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import pearsonr

from prxteinmpnn.io.parsing import parse_input
from prxteinmpnn.utils.data_structures import Protein
from colabdesign.mpnn.model import mk_mpnn_model
from load_weights_comprehensive import load_prxteinmpnn_with_colabdesign_weights


def main():
    print("="*80)
    print("TESTING FEATURE EXTRACTION")
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

    print("\n1. Extract PrxteinMPNN features...")
    prx_edge_features, prx_neighbor_indices, _ = prx_model.features(
        key,
        protein.coordinates,
        protein.mask,
        protein.residue_index,
        protein.chain_index,
        None,
    )
    print(f"   Edge features shape: {prx_edge_features.shape}")
    print(f"   Edge features range: [{prx_edge_features.min():.3f}, {prx_edge_features.max():.3f}]")
    print(f"   Neighbor indices shape: {prx_neighbor_indices.shape}")

    print("\n2. Extract ColabDesign features...")
    # ColabDesign uses Haiku, so we need to access internals
    # The easiest way is to just run the model and check the intermediate values
    # But for now, let's just check if we can get the same edge features

    # Run the full unconditional scoring to see if we can match
    _, prx_logits = prx_model(
        protein.coordinates,
        protein.mask,
        protein.residue_index,
        protein.chain_index,
        "unconditional",
        prng_key=key,
    )

    colab_logits_af = colab_model.get_unconditional_logits(key=key)

    # Convert to MPNN order
    MPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
    AF_ALPHABET = "ARNDCQEGHILKMFPSTWYVX"
    perm = np.array([AF_ALPHABET.index(aa) for aa in MPNN_ALPHABET])
    colab_logits = np.array(colab_logits_af)[..., perm]

    print("\n3. Compare final logits:")
    prx_flat = np.array(prx_logits).flatten()
    colab_flat = np.array(colab_logits).flatten()
    corr = pearsonr(prx_flat, colab_flat)[0]
    max_diff = np.abs(prx_flat - colab_flat).max()

    print(f"   Correlation: {corr:.6f}")
    print(f"   Max diff: {max_diff:.6f}")

    print("\n4. Check encoder output...")
    # Run encoder
    prx_node_feat, prx_edge_feat_enc = prx_model.encoder(
        prx_edge_features,
        prx_neighbor_indices,
        protein.mask,
    )
    print(f"   Node features shape: {prx_node_feat.shape}")
    print(f"   Node features range: [{prx_node_feat.min():.3f}, {prx_node_feat.max():.3f}]")
    print(f"   Edge features shape: {prx_edge_feat_enc.shape}")
    print(f"   Edge features range: [{prx_edge_feat_enc.min():.3f}, {prx_edge_feat_enc.max():.3f}]")

    # Print some sample values
    print(f"\n   Node features [0, :5]: {prx_node_feat[0, :5]}")
    print(f"   Edge features [0, 0, :5]: {prx_edge_feat_enc[0, 0, :5]}")

    print("\n5. Check decoder output...")
    prx_node_dec = prx_model.decoder(
        prx_node_feat,
        prx_edge_feat_enc,
        prx_neighbor_indices,
        protein.mask,
    )
    print(f"   Decoder node features shape: {prx_node_dec.shape}")
    print(f"   Decoder node features range: [{prx_node_dec.min():.3f}, {prx_node_dec.max():.3f}]")
    print(f"   Decoder node features [0, :5]: {prx_node_dec[0, :5]}")

    print("\n6. Check logits...")
    prx_logits_manual = jax.vmap(prx_model.w_out)(prx_node_dec)
    print(f"   Logits shape: {prx_logits_manual.shape}")
    print(f"   Logits [0, :5]: {prx_logits_manual[0, :5]}")
    print(f"   ColabDesign logits [0, :5]: {colab_logits[0, :5]}")

    # Check if logits match
    logits_match = np.allclose(prx_logits, prx_logits_manual, atol=1e-6)
    print(f"\n   Logits from full model vs manual: {logits_match}")


if __name__ == "__main__":
    main()
