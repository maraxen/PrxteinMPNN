"""Detailed investigation of real ColabDesign output."""

import jax
import jax.numpy as jnp
import sys
sys.path.insert(0, "/tmp/ColabDesign")

from colabdesign.mpnn.model import mk_mpnn_model

from prxteinmpnn.io.parsing import parse_input
from prxteinmpnn.utils.data_structures import Protein
from load_weights_comprehensive import load_prxteinmpnn_with_colabdesign_weights


def main():
    print("="*80)
    print("DETAILED INVESTIGATION OF REAL COLABDESIGN")
    print("="*80)

    # Load protein
    pdb_path = "tests/data/1ubq.pdb"
    protein_tuple = next(parse_input(pdb_path))
    protein = Protein.from_tuple(protein_tuple)

    print("\n1. Initialize REAL ColabDesign...")
    mpnn_model = mk_mpnn_model()
    mpnn_model.prep_inputs(pdb_filename=pdb_path)

    # Get unconditional logits
    colab_logits = mpnn_model.get_unconditional_logits()
    print(f"   Logits shape: {colab_logits.shape}")
    print(f"   Logits range: [{jnp.min(colab_logits):.3f}, {jnp.max(colab_logits):.3f}]")
    print(f"   Sample (residue 0, first 5): {colab_logits[0, :5]}")

    # Try to get more info
    print("\n2. Check ColabDesign model structure...")
    print(f"   Model type: {type(mpnn_model)}")
    print(f"   Has '_model_params': {hasattr(mpnn_model, '_model_params')}")
    print(f"   Has '_params': {hasattr(mpnn_model, '_params')}")

    # Check if there's temperature or sampling involved
    print("\n3. Check for temperature/sampling...")
    if hasattr(mpnn_model, '_temp'):
        print(f"   Temperature: {mpnn_model._temp}")
    if hasattr(mpnn_model, '_sample'):
        print(f"   Sample mode: {mpnn_model._sample}")

    # Try running multiple times to see if there's randomness
    print("\n4. Run multiple times to check for randomness...")
    colab_logits_2 = mpnn_model.get_unconditional_logits()
    same = jnp.allclose(colab_logits, colab_logits_2)
    print(f"   Same output on second run: {same}")

    if not same:
        print(f"   ⚠️  Output is RANDOM! Max diff: {jnp.max(jnp.abs(colab_logits - colab_logits_2))}")

    # Get PrxteinMPNN output
    print("\n5. Run PrxteinMPNN...")
    key = jax.random.PRNGKey(42)
    colab_weights_path = "/tmp/ColabDesign/colabdesign/mpnn/weights/v_48_020.pkl"
    prx_model = load_prxteinmpnn_with_colabdesign_weights(colab_weights_path, key=key)

    _, prx_logits = prx_model(
        protein.coordinates, protein.mask, protein.residue_index,
        protein.chain_index, "unconditional", prng_key=key
    )

    print(f"   PrxteinMPNN logits shape: {prx_logits.shape}")
    print(f"   PrxteinMPNN logits range: [{jnp.min(prx_logits):.3f}, {jnp.max(prx_logits):.3f}]")
    print(f"   Sample (residue 0, first 5): {prx_logits[0, :5]}")

    # Compare statistics
    print("\n6. Compare statistics...")
    print(f"   ColabDesign mean: {jnp.mean(colab_logits):.3f}")
    print(f"   PrxteinMPNN mean: {jnp.mean(prx_logits):.3f}")

    print(f"\n   ColabDesign std: {jnp.std(colab_logits):.3f}")
    print(f"   PrxteinMPNN std: {jnp.std(prx_logits):.3f}")

    # Check if one might be softmax'd
    print("\n7. Check if one is softmaxed...")
    colab_sum_per_res = colab_logits.sum(axis=-1)
    prx_sum_per_res = prx_logits.sum(axis=-1)

    print(f"   ColabDesign sum per residue (first 5): {colab_sum_per_res[:5]}")
    print(f"   PrxteinMPNN sum per residue (first 5): {prx_sum_per_res[:5]}")

    if jnp.allclose(colab_sum_per_res, 1.0, atol=0.01):
        print("   ⚠️  ColabDesign output looks like PROBABILITIES (sums to 1)")
    if jnp.allclose(prx_sum_per_res, 1.0, atol=0.01):
        print("   ⚠️  PrxteinMPNN output looks like PROBABILITIES (sums to 1)")

    # Try comparing with softmax
    print("\n8. Try comparing after softmax...")
    from scipy.stats import pearsonr

    colab_probs = jax.nn.softmax(colab_logits, axis=-1)
    prx_probs = jax.nn.softmax(prx_logits, axis=-1)

    corr_probs = pearsonr(colab_probs.flatten(), prx_probs.flatten())[0]
    print(f"   Correlation after softmax: {corr_probs:.6f}")

    # Try log of ColabDesign if it's probabilities
    if jnp.allclose(colab_sum_per_res, 1.0, atol=0.01):
        print("\n9. Try log(ColabDesign) vs PrxteinMPNN...")
        colab_log = jnp.log(colab_logits + 1e-10)
        corr_log = pearsonr(colab_log.flatten(), prx_logits.flatten())[0]
        print(f"   Correlation log(ColabDesign) vs PrxteinMPNN: {corr_log:.6f}")


if __name__ == "__main__":
    main()
