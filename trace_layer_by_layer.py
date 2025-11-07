"""
Layer-by-layer trace of both implementations to find where they diverge.

Strategy:
1. Prepare identical inputs
2. Extract features (edge, node)
3. Run through encoder layer-by-layer
4. Run through decoder layer-by-layer
5. Compare outputs at each step
"""

import jax
import jax.numpy as jnp
import numpy as np
from prxteinmpnn.io.parsing import parse_input
from prxteinmpnn.utils.data_structures import Protein
from prxteinmpnn.io.weights import load_model as load_prxteinmpnn
from colabdesign.mpnn.model import mk_mpnn_model

MPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
AF_ALPHABET = "ARNDCQEGHILKMFPSTWYVX"

def af_logits_to_mpnn(logits_af):
    """Convert logits from AF alphabet order to MPNN alphabet order."""
    perm = np.array([AF_ALPHABET.index(aa) for aa in MPNN_ALPHABET])
    return logits_af[..., perm]

def compare_arrays(arr1, arr2, name, tolerance=1e-5):
    """Compare two arrays and print statistics."""
    arr1 = np.array(arr1)
    arr2 = np.array(arr2)

    print(f"\n{name}:")
    print(f"  Shape: {arr1.shape} vs {arr2.shape}")

    if arr1.shape != arr2.shape:
        print(f"  ❌ Shape mismatch!")
        return False

    diff = np.abs(arr1 - arr2)
    print(f"  Max diff: {diff.max():.6f}")
    print(f"  Mean diff: {diff.mean():.6f}")
    print(f"  Correlation: {np.corrcoef(arr1.flatten(), arr2.flatten())[0, 1]:.6f}")

    if diff.max() < tolerance:
        print(f"  ✅ Match (within {tolerance})")
        return True
    else:
        print(f"  ❌ Differ (max diff {diff.max():.6f} > {tolerance})")
        return False

print("="*80)
print("SETUP: Loading models and preparing inputs")
print("="*80)

# Load test structure
pdb_path = "tests/data/1ubq.pdb"
protein_tuple = next(parse_input(pdb_path))
protein = Protein.from_tuple(protein_tuple)
print(f"✅ Loaded 1UBQ: {int(protein.mask.sum())} residues")

# Load models
prx_model = load_prxteinmpnn(local_path="src/prxteinmpnn/io/weights/original_v_48_020.eqx")
print("✅ PrxteinMPNN loaded")

colab_model = mk_mpnn_model(model_name="v_48_020", weights="original", seed=42)
colab_model.prep_inputs(pdb_filename=pdb_path)
print("✅ ColabDesign loaded")

# Prepare inputs
key = jax.random.PRNGKey(42)

print("\n" + "="*80)
print("STEP 1: Feature Extraction")
print("="*80)

print("\nPrxteinMPNN feature extraction...")
# We need to manually call the feature extraction
# Let's trace through the model's __call__ to extract features

# For now, let me check if we can access the features module
try:
    prx_features = prx_model.features
    print(f"  Features module: {type(prx_features)}")

    # Try to extract features manually
    # This requires understanding the exact input format
    print("\n  Attempting to extract edge features...")

    # The features module needs coordinates, mask, residue_index, chain_index
    # Let's see what it returns

    # Actually, let me just run the full model first to understand the flow
    print("\n  Running full PrxteinMPNN forward pass...")
    _, prx_logits = prx_model(
        protein.coordinates,
        protein.mask,
        protein.residue_index,
        protein.chain_index,
        "unconditional",
        prng_key=key,
    )
    print(f"  ✅ PrxteinMPNN logits shape: {prx_logits.shape}")
    print(f"  Logits range: [{prx_logits.min():.3f}, {prx_logits.max():.3f}]")

except Exception as e:
    print(f"  Error: {e}")
    import traceback
    traceback.print_exc()

print("\nColabDesign forward pass...")
try:
    colab_logits_af = colab_model.get_unconditional_logits(key=key)
    colab_logits_mpnn = af_logits_to_mpnn(colab_logits_af)

    print(f"  ✅ ColabDesign logits shape: {colab_logits_af.shape}")
    print(f"  Logits range (AF order): [{colab_logits_af.min():.3f}, {colab_logits_af.max():.3f}]")
    print(f"  Logits range (MPNN order): [{colab_logits_mpnn.min():.3f}, {colab_logits_mpnn.max():.3f}]")

except Exception as e:
    print(f"  Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("STEP 2: Compare Final Logits (sanity check)")
print("="*80)

match = compare_arrays(prx_logits, colab_logits_mpnn, "Final logits", tolerance=1e-3)

if not match:
    print("\n⚠️  Logits don't match - need to trace through layers")

    print("\n" + "="*80)
    print("STEP 3: Manual Layer-by-Layer Trace Required")
    print("="*80)

    print("""
To trace layer-by-layer, we need to:
1. Modify the models to return intermediate values
2. Or use JAX's intermediate value extraction
3. Or manually call each layer with the same inputs

The issue is that both models have different internal structures:
- PrxteinMPNN uses Equinox modules
- ColabDesign uses Haiku (different paradigm)

Let me try a different approach: instrument the code to print intermediate values.
""")

    # Try to access encoder directly
    print("\nAttempting to access encoder layers...")

    try:
        print(f"\nPrxteinMPNN encoder: {type(prx_model.encoder)}")
        print(f"  Num layers: {len(prx_model.encoder.layers)}")

        # Try to manually run encoder
        # Need to prepare edge and node features first

        # Looking at the model code, we need to extract these from features module
        print("\n  This requires modifying the model code to expose intermediate values.")
        print("  Creating a monkey-patch approach...")

    except Exception as e:
        print(f"  Error: {e}")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)

print("""
To complete the layer-by-layer trace, we need to:

1. Create modified versions of both models that expose intermediate values
2. Or use JAX's value_and_grad / intermediate transforms
3. Or add print statements directly to the source code

The correlation of 0.62 suggests the models are computing SIMILAR but not IDENTICAL
values. This could be due to:
- Different random number generation
- Different feature preprocessing
- Different attention mechanisms
- Numerical precision differences

Recommended approach:
1. Add debug print statements to PrxteinMPNN source code
2. Add debug hooks to ColabDesign
3. Run both on same input with same random seed
4. Compare intermediate values step by step
""")

print("\nFor now, let's check if random seed is causing the issue...")

# Try multiple seeds
print("\nTesting with multiple random seeds:")
for seed_val in [42, 123, 456]:
    test_key = jax.random.PRNGKey(seed_val)

    _, prx_logits_test = prx_model(
        protein.coordinates,
        protein.mask,
        protein.residue_index,
        protein.chain_index,
        "unconditional",
        prng_key=test_key,
    )

    colab_logits_test = af_logits_to_mpnn(colab_model.get_unconditional_logits(key=test_key))

    corr = np.corrcoef(prx_logits_test.flatten(), colab_logits_test.flatten())[0, 1]
    print(f"  Seed {seed_val}: correlation = {corr:.4f}")

print("\nIf correlations are consistent across seeds, the issue is NOT randomness.")
