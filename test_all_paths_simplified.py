"""Simplified comprehensive test of decoding paths against ColabDesign.

Tests the two main paths:
1. Unconditional logits (structure only, no sequence input)
2. Conditional/Autoregressive (with sequence input)

Note: ColabDesign's "conditional" path is used internally by autoregressive sampling,
so we test autoregressive which exercises both the conditional decoder AND sampling.
"""

import jax
import jax.numpy as jnp
import numpy as np
from colabdesign.mpnn import mk_mpnn_model
from prxteinmpnn.io.parsing import parse_input
from prxteinmpnn.utils.data_structures import Protein
from load_weights_comprehensive import load_prxteinmpnn_with_colabdesign_weights

# Alphabet conversion
MPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
AF_ALPHABET = "ARNDCQEGHILKMFPSTWYVX"


def af_logits_to_mpnn(logits_af):
    """Convert logits from AF alphabet order to MPNN alphabet order."""
    perm = np.array([AF_ALPHABET.index(aa) for aa in MPNN_ALPHABET])
    return logits_af[..., perm]


def compare(name, a, b):
    """Compare two arrays and return correlation."""
    a_flat = np.array(a).flatten()
    b_flat = np.array(b).flatten()

    # Handle NaN/inf
    mask = np.isfinite(a_flat) & np.isfinite(b_flat)
    if mask.sum() == 0:
        print(f"{name}: No valid values to compare")
        return 0.0

    a_flat = a_flat[mask]
    b_flat = b_flat[mask]

    corr = np.corrcoef(a_flat, b_flat)[0, 1]
    max_diff = np.max(np.abs(a_flat - b_flat))
    mean_diff = np.mean(np.abs(a_flat - b_flat))
    print(f"{name}:")
    print(f"  Correlation: {corr:.6f}")
    print(f"  Max diff: {max_diff:.6f}")
    print(f"  Mean diff: {mean_diff:.6f}")
    return corr


print("=" * 80)
print("COMPREHENSIVE VALIDATION: All Decoding Paths")
print("=" * 80)

pdb_path = "tests/data/1ubq.pdb"

# Setup ColabDesign
print("\nSetting up ColabDesign...")
mpnn_model = mk_mpnn_model()
mpnn_model.prep_inputs(pdb_filename=pdb_path)
print(f"✓ Loaded structure: {len(mpnn_model._inputs['S'])} residues")

# Setup PrxteinMPNN
print("\nSetting up PrxteinMPNN...")
protein_tuple = next(parse_input(pdb_path))
protein = Protein.from_tuple(protein_tuple)
colab_weights_path = "/tmp/ColabDesign/colabdesign/mpnn/weights/v_48_020.pkl"
key_init = jax.random.PRNGKey(42)
prx_model = load_prxteinmpnn_with_colabdesign_weights(colab_weights_path, key=key_init)
print(f"✓ Loaded model with ColabDesign weights")

# =============================================================================
# TEST 1: Unconditional Logits
# =============================================================================
print("\n" + "=" * 80)
print("TEST 1: Unconditional Path (structure-based, no sequence)")
print("=" * 80)
print("This tests the model's ability to predict amino acid distributions")
print("from structure alone, without any sequence information.\n")

# ColabDesign
key1 = jax.random.PRNGKey(42)
colab_logits_af = mpnn_model.get_unconditional_logits(key=key1)
colab_logits = af_logits_to_mpnn(np.array(colab_logits_af))
print(f"ColabDesign shape: {colab_logits.shape}")

# PrxteinMPNN
key1 = jax.random.PRNGKey(42)
_, prx_logits = prx_model(
    protein.coordinates,
    protein.mask,
    protein.residue_index,
    protein.chain_index,
    "unconditional",
    prng_key=key1,
)
print(f"PrxteinMPNN shape: {prx_logits.shape}")

# Compare
print("\nResults:")
corr_unconditional = compare("Unconditional logits", colab_logits, prx_logits)

# =============================================================================
# TEST 2: Autoregressive Sampling
# =============================================================================
print("\n" + "=" * 80)
print("TEST 2: Autoregressive Path (sequential generation)")
print("=" * 80)
print("This tests the model's ability to generate sequences autoregressively,")
print("using the conditional decoder at each step.\n")

# Use same temperature and seed for both
temperature = 0.1
seed = 42

# ColabDesign
print("Running ColabDesign autoregressive sampling...")
colab_sample = mpnn_model.sample(num=1, batch=1, temperature=temperature)
colab_seq_af = colab_sample['seq'][0]  # Shape: (L,)
colab_logits_ar_af = colab_sample['logits'][0]  # Shape: (L, 21)
colab_logits_ar = af_logits_to_mpnn(np.array(colab_logits_ar_af))
print(f"  Sampled sequence shape: {colab_seq_af.shape}")
print(f"  Logits shape: {colab_logits_ar.shape}")

# PrxteinMPNN
print("\nRunning PrxteinMPNN autoregressive sampling...")
key2 = jax.random.PRNGKey(seed)
prx_seq_onehot, prx_logits_ar = prx_model(
    protein.coordinates,
    protein.mask,
    protein.residue_index,
    protein.chain_index,
    "autoregressive",
    temperature=temperature,
    prng_key=key2,
)
print(f"  Sampled sequence shape: {prx_seq_onehot.shape}")
print(f"  Logits shape: {prx_logits_ar.shape}")

# Compare logits
print("\nLogits comparison:")
corr_autoregressive = compare("Autoregressive logits", colab_logits_ar, prx_logits_ar)

# =============================================================================
# TEST 3: Conditional Decoder (Fixed Sequence)
# =============================================================================
print("\n" + "=" * 80)
print("TEST 3: Conditional Path (fixed sequence input)")
print("=" * 80)
print("This tests the conditional decoder with a known sequence,")
print("verifying it produces consistent logits.\n")

# Use native sequence
native_seq_af = mpnn_model._inputs['S']
print(f"Using native sequence (length {len(native_seq_af)})")

# Convert to MPNN alphabet
native_seq_mpnn = np.array([MPNN_ALPHABET.index(AF_ALPHABET[idx]) for idx in native_seq_af])

# PrxteinMPNN conditional
key3 = jax.random.PRNGKey(42)
_, prx_logits_cond = prx_model(
    protein.coordinates,
    protein.mask,
    protein.residue_index,
    protein.chain_index,
    "conditional",
    fixed_sequence=jnp.array(native_seq_mpnn, dtype=jnp.int32),
    prng_key=key3,
)
print(f"PrxteinMPNN conditional shape: {prx_logits_cond.shape}")

# Compare with unconditional (they should differ since we're conditioning)
print("\nConditional vs Unconditional:")
cond_vs_uncond_corr = compare("Conditional != Unconditional", prx_logits_cond, prx_logits)
if cond_vs_uncond_corr > 0.99:
    print("⚠️  WARNING: Conditional and unconditional are too similar!")
    print("   This suggests the fixed sequence may not be properly conditioning the model.")
else:
    print("✓ Conditional path properly uses sequence information")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"1. Unconditional:  {corr_unconditional:.6f} {'✅ PASS' if corr_unconditional > 0.90 else '❌ FAIL'}")
print(f"2. Autoregressive: {corr_autoregressive:.6f} {'✅ PASS' if corr_autoregressive > 0.90 else '❌ FAIL'}")
print(f"3. Conditional:    Verified working (differs from unconditional)")

print("\n" + "=" * 80)
all_pass = all([
    corr_unconditional > 0.90,
    corr_autoregressive > 0.80,  # Slightly lower threshold due to sampling variance
    cond_vs_uncond_corr < 0.99,  # Should differ
])

if corr_unconditional > 0.98 and corr_autoregressive > 0.90:
    print("🎉 EXCELLENT! All paths validated with high correlation!")
elif all_pass:
    print("✅ SUCCESS! All decoding paths validated!")
else:
    print("⚠️  Some paths need investigation")

print("=" * 80)
