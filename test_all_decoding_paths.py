"""Comprehensive test of all three decoding paths against ColabDesign.

Tests:
1. Unconditional logits (structure only, no sequence)
2. Conditional logits (given a fixed sequence)
3. Autoregressive sampling (sequential generation)
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


def mpnn_seq_to_af(seq_mpnn):
    """Convert sequence from MPNN alphabet indices to AF alphabet indices."""
    # seq_mpnn: indices into MPNN_ALPHABET
    # We need to convert to indices into AF_ALPHABET
    result = np.zeros_like(seq_mpnn)
    for i, mpnn_idx in enumerate(seq_mpnn):
        if mpnn_idx < len(MPNN_ALPHABET):
            aa = MPNN_ALPHABET[mpnn_idx]
            result[i] = AF_ALPHABET.index(aa)
    return result


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
print("COMPREHENSIVE TEST: All Decoding Paths")
print("=" * 80)

pdb_path = "tests/data/1ubq.pdb"
key = jax.random.PRNGKey(42)

# Setup ColabDesign
print("\n" + "=" * 80)
print("Setup ColabDesign")
print("=" * 80)
mpnn_model = mk_mpnn_model()
mpnn_model.prep_inputs(pdb_filename=pdb_path)
print(f"Loaded structure: {pdb_path}")
print(f"Length: {len(mpnn_model._inputs['S'])} residues")

# Setup PrxteinMPNN
print("\n" + "=" * 80)
print("Setup PrxteinMPNN")
print("=" * 80)
protein_tuple = next(parse_input(pdb_path))
protein = Protein.from_tuple(protein_tuple)
colab_weights_path = "/tmp/ColabDesign/colabdesign/mpnn/weights/v_48_020.pkl"
prx_model = load_prxteinmpnn_with_colabdesign_weights(colab_weights_path, key=key)
print(f"Loaded model with ColabDesign weights")
print(f"Structure shape: {protein.coordinates.shape}")

# =============================================================================
# TEST 1: Unconditional Logits
# =============================================================================
print("\n" + "=" * 80)
print("TEST 1: Unconditional Logits (structure only)")
print("=" * 80)

# ColabDesign
key1 = jax.random.PRNGKey(42)
colab_logits_af = mpnn_model.get_unconditional_logits(key=key1)
colab_logits = af_logits_to_mpnn(np.array(colab_logits_af))
print(f"ColabDesign output shape: {colab_logits.shape}")

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
print(f"PrxteinMPNN output shape: {prx_logits.shape}")

# Compare
print("\nComparison:")
corr_unconditional = compare("Unconditional logits", colab_logits, prx_logits)

# =============================================================================
# TEST 2: Conditional Logits
# =============================================================================
print("\n" + "=" * 80)
print("TEST 2: Conditional Logits (given fixed sequence)")
print("=" * 80)

# Use the native sequence from the PDB
native_seq_af = mpnn_model._inputs['S']  # In AF alphabet
print(f"Using native sequence (length {len(native_seq_af)})")
print(f"First 10 AAs (AF indices): {native_seq_af[:10]}")

# ColabDesign - decode with fixed sequence
key2 = jax.random.PRNGKey(42)
# ColabDesign's conditional path needs the sequence in the inputs
mpnn_model._inputs['S'] = native_seq_af
colab_logits_cond_af = mpnn_model.score(
    seq=native_seq_af,
    key=key2,
)
colab_logits_cond = af_logits_to_mpnn(np.array(colab_logits_cond_af))
print(f"ColabDesign conditional output shape: {colab_logits_cond.shape}")

# PrxteinMPNN - decode with fixed sequence
key2 = jax.random.PRNGKey(42)
# Convert native sequence from AF to MPNN alphabet
native_seq_mpnn = np.array([MPNN_ALPHABET.index(AF_ALPHABET[idx]) for idx in native_seq_af])
print(f"Converted to MPNN alphabet, first 10: {native_seq_mpnn[:10]}")

_, prx_logits_cond = prx_model(
    protein.coordinates,
    protein.mask,
    protein.residue_index,
    protein.chain_index,
    "conditional",
    fixed_sequence=jnp.array(native_seq_mpnn, dtype=jnp.int32),
    prng_key=key2,
)
print(f"PrxteinMPNN conditional output shape: {prx_logits_cond.shape}")

# Compare
print("\nComparison:")
corr_conditional = compare("Conditional logits", colab_logits_cond, prx_logits_cond)

# =============================================================================
# TEST 3: Autoregressive Sampling
# =============================================================================
print("\n" + "=" * 80)
print("TEST 3: Autoregressive Sampling (sequential generation)")
print("=" * 80)

# ColabDesign
key3 = jax.random.PRNGKey(42)
colab_sample = mpnn_model.sample(
    num=1,
    batch=1,
    temperature=0.1,  # Low temperature for deterministic-like behavior
)
colab_seq_af = colab_sample['seq'][0]  # Shape: (L,)
colab_logits_ar_af = colab_sample['logits'][0]  # Shape: (L, 21)
colab_logits_ar = af_logits_to_mpnn(np.array(colab_logits_ar_af))
print(f"ColabDesign sampled sequence shape: {colab_seq_af.shape}")
print(f"ColabDesign autoregressive logits shape: {colab_logits_ar.shape}")
print(f"First 10 AAs (AF indices): {colab_seq_af[:10]}")

# PrxteinMPNN
key3 = jax.random.PRNGKey(42)
prx_seq, prx_logits_ar = prx_model(
    protein.coordinates,
    protein.mask,
    protein.residue_index,
    protein.chain_index,
    "autoregressive",
    temperature=0.1,
    prng_key=key3,
)
print(f"PrxteinMPNN sampled sequence shape: {prx_seq.shape}")
print(f"PrxteinMPNN autoregressive logits shape: {prx_logits_ar.shape}")
print(f"First 10 AAs (MPNN indices): {np.argmax(prx_seq[:10], axis=-1)}")

# Compare logits
print("\nLogits comparison:")
corr_autoregressive = compare("Autoregressive logits", colab_logits_ar, prx_logits_ar)

# Compare sequences
print("\nSequence comparison:")
prx_seq_indices = np.argmax(np.array(prx_seq), axis=-1)  # MPNN alphabet indices
prx_seq_af = mpnn_seq_to_af(prx_seq_indices)  # Convert to AF alphabet
seq_match = np.mean(colab_seq_af == prx_seq_af)
print(f"Sequence identity: {seq_match:.6f} ({np.sum(colab_seq_af == prx_seq_af)}/{len(colab_seq_af)} matches)")

if seq_match < 0.5:
    print("⚠️  Sequences differ significantly - may be due to temperature/sampling")
    print("First 20 positions:")
    print("  ColabDesign (AF):", colab_seq_af[:20])
    print("  PrxteinMPNN (AF):", prx_seq_af[:20])
    print("  Match:            ", ["✓" if a == b else "✗" for a, b in zip(colab_seq_af[:20], prx_seq_af[:20])])

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"1. Unconditional:     {corr_unconditional:.6f} {'✅' if corr_unconditional > 0.90 else '❌'}")
print(f"2. Conditional:       {corr_conditional:.6f} {'✅' if corr_conditional > 0.90 else '❌'}")
print(f"3. Autoregressive:    {corr_autoregressive:.6f} {'✅' if corr_autoregressive > 0.90 else '❌'}")
print(f"   Sequence identity: {seq_match:.6f} {'✅' if seq_match > 0.90 else '⚠️'}")

print("\n" + "=" * 80)
all_pass = all([
    corr_unconditional > 0.90,
    corr_conditional > 0.90,
    corr_autoregressive > 0.90,
])

if all_pass:
    print("🎉 SUCCESS! All decoding paths validated with >0.90 correlation!")
else:
    print("⚠️  Some paths need investigation")
print("=" * 80)
