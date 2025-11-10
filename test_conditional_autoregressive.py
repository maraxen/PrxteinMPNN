"""Direct comparison of conditional and autoregressive paths.

This test focuses specifically on:
1. Conditional path with a fixed sequence
2. Autoregressive path with sampling
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


def af_seq_to_mpnn_onehot(seq_af, vocab_size=21):
    """Convert AF sequence indices to MPNN one-hot."""
    # seq_af: indices into AF_ALPHABET
    seq_mpnn_indices = np.array([MPNN_ALPHABET.index(AF_ALPHABET[idx]) for idx in seq_af])
    return jax.nn.one_hot(seq_mpnn_indices, vocab_size)


def compare(name, a, b):
    """Compare two arrays and return correlation."""
    a_flat = np.array(a).flatten()
    b_flat = np.array(b).flatten()

    mask = np.isfinite(a_flat) & np.isfinite(b_flat)
    if mask.sum() == 0:
        print(f"{name}: No valid values")
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
print("FOCUSED TEST: Conditional and Autoregressive Paths")
print("=" * 80)

pdb_path = "tests/data/1ubq.pdb"

# Setup
print("\nSetup...")
mpnn_model = mk_mpnn_model()
mpnn_model.prep_inputs(pdb_filename=pdb_path)

protein_tuple = next(parse_input(pdb_path))
protein = Protein.from_tuple(protein_tuple)
colab_weights_path = "/tmp/ColabDesign/colabdesign/mpnn/weights/v_48_020.pkl"
key_init = jax.random.PRNGKey(42)
prx_model = load_prxteinmpnn_with_colabdesign_weights(colab_weights_path, key=key_init)

native_seq_af = mpnn_model._inputs['S']
print(f"✓ Structure length: {len(native_seq_af)} residues")

# =============================================================================
# TEST 1: Conditional Path (Fixed Sequence)
# =============================================================================
print("\n" + "=" * 80)
print("TEST 1: Conditional Path (with fixed sequence)")
print("=" * 80)

# ColabDesign - score with sequence
key1 = jax.random.PRNGKey(42)
colab_result = mpnn_model.score(seq=native_seq_af, key=key1)
colab_logits_cond_af = colab_result['logits']
colab_logits_cond = af_logits_to_mpnn(colab_logits_cond_af)
print(f"ColabDesign: shape {colab_logits_cond.shape}")
print(f"  First 5 logits[0]: {colab_logits_cond[0, :5]}")

# PrxteinMPNN - conditional path
key1 = jax.random.PRNGKey(42)
native_seq_mpnn_onehot = af_seq_to_mpnn_onehot(native_seq_af)
_, prx_logits_cond = prx_model(
    protein.coordinates,
    protein.mask,
    protein.residue_index,
    protein.chain_index,
    "conditional",
    one_hot_sequence=native_seq_mpnn_onehot,
    prng_key=key1,
)
print(f"PrxteinMPNN: shape {prx_logits_cond.shape}")
print(f"  First 5 logits[0]: {prx_logits_cond[0, :5]}")

print("\nComparison:")
corr_conditional = compare("Conditional logits", colab_logits_cond, prx_logits_cond)

# =============================================================================
# TEST 2: Autoregressive Path (Sampling)
# =============================================================================
print("\n" + "=" * 80)
print("TEST 2: Autoregressive Path (sampling)")
print("=" * 80)

# Important: Use fixed decoding order for reproducibility
L = len(native_seq_af)
# Use a simple fixed order: 0, 1, 2, ..., L-1
fixed_order = np.arange(L)

# ColabDesign
key2 = jax.random.PRNGKey(42)
colab_sample = mpnn_model.sample(
    num=1,
    batch=1,
    temperature=0.1,
    decoding_order=fixed_order,
)
colab_logits_ar_af = colab_sample['logits'][0]
colab_logits_ar = af_logits_to_mpnn(colab_logits_ar_af)
print(f"ColabDesign: shape {colab_logits_ar.shape}")
print(f"  First 5 logits[0]: {colab_logits_ar[0, :5]}")

# PrxteinMPNN
key2 = jax.random.PRNGKey(42)
# Need to create ar_mask from decoding order
ar_mask = jnp.zeros((L, L), dtype=jnp.int32)
for i, pos in enumerate(fixed_order):
    # Position `pos` can attend to all positions decoded before it
    ar_mask = ar_mask.at[pos, fixed_order[:i]].set(1)

prx_seq_onehot, prx_logits_ar = prx_model(
    protein.coordinates,
    protein.mask,
    protein.residue_index,
    protein.chain_index,
    "autoregressive",
    ar_mask=ar_mask,
    temperature=0.1,
    prng_key=key2,
)
print(f"PrxteinMPNN: shape {prx_logits_ar.shape}")
print(f"  First 5 logits[0]: {prx_logits_ar[0, :5]}")

print("\nComparison:")
corr_autoregressive = compare("Autoregressive logits", colab_logits_ar, prx_logits_ar)

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"1. Conditional:    {corr_conditional:.6f} {'✅' if corr_conditional > 0.90 else '❌'}")
print(f"2. Autoregressive: {corr_autoregressive:.6f} {'✅' if corr_autoregressive > 0.90 else '❌'}")
print("=" * 80)

if corr_conditional > 0.98 and corr_autoregressive > 0.98:
    print("🎉 EXCELLENT! Both paths validated!")
elif corr_conditional > 0.90 and corr_autoregressive > 0.90:
    print("✅ SUCCESS! Both paths working!")
else:
    print("⚠️  Investigation needed")
    if corr_conditional < 0.90:
        print("  - Conditional path needs debugging")
    if corr_autoregressive < 0.90:
        print("  - Autoregressive path needs debugging")
