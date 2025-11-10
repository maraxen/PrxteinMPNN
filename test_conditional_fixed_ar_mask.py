"""Test conditional path with explicitly controlled ar_mask.

The key insight: conditional scoring should use ar_mask = zeros (no masking)
to score all positions in parallel given a fixed sequence.
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
print("CONDITIONAL TEST: With Explicit AR Mask Control")
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
L = len(native_seq_af)
print(f"✓ Structure length: {L} residues")

# =============================================================================
# TEST: Conditional with ar_mask = zeros (score all positions in parallel)
# =============================================================================
print("\n" + "=" * 80)
print("Conditional Scoring (ar_mask = zeros, all positions in parallel)")
print("=" * 80)
print("This tests scoring a fixed sequence with no autoregressive masking.")
print("All positions see the full sequence context.\n")

# Create ar_mask = zeros (no autoregressive masking)
ar_mask_zeros = np.zeros((L, L))

# ColabDesign - score with sequence and ar_mask=zeros
key1 = jax.random.PRNGKey(42)
colab_result = mpnn_model.score(seq=native_seq_af, key=key1, ar_mask=ar_mask_zeros)
colab_logits_af = colab_result['logits']
colab_logits = af_logits_to_mpnn(colab_logits_af)
print(f"ColabDesign: shape {colab_logits.shape}")
print(f"  logits[0, :5]: {colab_logits[0, :5]}")
print(f"  logits[10, :5]: {colab_logits[10, :5]}")

# PrxteinMPNN - conditional with ar_mask=zeros
key1 = jax.random.PRNGKey(42)
native_seq_mpnn_onehot = af_seq_to_mpnn_onehot(native_seq_af)
ar_mask_zeros_jnp = jnp.zeros((L, L), dtype=jnp.int32)

_, prx_logits = prx_model(
    protein.coordinates,
    protein.mask,
    protein.residue_index,
    protein.chain_index,
    "conditional",
    one_hot_sequence=native_seq_mpnn_onehot,
    ar_mask=ar_mask_zeros_jnp,
    prng_key=key1,
)
print(f"\nPrxteinMPNN: shape {prx_logits.shape}")
print(f"  logits[0, :5]: {prx_logits[0, :5]}")
print(f"  logits[10, :5]: {prx_logits[10, :5]}")

print("\nComparison:")
corr = compare("Conditional (ar_mask=0)", colab_logits, prx_logits)

print("\n" + "=" * 80)
if corr > 0.98:
    print(f"🎉 EXCELLENT! Conditional path: {corr:.6f}")
elif corr > 0.90:
    print(f"✅ GOOD! Conditional path: {corr:.6f}")
else:
    print(f"⚠️  Conditional path needs work: {corr:.6f}")
print("=" * 80)
