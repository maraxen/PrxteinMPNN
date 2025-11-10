"""Debug conditional path with detailed intermediate comparisons."""

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


print("=" * 80)
print("CONDITIONAL PATH DETAILED DEBUG")
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
# Compare: Unconditional vs Conditional
# =============================================================================
print("\n" + "=" * 80)
print("Comparison: Unconditional vs Conditional (ar_mask=0)")
print("=" * 80)
print("If conditional with ar_mask=0 differs from unconditional, there's an issue.")
print()

# Get unconditional logits
key1 = jax.random.PRNGKey(42)
_, prx_uncond = prx_model(
    protein.coordinates,
    protein.mask,
    protein.residue_index,
    protein.chain_index,
    "unconditional",
    prng_key=key1,
)

# Get conditional logits with ar_mask=0 (should be same as unconditional?)
key1 = jax.random.PRNGKey(42)
native_seq_mpnn_onehot = af_seq_to_mpnn_onehot(native_seq_af)
ar_mask_zeros = jnp.zeros((L, L), dtype=jnp.int32)

_, prx_cond = prx_model(
    protein.coordinates,
    protein.mask,
    protein.residue_index,
    protein.chain_index,
    "conditional",
    one_hot_sequence=native_seq_mpnn_onehot,
    ar_mask=ar_mask_zeros,
    prng_key=key1,
)

print("PrxteinMPNN Unconditional:")
print(f"  logits[0, :5]: {prx_uncond[0, :5]}")
print(f"  logits[10, :5]: {prx_uncond[10, :5]}")

print("\nPrxteinMPNN Conditional (ar_mask=0):")
print(f"  logits[0, :5]: {prx_cond[0, :5]}")
print(f"  logits[10, :5]: {prx_cond[10, :5]}")

diff = np.abs(np.array(prx_uncond) - np.array(prx_cond))
print(f"\nDifference:")
print(f"  Max diff: {diff.max():.6f}")
print(f"  Mean diff: {diff.mean():.6f}")
print(f"  Median diff: {np.median(diff):.6f}")

if diff.max() < 0.01:
    print("\n✅ Unconditional and Conditional (ar_mask=0) match perfectly!")
    print("   This means the conditional decoder works correctly when ar_mask=0.")
else:
    print(f"\n⚠️  Unconditional and Conditional differ by up to {diff.max():.6f}")
    print("   This suggests the conditional path does something different even with ar_mask=0")

# =============================================================================
# Now compare with ColabDesign
# =============================================================================
print("\n" + "=" * 80)
print("Comparison with ColabDesign")
print("=" * 80)

# ColabDesign unconditional
key2 = jax.random.PRNGKey(42)
colab_uncond_af = mpnn_model.get_unconditional_logits(key=key2)
colab_uncond = af_logits_to_mpnn(colab_uncond_af)

# ColabDesign conditional with ar_mask=0
key2 = jax.random.PRNGKey(42)
colab_cond_result = mpnn_model.score(seq=native_seq_af, key=key2, ar_mask=np.zeros((L, L)))
colab_cond_af = colab_cond_result['logits']
colab_cond = af_logits_to_mpnn(colab_cond_af)

print("\nColabDesign Unconditional:")
print(f"  logits[0, :5]: {colab_uncond[0, :5]}")
print(f"  logits[10, :5]: {colab_uncond[10, :5]}")

print("\nColabDesign Conditional (ar_mask=0):")
print(f"  logits[0, :5]: {colab_cond[0, :5]}")
print(f"  logits[10, :5]: {colab_cond[10, :5]}")

diff_colab = np.abs(colab_uncond - colab_cond)
print(f"\nColabDesign difference (uncond vs cond):")
print(f"  Max diff: {diff_colab.max():.6f}")
print(f"  Mean diff: {diff_colab.mean():.6f}")

if diff_colab.max() > 0.01:
    print(f"\n⚠️  ColabDesign unconditional != conditional even with ar_mask=0")
    print("   This means conditional path SHOULD differ from unconditional")
    print("   (because sequence information changes the decoder behavior)")

# Compare PrxteinMPNN conditional vs ColabDesign conditional
diff_cond = np.abs(np.array(prx_cond) - colab_cond)
corr_cond = np.corrcoef(np.array(prx_cond).flatten(), colab_cond.flatten())[0, 1]

print(f"\n" + "=" * 80)
print("PrxteinMPNN Conditional vs ColabDesign Conditional:")
print(f"  Correlation: {corr_cond:.6f}")
print(f"  Max diff: {diff_cond.max():.6f}")
print(f"  Mean diff: {diff_cond.mean():.6f}")

if corr_cond > 0.98:
    print("\n🎉 EXCELLENT! Conditional paths match!")
elif corr_cond > 0.90:
    print("\n✅ GOOD! Conditional paths are close!")
else:
    print(f"\n⚠️  Conditional correlation needs improvement: {corr_cond:.6f}")
print("=" * 80)
