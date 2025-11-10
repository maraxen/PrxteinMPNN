"""Debug autoregressive path by examining ar_mask construction and sampling."""

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


print("=" * 80)
print("AUTOREGRESSIVE DEBUG")
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

L = len(mpnn_model._inputs['S'])
print(f"✓ Structure length: {L} residues")

# Test 1: Check ar_mask construction
print("\n" + "=" * 80)
print("TEST 1: ar_mask construction")
print("=" * 80)

fixed_order = np.arange(L)
print(f"Fixed decoding order: {fixed_order[:10]}...")

# Construct ar_mask as in the test
ar_mask = jnp.zeros((L, L), dtype=jnp.int32)
for i, pos in enumerate(fixed_order):
    ar_mask = ar_mask.at[pos, fixed_order[:i]].set(1)

print(f"ar_mask shape: {ar_mask.shape}")
print(f"ar_mask[0, :5]: {ar_mask[0, :5]} (position 0 attends to nothing)")
print(f"ar_mask[1, :5]: {ar_mask[1, :5]} (position 1 attends to position 0)")
print(f"ar_mask[2, :5]: {ar_mask[2, :5]} (position 2 attends to 0,1)")
print(f"ar_mask[5, :10]: {ar_mask[5, :10]} (position 5 attends to 0-4)")

# Check decoding order derivation
row_sums = jnp.sum(ar_mask, axis=1)
derived_order = jnp.argsort(row_sums)
print(f"\nRow sums (first 10): {row_sums[:10]}")
print(f"Derived decoding order (first 10): {derived_order[:10]}")
print(f"Matches fixed order: {np.array_equal(derived_order, fixed_order)}")

# Test 2: Sample with ColabDesign default ar_mask
print("\n" + "=" * 80)
print("TEST 2: ColabDesign with default ar_mask (no decoding_order)")
print("=" * 80)

key1 = jax.random.PRNGKey(42)
# ColabDesign without explicit decoding_order
colab_sample_default = mpnn_model.sample(num=1, batch=1, temperature=0.1, key=key1)
colab_logits_default = af_logits_to_mpnn(colab_sample_default['logits'][0])
print(f"ColabDesign (default) logits shape: {colab_logits_default.shape}")
print(f"  logits[0, :5]: {colab_logits_default[0, :5]}")

# Test 3: Sample with fixed decoding order
print("\n" + "=" * 80)
print("TEST 3: ColabDesign with fixed decoding_order")
print("=" * 80)

key2 = jax.random.PRNGKey(42)
colab_sample_fixed = mpnn_model.sample(
    num=1,
    batch=1,
    temperature=0.1,
    decoding_order=fixed_order,
    key=key2,
)
colab_logits_fixed = af_logits_to_mpnn(colab_sample_fixed['logits'][0])
print(f"ColabDesign (fixed order) logits shape: {colab_logits_fixed.shape}")
print(f"  logits[0, :5]: {colab_logits_fixed[0, :5]}")

# Test 4: PrxteinMPNN with ar_mask
print("\n" + "=" * 80)
print("TEST 4: PrxteinMPNN with ar_mask")
print("=" * 80)

key3 = jax.random.PRNGKey(42)
prx_seq, prx_logits = prx_model(
    protein.coordinates,
    protein.mask,
    protein.residue_index,
    protein.chain_index,
    "autoregressive",
    ar_mask=ar_mask,
    temperature=0.1,
    prng_key=key3,
)
print(f"PrxteinMPNN logits shape: {prx_logits.shape}")
print(f"  logits[0, :5]: {prx_logits[0, :5]}")

# Test 5: PrxteinMPNN without ar_mask (all zeros)
print("\n" + "=" * 80)
print("TEST 5: PrxteinMPNN without ar_mask (defaults to zeros)")
print("=" * 80)

key4 = jax.random.PRNGKey(42)
prx_seq_no_mask, prx_logits_no_mask = prx_model(
    protein.coordinates,
    protein.mask,
    protein.residue_index,
    protein.chain_index,
    "autoregressive",
    temperature=0.1,
    prng_key=key4,
)
print(f"PrxteinMPNN (no ar_mask) logits shape: {prx_logits_no_mask.shape}")
print(f"  logits[0, :5]: {prx_logits_no_mask[0, :5]}")

# Comparison
print("\n" + "=" * 80)
print("COMPARISONS")
print("=" * 80)

def compare(a, b, name):
    corr = np.corrcoef(a.flatten(), b.flatten())[0, 1]
    max_diff = np.max(np.abs(a - b))
    print(f"{name}: corr={corr:.6f}, max_diff={max_diff:.6f}")
    return corr

compare(colab_logits_default, colab_logits_fixed, "ColabDesign: default vs fixed")
compare(colab_logits_fixed, prx_logits, "ColabDesign vs PrxteinMPNN (with ar_mask)")
compare(prx_logits, prx_logits_no_mask, "PrxteinMPNN: with vs without ar_mask")
