"""Debug the first step of autoregressive sampling."""

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
print("FIRST AUTOREGRESSIVE STEP DEBUG")
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

# The first step should be equivalent to unconditional with ar_mask=0
# because position 0 has no previous context
print("\n" + "=" * 80)
print("Hypothesis: First AR step should match unconditional logits")
print("=" * 80)

# Get unconditional logits
key1 = jax.random.PRNGKey(42)
_, unconditional_logits = prx_model(
    protein.coordinates,
    protein.mask,
    protein.residue_index,
    protein.chain_index,
    "unconditional",
    prng_key=key1,
)

print(f"\nUnconditional logits[0, :5]: {unconditional_logits[0, :5]}")

# Get first position logits from autoregressive
fixed_order = np.arange(L)
ar_mask = jnp.zeros((L, L), dtype=jnp.int32)
for i, pos in enumerate(fixed_order):
    ar_mask = ar_mask.at[pos, fixed_order[:i]].set(1)

key2 = jax.random.PRNGKey(42)
_, ar_logits = prx_model(
    protein.coordinates,
    protein.mask,
    protein.residue_index,
    protein.chain_index,
    "autoregressive",
    ar_mask=ar_mask,
    temperature=0.1,
    prng_key=key2,
)

print(f"Autoregressive logits[0, :5]: {ar_logits[0, :5]}")

# Compare
print(f"\nDifference: {np.abs(unconditional_logits[0] - ar_logits[0]).max()}")
print(f"Are they close? {np.allclose(unconditional_logits[0], ar_logits[0], atol=0.01)}")

# Now compare with ColabDesign
key3 = jax.random.PRNGKey(42)
colab_unconditional = mpnn_model.get_unconditional_logits(key=key3)
colab_unconditional_mpnn = af_logits_to_mpnn(colab_unconditional)

print(f"\nColabDesign unconditional[0, :5]: {colab_unconditional_mpnn[0, :5]}")

key4 = jax.random.PRNGKey(42)
colab_ar = mpnn_model.sample(
    num=1,
    batch=1,
    temperature=0.1,
    decoding_order=fixed_order,
    key=key4,
)
colab_ar_logits = af_logits_to_mpnn(colab_ar['logits'][0])

print(f"ColabDesign AR logits[0, :5]: {colab_ar_logits[0, :5]}")

print("\n" + "=" * 80)
print("COMPARISONS")
print("=" * 80)

def compare(a, b, name):
    diff = np.abs(a - b)
    print(f"{name}:")
    print(f"  Max diff: {diff.max():.6f}")
    print(f"  Mean diff: {diff.mean():.6f}")
    print(f"  Close (atol=0.01): {np.allclose(a, b, atol=0.01)}")

compare(unconditional_logits[0], ar_logits[0], "PrxteinMPNN: unconditional vs AR[0]")
compare(colab_unconditional_mpnn[0], colab_ar_logits[0], "ColabDesign: unconditional vs AR[0]")
compare(unconditional_logits[0], colab_unconditional_mpnn[0], "Unconditional: PrxteinMPNN vs ColabDesign")
compare(ar_logits[0], colab_ar_logits[0], "AR[0]: PrxteinMPNN vs ColabDesign")
