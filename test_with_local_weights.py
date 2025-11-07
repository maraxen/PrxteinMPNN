"""
Test script using locally available ColabDesign weights for both models.
This ensures we're using identical weights.
"""

import jax
import jax.numpy as jnp
import numpy as np
import joblib
from prxteinmpnn.io.parsing import parse_input
from prxteinmpnn.utils.data_structures import Protein
from prxteinmpnn.model import PrxteinMPNN
from colabdesign.mpnn.model import mk_mpnn_model
import equinox as eqx

# Define alphabets
MPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
AF_ALPHABET = "ARNDCQEGHILKMFPSTWYVX"

def af_logits_to_mpnn(logits_af):
    """Convert logits from AF alphabet order to MPNN alphabet order."""
    perm = np.array([AF_ALPHABET.index(aa) for aa in MPNN_ALPHABET])
    return logits_af[..., perm]

# Load test PDB
print("="*80)
print("Loading test structure")
print("="*80)
pdb_path = "tests/data/1ubq.pdb"
protein_tuple = next(parse_input(pdb_path))
protein = Protein.from_tuple(protein_tuple)
print(f"Sequence length: {protein.mask.sum():.0f}")

# Load PrxteinMPNN with ColabDesign weights
print("\n" + "="*80)
print("Loading PrxteinMPNN with ColabDesign weights")
print("="*80)
from load_weights_from_colabdesign import load_prxteinmpnn_from_colabdesign_weights

colab_weights_path = "/tmp/ColabDesign/colabdesign/mpnn/weights/v_48_020.pkl"
key = jax.random.PRNGKey(42)
prxtein_model = load_prxteinmpnn_from_colabdesign_weights(colab_weights_path, key=key)
print("✅ PrxteinMPNN model created with ColabDesign weights")

# Load ColabDesign model (will use same weights)
print("\n" + "="*80)
print("Loading ColabDesign model")
print("="*80)
colab_model = mk_mpnn_model(model_name="v_48_020", weights="original", seed=42)
colab_model.prep_inputs(pdb_filename=pdb_path)
print("✅ ColabDesign model loaded")

# Get unconditional logits from both
print("\n" + "="*80)
print("Getting unconditional logits")
print("="*80)

# PrxteinMPNN (returns logits in MPNN alphabet)
_, prxtein_logits = prxtein_model(
    protein.coordinates,
    protein.mask,
    protein.residue_index,
    protein.chain_index,
    "unconditional",
    prng_key=key,
)
print(f"PrxteinMPNN logits shape: {prxtein_logits.shape}")
print(f"PrxteinMPNN logits range: [{prxtein_logits.min():.3f}, {prxtein_logits.max():.3f}]")

# ColabDesign (returns logits in AF alphabet)
colab_logits_af = colab_model.get_unconditional_logits(key=key)
print(f"ColabDesign logits (AF) shape: {colab_logits_af.shape}")
print(f"ColabDesign logits (AF) range: [{colab_logits_af.min():.3f}, {colab_logits_af.max():.3f}]")

# Convert ColabDesign logits to MPNN alphabet
colab_logits_mpnn = af_logits_to_mpnn(np.array(colab_logits_af))
print(f"ColabDesign logits (MPNN) shape: {colab_logits_mpnn.shape}")

# Compare
print("\n" + "="*80)
print("Comparison Results")
print("="*80)

prx = np.array(prxtein_logits).flatten()
col = colab_logits_mpnn.flatten()

# Pearson correlation
corr = np.corrcoef(prx, col)[0, 1]
print(f"Pearson correlation: {corr:.4f}")

# Cosine similarity
cos_sim = np.dot(prx, col) / (np.linalg.norm(prx) * np.linalg.norm(col))
print(f"Cosine similarity: {cos_sim:.4f}")

# Max difference
max_diff = np.abs(prx - col).max()
max_diff_idx = np.abs(prx - col).argmax()
pos, aa = np.unravel_index(max_diff_idx, prxtein_logits.shape)
print(f"Max logit difference: {max_diff:.3f} at position ({pos}, {aa})")

# Get predictions
prx_pred = np.array(prxtein_logits).argmax(axis=1)
col_pred = colab_logits_mpnn.argmax(axis=1)

# Prediction agreement
agreement = (prx_pred == col_pred).mean()
print(f"Prediction agreement: {agreement*100:.1f}%")

# Sequence recovery
native_seq = np.array(protein.aatype[:len(prx_pred)])
prx_recovery = (prx_pred == native_seq).mean()
col_recovery = (col_pred == native_seq).mean()
print(f"PrxteinMPNN sequence recovery: {prx_recovery*100:.1f}%")
print(f"ColabDesign sequence recovery: {col_recovery*100:.1f}%")

print("\n" + "="*80)
print("Status")
print("="*80)
if corr > 0.9:
    print("✅ PASS: Correlation >0.9")
else:
    print(f"❌ FAIL: Correlation {corr:.4f} < 0.9 (gap: {0.9 - corr:.4f})")
    print("Need to investigate forward pass differences!")
