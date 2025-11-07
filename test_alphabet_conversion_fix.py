"""
Test script to verify that alphabet conversion fixes the comparison issue.

This script:
1. Loads both models
2. Gets logits from both
3. Converts ColabDesign logits from AF to MPNN alphabet
4. Compares the properly aligned logits
"""

import jax
import jax.numpy as jnp
import numpy as np
from prxteinmpnn.io.parsing import parse_input
from prxteinmpnn.utils.data_structures import Protein
from prxteinmpnn.io.weights import load_model as load_prxteinmpnn
from colabdesign.mpnn.model import mk_mpnn_model

# Define alphabets
MPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
AF_ALPHABET = "ARNDCQEGHILKMFPSTWYVX"

def af_logits_to_mpnn(logits_af):
    """
    Convert logits from AlphaFold alphabet order to MPNN alphabet order.

    ColabDesign returns logits where column i corresponds to AF_ALPHABET[i].
    We need to reorder columns so column i corresponds to MPNN_ALPHABET[i].

    Args:
        logits_af: Array of shape [..., 21] with logits in AF alphabet order

    Returns:
        logits_mpnn: Array of shape [..., 21] with logits in MPNN alphabet order
    """
    # Create permutation: for each MPNN position, find where it is in AF alphabet
    perm = np.array([AF_ALPHABET.index(aa) for aa in MPNN_ALPHABET])
    return logits_af[..., perm]

def mpnn_logits_to_af(logits_mpnn):
    """
    Convert logits from MPNN alphabet order to AlphaFold alphabet order.

    Args:
        logits_mpnn: Array of shape [..., 21] with logits in MPNN alphabet order

    Returns:
        logits_af: Array of shape [..., 21] with logits in AF alphabet order
    """
    # Create permutation: for each AF position, find where it is in MPNN alphabet
    perm = np.array([MPNN_ALPHABET.index(aa) for aa in AF_ALPHABET])
    return logits_mpnn[..., perm]

# Load test PDB
print("="*80)
print("Loading test structure")
print("="*80)
pdb_path = "tests/data/1ubq.pdb"
protein_tuple = next(parse_input(pdb_path))
protein = Protein.from_tuple(protein_tuple)
print(f"Sequence length: {protein.mask.sum():.0f}")
print(f"Native sequence (MPNN): {protein.aatype[:20]}")

# Load models
print("\n" + "="*80)
print("Loading models")
print("="*80)
local_weights_path = "src/prxteinmpnn/io/weights/original_v_48_020.eqx"
prxtein_model = load_prxteinmpnn(local_path=local_weights_path)
print("✅ PrxteinMPNN loaded")

colab_model = mk_mpnn_model(model_name="v_48_020", weights="original", seed=42)
colab_model.prep_inputs(pdb_filename=pdb_path)
print("✅ ColabDesign loaded")

# Get unconditional logits from both
print("\n" + "="*80)
print("Getting unconditional logits")
print("="*80)
key = jax.random.PRNGKey(42)

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

# ColabDesign (returns logits in AF alphabet after internal conversion)
colab_logits_af = colab_model.get_unconditional_logits(key=key)
print(f"ColabDesign logits (AF) shape: {colab_logits_af.shape}")
print(f"ColabDesign logits (AF) range: [{colab_logits_af.min():.3f}, {colab_logits_af.max():.3f}]")

# Convert ColabDesign logits to MPNN alphabet
colab_logits_mpnn = af_logits_to_mpnn(np.array(colab_logits_af))
print(f"ColabDesign logits (MPNN) shape: {colab_logits_mpnn.shape}")
print(f"ColabDesign logits (MPNN) range: [{colab_logits_mpnn.min():.3f}, {colab_logits_mpnn.max():.3f}]")

# Compare WITHOUT conversion (original bug)
print("\n" + "="*80)
print("COMPARISON WITHOUT ALPHABET CONVERSION (BUGGY)")
print("="*80)
prx_flat = np.array(prxtein_logits).flatten()
colab_flat_af = np.array(colab_logits_af).flatten()

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

corr_buggy = np.corrcoef(prx_flat, colab_flat_af)[0, 1]
cos_buggy = cosine_similarity(prx_flat, colab_flat_af)

print(f"Pearson correlation: {corr_buggy:.4f} (WRONG - different alphabets!)")
print(f"Cosine similarity:   {cos_buggy:.4f} (WRONG - different alphabets!)")

# Get predictions in wrong alphabets
prx_preds_mpnn = prxtein_logits.argmax(axis=-1)
colab_preds_af = colab_logits_af.argmax(axis=-1)
agreement_buggy = (prx_preds_mpnn == colab_preds_af).sum() / protein.mask.sum()
print(f"Prediction agreement: {agreement_buggy:.1%} (WRONG - comparing different alphabets!)")

# Compare WITH conversion (fixed)
print("\n" + "="*80)
print("COMPARISON WITH ALPHABET CONVERSION (FIXED)")
print("="*80)
colab_flat_mpnn = colab_logits_mpnn.flatten()

corr_fixed = np.corrcoef(prx_flat, colab_flat_mpnn)[0, 1]
cos_fixed = cosine_similarity(prx_flat, colab_flat_mpnn)

print(f"Pearson correlation: {corr_fixed:.4f} {'✅ GOOD' if corr_fixed > 0.9 else '❌ Still low'}")
print(f"Cosine similarity:   {cos_fixed:.4f} {'✅ GOOD' if cos_fixed > 0.9 else '❌ Still low'}")

# Get predictions in same alphabet
colab_preds_mpnn = colab_logits_mpnn.argmax(axis=-1)
agreement_fixed = (prx_preds_mpnn == colab_preds_mpnn).sum() / protein.mask.sum()
print(f"Prediction agreement: {agreement_fixed:.1%} {'✅ GOOD' if agreement_fixed > 0.8 else '❌ Still low'}")

# Recovery rates (both against MPNN native sequence)
prx_recovery = (prx_preds_mpnn == protein.aatype).sum() / protein.mask.sum()
colab_recovery = (colab_preds_mpnn == protein.aatype).sum() / protein.mask.sum()
print(f"\nSequence recovery:")
print(f"  PrxteinMPNN: {prx_recovery:.1%}")
print(f"  ColabDesign: {colab_recovery:.1%}")

# Detailed position comparison for first 10 positions
print("\n" + "="*80)
print("DETAILED POSITION COMPARISON (First 10 positions)")
print("="*80)
print(f"{'Pos':<4} {'Native(MPNN)':<12} {'PrxteinMPNN':<13} {'Colab(AF)':<11} {'Colab(MPNN)':<12} {'Match?':<7}")
print("-" * 70)

for i in range(min(10, len(protein.aatype))):
    native_aa = MPNN_ALPHABET[protein.aatype[i]]
    prx_aa = MPNN_ALPHABET[prx_preds_mpnn[i]]
    colab_af_aa = AF_ALPHABET[colab_preds_af[i]] if colab_preds_af[i] < 21 else 'X'
    colab_mpnn_aa = MPNN_ALPHABET[colab_preds_mpnn[i]]
    match = "✅" if prx_preds_mpnn[i] == colab_preds_mpnn[i] else "❌"

    print(f"{i:<4} {native_aa:<12} {prx_aa:<13} {colab_af_aa:<11} {colab_mpnn_aa:<12} {match:<7}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Correlation improvement: {corr_buggy:.4f} → {corr_fixed:.4f} (Δ = {corr_fixed - corr_buggy:+.4f})")
print(f"Cosine sim improvement:  {cos_buggy:.4f} → {cos_fixed:.4f} (Δ = {cos_fixed - cos_buggy:+.4f})")
print(f"Agreement improvement:   {agreement_buggy:.1%} → {agreement_fixed:.1%} (Δ = {(agreement_fixed - agreement_buggy)*100:+.1f}%)")

if corr_fixed > 0.95 and cos_fixed > 0.95:
    print("\n✅ SUCCESS! Logits match after alphabet conversion.")
    print("   This confirms the implementations are equivalent.")
else:
    print("\n⚠️  Logits still don't match perfectly after alphabet conversion.")
    print("   There may be other implementation differences to investigate.")
