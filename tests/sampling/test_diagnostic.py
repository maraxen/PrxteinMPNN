"""Diagnostic test to understand why recovery is low."""

import jax
import jax.numpy as jnp

from prxteinmpnn.io.parsing import parse_input
from prxteinmpnn.io.weights import load_model
from prxteinmpnn.utils.data_structures import Protein


def test_diagnostic_sampling_vs_unconditional():
  """Compare autoregressive sampling with unconditional scoring.
  
  This test helps diagnose why recovery is low by comparing:
  1. Unconditional logits (parallel scoring of all positions)
  2. Autoregressive sampling (sequential sampling)
  
  """
  # Load protein
  protein_tuple = next(parse_input('tests/data/1ubq.pdb'))
  protein = Protein.from_tuple(protein_tuple)
  
  # Load model
  model = load_model()
  key = jax.random.key(42)
  
  print("\n" + "="*80)
  print("DIAGNOSTIC TEST: Comparing Sampling Modes")
  print("="*80)
  
  # 1. Test unconditional logits
  print("\n1. UNCONDITIONAL MODE (parallel scoring):")
  seq_unconditional, logits_unconditional = model(
    protein.coordinates,
    protein.mask,
    protein.residue_index,
    protein.chain_index,
    decoding_approach="unconditional",
    prng_key=key,
  )
  
  predicted_unconditional = jnp.argmax(logits_unconditional, axis=-1)
  recovery_unconditional = float(jnp.sum(predicted_unconditional == protein.aatype) / len(protein.aatype))
  
  print(f"   Shape of logits: {logits_unconditional.shape}")
  print(f"   Logits mean: {jnp.mean(logits_unconditional):.3f}")
  print(f"   Logits std: {jnp.std(logits_unconditional):.3f}")
  print(f"   Predicted (first 20): {predicted_unconditional[:20]}")
  print(f"   Native (first 20):    {protein.aatype[:20]}")
  print(f"   Recovery (argmax): {recovery_unconditional:.1%}")
  
  # 2. Test autoregressive sampling (low temp)
  print("\n2. AUTOREGRESSIVE MODE (sequential sampling, T=0.1):")
  from prxteinmpnn.utils.autoregression import generate_ar_mask
  from prxteinmpnn.utils.decoding_order import random_decoding_order
  
  decoding_order, key = random_decoding_order(key, len(protein.aatype))
  ar_mask = generate_ar_mask(decoding_order, None, None, None)
  
  seq_autoregressive, logits_autoregressive = model(
    protein.coordinates,
    protein.mask,
    protein.residue_index,
    protein.chain_index,
    decoding_approach="autoregressive",
    prng_key=key,
    ar_mask=ar_mask,
    temperature=jnp.array(0.1, dtype=jnp.float32),
  )
  
  # Convert from one-hot if needed
  if seq_autoregressive.ndim == 2:
    seq_autoregressive = jnp.argmax(seq_autoregressive, axis=-1)
  
  recovery_autoregressive = float(jnp.sum(seq_autoregressive == protein.aatype) / len(protein.aatype))
  
  print(f"   Shape of sampled seq: {seq_autoregressive.shape}")
  print(f"   Shape of logits: {logits_autoregressive.shape}")
  print(f"   Logits mean: {jnp.mean(logits_autoregressive):.3f}")
  print(f"   Logits std: {jnp.std(logits_autoregressive):.3f}")
  print(f"   Sampled (first 20): {seq_autoregressive[:20]}")
  print(f"   Native (first 20):  {protein.aatype[:20]}")
  print(f"   Recovery: {recovery_autoregressive:.1%}")
  
  # 3. Compare logits at specific positions
  print("\n3. LOGITS COMPARISON (positions 0-5):")
  print(f"   Position | Native | Uncond Pred | AR Pred | Uncond Logit[native] | AR Logit[native]")
  print(f"   " + "-"*90)
  for i in range(min(5, len(protein.aatype))):
    native_aa = int(protein.aatype[i])
    uncond_pred = int(predicted_unconditional[i])
    ar_pred = int(seq_autoregressive[i])
    uncond_logit_native = float(logits_unconditional[i, native_aa])
    ar_logit_native = float(logits_autoregressive[i, native_aa])
    
    print(f"   {i:8d} | {native_aa:6d} | {uncond_pred:11d} | {ar_pred:7d} | "
          f"{uncond_logit_native:20.3f} | {ar_logit_native:16.3f}")
  
  # 4. Check if sequences are one-hot or indices
  print("\n4. SEQUENCE ENCODING CHECK:")
  print(f"   Unconditional seq shape: {seq_unconditional.shape}")
  print(f"   Unconditional seq dtype: {seq_unconditional.dtype}")
  print(f"   Unconditional seq (first 5): {seq_unconditional[:5]}")
  print(f"   Autoregressive seq shape: {seq_autoregressive.shape}")
  print(f"   Autoregressive seq dtype: {seq_autoregressive.dtype}")
  
  # 5. Check decoding order
  print("\n5. DECODING ORDER:")
  print(f"   Decoding order (first 20): {decoding_order[:20]}")
  print(f"   AR mask diagonal: {jnp.diag(ar_mask)[:20]}")
  print(f"   AR mask sum per row (first 10): {jnp.sum(ar_mask, axis=1)[:10]}")
  
  # 6. Test with very low temperature
  print("\n6. VERY LOW TEMPERATURE TEST (T=0.01):")
  seq_very_low_temp, _ = model(
    protein.coordinates,
    protein.mask,
    protein.residue_index,
    protein.chain_index,
    decoding_approach="autoregressive",
    prng_key=jax.random.key(99),
    ar_mask=ar_mask,
    temperature=jnp.array(0.01, dtype=jnp.float32),
  )
  
  if seq_very_low_temp.ndim == 2:
    seq_very_low_temp = jnp.argmax(seq_very_low_temp, axis=-1)
  
  recovery_very_low = float(jnp.sum(seq_very_low_temp == protein.aatype) / len(protein.aatype))
  print(f"   Recovery (T=0.01): {recovery_very_low:.1%}")
  print(f"   Sampled (first 20): {seq_very_low_temp[:20]}")
  
  # 7. Check if logits from unconditional match argmax sampling
  print("\n7. GREEDY SAMPLING FROM UNCONDITIONAL:")
  print(f"   If we greedily pick argmax from unconditional logits: {recovery_unconditional:.1%}")
  print(f"   This should be an upper bound for autoregressive sampling.")
  
  print("\n" + "="*80)
  print("SUMMARY:")
  print("="*80)
  print(f"Unconditional (argmax):      {recovery_unconditional:.1%}")
  print(f"Autoregressive (T=0.1):      {recovery_autoregressive:.1%}")
  print(f"Autoregressive (T=0.01):     {recovery_very_low:.1%}")
  print()
  print("If unconditional is low (~5%), the model itself has issues.")
  print("If autoregressive << unconditional, the sampling logic has issues.")
  print("="*80)


if __name__ == "__main__":
  test_diagnostic_sampling_vs_unconditional()
