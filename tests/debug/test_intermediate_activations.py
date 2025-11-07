"""Compare intermediate activations through the model pipeline.

This test tracks activations at each stage to identify where predictions
diverge from expected behavior.
"""

import jax
import jax.numpy as jnp
import pytest
from prxteinmpnn.io.parsing import parse_input
from prxteinmpnn.io.weights import load_model


def test_intermediate_activations():
  """Track activations through the full model pipeline.
  
  This test examines intermediate outputs at each stage:
  1. Input coordinates
  2. Feature extraction (edges, nodes)
  3. Encoder output
  4. Decoder output (unconditional)
  5. Output logits
  
  We're looking for:
  - Where activations become biased toward Alanine
  - Unusual patterns in specific layers
  - NaN/Inf propagation
  """
  print("\nINTERMEDIATE ACTIVATIONS ANALYSIS")
  print("=" * 80)
  
  # Load model and test structure
  model = load_model()
  protein_tuple = next(parse_input("tests/data/1ubq.pdb"))
  coords = protein_tuple.coordinates
  aatype = protein_tuple.aatype
  residue_idx = protein_tuple.residue_index  
  chain_idx = protein_tuple.chain_index
  mask = jnp.ones(len(aatype), dtype=jnp.float32)
  
  print(f"\nInput structure: 1ubq")
  print(f"  Coordinates shape: {coords.shape}")
  print(f"  Sequence length: {len(aatype)}")
  print(f"  True sequence (first 10): {aatype[:10]}")
  
  # Stage 1: Feature extraction
  print("\n" + "=" * 80)
  print("STAGE 1: Feature Extraction")
  print("=" * 80)
  
  key = jax.random.PRNGKey(0)
  edge_features, neighbor_indices, _ = model.features(
    key, coords, mask, residue_idx, chain_idx, backbone_noise=0.0
  )
  
  print(f"\nEdge features shape: {edge_features.shape}")
  print(f"  Mean: {float(jnp.mean(edge_features)):.6f}")
  print(f"  Std: {float(jnp.std(edge_features)):.6f}")
  print(f"  Range: [{float(jnp.min(edge_features)):.6f}, {float(jnp.max(edge_features)):.6f}]")
  print(f"  NaN: {bool(jnp.any(jnp.isnan(edge_features)))}")
  print(f"  Inf: {bool(jnp.any(jnp.isinf(edge_features)))}")
  
  # Check if any positions have unusual edge features
  edge_norms = jnp.linalg.norm(edge_features, axis=(1, 2))  # (L,)
  print(f"\nEdge norms per position (first 10):")
  for i in range(min(10, len(edge_norms))):
    print(f"  Position {i}: {float(edge_norms[i]):.4f}")
  
  # Stage 2: Encoder
  print("\n" + "=" * 80)
  print("STAGE 2: Encoder")
  print("=" * 80)
  
  node_features, edge_features_enc = model.encoder(
    edge_features, neighbor_indices, mask
  )
  
  print(f"\nNode features shape: {node_features.shape}")
  print(f"  Mean: {float(jnp.mean(node_features)):.6f}")
  print(f"  Std: {float(jnp.std(node_features)):.6f}")
  print(f"  Range: [{float(jnp.min(node_features)):.6f}, {float(jnp.max(node_features)):.6f}]")
  print(f"  NaN: {bool(jnp.any(jnp.isnan(node_features)))}")
  print(f"  Inf: {bool(jnp.any(jnp.isinf(node_features)))}")
  
  # Check per-position node features
  node_norms = jnp.linalg.norm(node_features, axis=1)  # (L,)
  print(f"\nNode norms per position (first 10):")
  for i in range(min(10, len(node_norms))):
    print(f"  Position {i}: {float(node_norms[i]):.4f}")
  
  # Check if norms are suspiciously uniform or varied
  node_norm_std = float(jnp.std(node_norms))
  node_norm_mean = float(jnp.mean(node_norms))
  cv = node_norm_std / node_norm_mean if node_norm_mean > 0 else 0
  print(f"\nNode norm coefficient of variation: {cv:.4f}")
  if cv < 0.01:
    print("  ⚠️  Very uniform norms - might indicate collapsed representations")
  if cv > 1.0:
    print("  ⚠️  Very varied norms - some positions might dominate")
  
  # Stage 3: Decoder (unconditional)
  print("\n" + "=" * 80)
  print("STAGE 3: Decoder (Unconditional)")
  print("=" * 80)

  decoded_node_features = model.decoder(
    node_features, edge_features_enc, neighbor_indices, mask
  )
  print(f"\nDecoded node features shape: {decoded_node_features.shape}")
  print(f"  Mean: {float(jnp.mean(decoded_node_features)):.6f}")
  print(f"  Std: {float(jnp.std(decoded_node_features)):.6f}")
  print(f"  Range: [{float(jnp.min(decoded_node_features)):.6f}, {float(jnp.max(decoded_node_features)):.6f}]")
  print(f"  NaN: {bool(jnp.any(jnp.isnan(decoded_node_features)))}")
  print(f"  Inf: {bool(jnp.any(jnp.isinf(decoded_node_features)))}")
  
  # Check per-position decoded features
  decoded_norms = jnp.linalg.norm(decoded_node_features, axis=1)  # (L,)
  print(f"\nDecoded norms per position (first 10):")
  for i in range(min(10, len(decoded_norms))):
    print(f"  Position {i}: {float(decoded_norms[i]):.4f}")
  
  # Stage 4: Output projection
  print("\n" + "=" * 80)
  print("STAGE 4: Output Projection (W_out)")
  print("=" * 80)
  
  logits = jnp.einsum("...d,ad->...a", decoded_node_features, model.w_out.weight)
  logits = logits + model.w_out.bias
  
  print(f"\nLogits shape: {logits.shape}")
  print(f"  Mean: {float(jnp.mean(logits)):.6f}")
  print(f"  Std: {float(jnp.std(logits)):.6f}")
  print(f"  Range: [{float(jnp.min(logits)):.6f}, {float(jnp.max(logits)):.6f}]")
  
  # Analyze logit distribution across amino acids
  from prxteinmpnn.io.parsing.mappings import MPNN_ALPHABET
  
  print("\nLogit statistics per amino acid (averaged over all positions):")
  print("-" * 80)
  print(f"{'AA':3s} {'Index':5s} {'Mean Logit':12s} {'Std Logit':12s} {'Prediction %':14s}")
  print("-" * 80)
  
  predictions = jnp.argmax(logits, axis=-1)
  prediction_counts = jnp.bincount(predictions, length=21)
  prediction_freqs = prediction_counts / jnp.sum(prediction_counts)
  
  for i in range(21):
    aa = MPNN_ALPHABET[i]
    aa_logits = logits[:, i]
    mean_logit = float(jnp.mean(aa_logits))
    std_logit = float(jnp.std(aa_logits))
    pred_freq = float(prediction_freqs[i]) * 100
    
    marker = "⚠️" if i == 0 else "  "  # Mark Alanine
    print(f"{marker} {aa:3s} {i:5d} {mean_logit:12.6f} {std_logit:12.6f} {pred_freq:13.1f}%")
  
  # Check position-specific logits
  print("\n" + "=" * 80)
  print("Position-Specific Analysis (first 10 positions):")
  print("=" * 80)
  
  for pos in range(min(10, logits.shape[0])):
    pos_logits = logits[pos, :]
    pred_aa = int(predictions[pos])
    true_aa = int(aatype[pos])
    
    # Get top 3 predictions
    top3_indices = jnp.argsort(pos_logits)[-3:][::-1]
    
    print(f"\nPosition {pos}:")
    print(f"  True AA: {MPNN_ALPHABET[true_aa]} (index {true_aa})")
    print(f"  Predicted AA: {MPNN_ALPHABET[pred_aa]} (index {pred_aa})")
    print(f"  Top 3 predictions:")
    for rank, idx in enumerate(top3_indices, 1):
      aa = MPNN_ALPHABET[int(idx)]
      logit = float(pos_logits[int(idx)])
      print(f"    {rank}. {aa:3s} (index {int(idx):2d}): logit = {logit:8.4f}")
  
  # Final summary
  print("\n" + "=" * 80)
  print("SUMMARY")
  print("=" * 80)
  
  alanine_freq = float(prediction_freqs[0]) * 100
  print(f"\nAlanine prediction frequency: {alanine_freq:.1f}%")
  
  if alanine_freq > 50:
    print("⚠️  ALANINE BIAS DETECTED!")
    
    # Try to identify where the bias originates
    print("\nDiagnostic checks:")
    
    # Check if decoded features are biased
    decoded_mean = jnp.mean(decoded_node_features, axis=0)  # (128,)
    alanine_weight = model.w_out.weight[0, :]  # (128,)
    alanine_dot = float(jnp.dot(decoded_mean, alanine_weight))
    
    other_dots = []
    for i in range(1, 21):
      other_weight = model.w_out.weight[i, :]
      dot = float(jnp.dot(decoded_mean, other_weight))
      other_dots.append(dot)
    
    mean_other_dot = sum(other_dots) / len(other_dots)
    
    print(f"  1. Decoded features · Alanine weights: {alanine_dot:.6f}")
    print(f"  2. Decoded features · Other weights (mean): {mean_other_dot:.6f}")
    print(f"  3. Ratio (Alanine/Others): {alanine_dot / mean_other_dot if mean_other_dot != 0 else float('inf'):.4f}")
    
    if alanine_dot / mean_other_dot > 2.0:
      print("     → Decoded features are aligned with Alanine weights!")
    
    # Check Alanine bias
    alanine_bias = float(model.w_out.bias[0])
    other_bias_mean = float(jnp.mean(model.w_out.bias[1:]))
    print(f"  4. Alanine bias: {alanine_bias:.6f}")
    print(f"  5. Other biases (mean): {other_bias_mean:.6f}")
    
    if alanine_bias - other_bias_mean > 1.0:
      print("     → Alanine has unusually high bias!")


def test_decoder_with_sequence_context():
  """Test decoder WITH sequence context (conditional mode).
  
  This helps determine if the issue is in:
  - The decoder itself (would affect both conditional/unconditional)
  - The unconditional path specifically (zeros for sequence embeddings)
  """
  print("\nDECODER WITH SEQUENCE CONTEXT")
  print("=" * 80)
  
  model = load_model()
  protein_tuple = next(parse_input("tests/data/1ubq.pdb"))
  coords = protein_tuple.coordinates
  aatype = protein_tuple.aatype
  residue_idx = protein_tuple.residue_index  
  chain_idx = protein_tuple.chain_index
  mask = jnp.ones(len(aatype), dtype=jnp.float32)
  
  # Actually, let's just compute conditional logits directly
  print("\nComputing conditional logits (with true sequence)...")
  
  conditional_logits = model(
    coords,
    mask,
    residue_idx,
    chain_idx,
    aatype,  # Provide true sequence
    mode="conditional",
  )
  
  print(f"\nConditional logits shape: {conditional_logits.shape}")
  
  predictions_cond = jnp.argmax(conditional_logits, axis=-1)
  accuracy = float(jnp.mean(predictions_cond == aatype))
  
  print(f"Accuracy with true sequence context: {accuracy * 100:.1f}%")
  
  # Compare with unconditional
  unconditional_logits = model(coords, mask, residue_idx, chain_idx, mode="unconditional")
  predictions_uncond = jnp.argmax(unconditional_logits, axis=-1)
  accuracy_uncond = float(jnp.mean(predictions_uncond == aatype))
  
  print(f"Accuracy without sequence context: {accuracy_uncond * 100:.1f}%")
  print(f"\nDifference: {(accuracy - accuracy_uncond) * 100:.1f}%")
  
  if accuracy > 0.15 and accuracy_uncond < 0.10:
    print("✓ Decoder works well with sequence context")
    print("→ Issue is likely in unconditional path (sequence embedding = 0)")


if __name__ == "__main__":
  # Run tests directly
  test_intermediate_activations()
  test_decoder_with_sequence_context()
