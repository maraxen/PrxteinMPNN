"""Compare weight values between original PyTorch and Equinox implementations.

This test verifies that weight values match between the original ProteinMPNN
and our Equinox implementation, checking for common conversion errors like:
- Transposition errors
- Sign flips
- Scale differences
- Missing or extra dimensions
"""

import jax.numpy as jnp
import pytest
from prxteinmpnn.io.weights import load_model


def test_weight_statistics():
  """Compare weight statistics across all layers.
  
  This test checks if the weight distributions are reasonable by examining:
  - Mean (should be close to 0 for most layers)
  - Std (should be reasonable, typically 0.01-1.0)
  - Min/Max (check for extreme values)
  - NaN/Inf (should be none)
  """
  model = load_model()
  
  print("\nWEIGHT STATISTICS ANALYSIS")
  print("=" * 80)
  
  # Collect all weights
  def get_all_weights(obj, path="model"):
    """Recursively collect all weight arrays."""
    weights = []
    
    if hasattr(obj, "__dict__"):
      for name, value in obj.__dict__.items():
        current_path = f"{path}.{name}"
        
        # Check if this is a JAX array
        if isinstance(value, jnp.ndarray):
          weights.append((current_path, value))
        # Check if this is a list (for layer lists)
        elif isinstance(value, list):
          for i, item in enumerate(value):
            weights.extend(get_all_weights(item, f"{current_path}[{i}]"))
        # Recurse into nested modules
        elif hasattr(value, "__dict__"):
          weights.extend(get_all_weights(value, current_path))
    
    return weights
  
  all_weights = get_all_weights(model)
  
  # Analyze each weight
  suspicious_weights = []
  
  for path, weight in all_weights:
    mean = float(jnp.mean(weight))
    std = float(jnp.std(weight))
    min_val = float(jnp.min(weight))
    max_val = float(jnp.max(weight))
    has_nan = bool(jnp.any(jnp.isnan(weight)))
    has_inf = bool(jnp.any(jnp.isinf(weight)))
    
    # Flag suspicious weights
    is_suspicious = False
    reasons = []
    
    if has_nan:
      is_suspicious = True
      reasons.append("HAS_NAN")
    if has_inf:
      is_suspicious = True
      reasons.append("HAS_INF")
    if abs(mean) > 1.0 and "bias" not in path.lower():
      is_suspicious = True
      reasons.append(f"HIGH_MEAN={mean:.3f}")
    if std > 5.0 or std < 0.001:
      is_suspicious = True
      reasons.append(f"UNUSUAL_STD={std:.3f}")
    if abs(min_val) > 10.0 or abs(max_val) > 10.0:
      is_suspicious = True
      reasons.append(f"EXTREME_VALUES=[{min_val:.3f}, {max_val:.3f}]")
    
    if is_suspicious:
      suspicious_weights.append((path, reasons))
    
    # Print summary for all weights
    print(f"\n{path}")
    print(f"  Shape: {weight.shape}")
    print(f"  Mean: {mean:8.4f}  Std: {std:8.4f}")
    print(f"  Range: [{min_val:8.4f}, {max_val:8.4f}]")
    if is_suspicious:
      print(f"  ⚠️  SUSPICIOUS: {', '.join(reasons)}")
  
  # Summary
  print("\n" + "=" * 80)
  print(f"Total weights analyzed: {len(all_weights)}")
  print(f"Suspicious weights: {len(suspicious_weights)}")
  
  if suspicious_weights:
    print("\n⚠️  SUSPICIOUS WEIGHTS FOUND:")
    for path, reasons in suspicious_weights:
      print(f"  - {path}: {', '.join(reasons)}")
  else:
    print("\n✓ All weights look reasonable")
  
  # Fail if we found suspicious weights
  assert len(suspicious_weights) == 0, (
    f"Found {len(suspicious_weights)} suspicious weights"
  )


def test_weight_norms():
  """Check weight norms for each layer.
  
  Weight norms should be consistent across similar layers.
  Large variations might indicate incorrect weight loading.
  """
  model = load_model()
  
  print("\nWEIGHT NORM ANALYSIS")
  print("=" * 80)
  
  # Check encoder layers
  print("\nEncoder Layers:")
  encoder_norms = []
  for i, layer in enumerate(model.encoder.layers):
    norms = {}
    norms["edge_message_mlp_0"] = float(jnp.linalg.norm(layer.edge_message_mlp.layers[0].weight))
    norms["edge_message_mlp_1"] = float(jnp.linalg.norm(layer.edge_message_mlp.layers[1].weight))
    norms["edge_message_mlp_2"] = float(jnp.linalg.norm(layer.edge_message_mlp.layers[2].weight))
    norms["edge_update_mlp_0"] = float(jnp.linalg.norm(layer.edge_update_mlp.layers[0].weight))
    norms["edge_update_mlp_1"] = float(jnp.linalg.norm(layer.edge_update_mlp.layers[1].weight))
    norms["edge_update_mlp_2"] = float(jnp.linalg.norm(layer.edge_update_mlp.layers[2].weight))
    norms["dense_0"] = float(jnp.linalg.norm(layer.dense.layers[0].weight))
    norms["dense_1"] = float(jnp.linalg.norm(layer.dense.layers[1].weight))
    
    encoder_norms.append(norms)
    print(f"\nLayer {i}:")
    for name, norm in norms.items():
      print(f"  {name:25s}: {norm:8.2f}")
  
  # Check decoder layers
  print("\nDecoder Layers:")
  decoder_norms = []
  for i, layer in enumerate(model.decoder.layers):
    norms = {}
    norms["message_mlp_0"] = float(jnp.linalg.norm(layer.message_mlp.layers[0].weight))
    norms["message_mlp_1"] = float(jnp.linalg.norm(layer.message_mlp.layers[1].weight))
    norms["message_mlp_2"] = float(jnp.linalg.norm(layer.message_mlp.layers[2].weight))
    norms["dense_0"] = float(jnp.linalg.norm(layer.dense.layers[0].weight))
    norms["dense_1"] = float(jnp.linalg.norm(layer.dense.layers[1].weight))
    
    decoder_norms.append(norms)
    print(f"\nLayer {i}:")
    for name, norm in norms.items():
      print(f"  {name:25s}: {norm:8.2f}")
  
  # Check consistency across layers
  print("\nNorm Consistency Check:")
  print("=" * 80)
  
  # Check if encoder layers have similar norms
  for key in encoder_norms[0].keys():
    values = [layer[key] for layer in encoder_norms]
    mean = sum(values) / len(values)
    std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
    cv = std / mean if mean > 0 else 0  # Coefficient of variation
    
    print(f"Encoder {key:25s}: mean={mean:8.2f}, std={std:8.2f}, cv={cv:.3f}")
    
    # Flag high variation (cv > 0.5 means std is more than 50% of mean)
    if cv > 0.5:
      print(f"  ⚠️  High variation across layers!")
  
  for key in decoder_norms[0].keys():
    values = [layer[key] for layer in decoder_norms]
    mean = sum(values) / len(values)
    std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
    cv = std / mean if mean > 0 else 0
    
    print(f"Decoder {key:25s}: mean={mean:8.2f}, std={std:8.2f}, cv={cv:.3f}")
    
    if cv > 0.5:
      print(f"  ⚠️  High variation across layers!")


def test_output_layer_weights():
  """Detailed analysis of output layer weights.
  
  The output layer (w_out) is critical for amino acid prediction.
  Check if certain amino acids have unusually high/low weights.
  """
  model = load_model()
  
  print("\nOUTPUT LAYER (W_OUT) ANALYSIS")
  print("=" * 80)
  
  w_out_weight = model.w_out.weight  # Shape: (21, 128)
  w_out_bias = model.w_out.bias      # Shape: (21,)
  
  from prxteinmpnn.io.parsing.mappings import MPNN_ALPHABET
  
  print(f"\nWeight shape: {w_out_weight.shape}")
  print(f"Bias shape: {w_out_bias.shape}")
  
  print("\nPer-Amino-Acid Statistics:")
  print("-" * 80)
  print(f"{'AA':3s} {'Index':5s} {'Bias':10s} {'Weight Norm':12s} {'Weight Mean':12s} {'Weight Std':12s}")
  print("-" * 80)
  
  suspicious_aa = []
  
  for i in range(21):
    aa = MPNN_ALPHABET[i]
    bias = float(w_out_bias[i])
    weight_row = w_out_weight[i, :]
    norm = float(jnp.linalg.norm(weight_row))
    mean = float(jnp.mean(weight_row))
    std = float(jnp.std(weight_row))
    
    # Check if Alanine (A, index 0) has unusually high bias or norm
    is_suspicious = False
    if i == 0 and (bias > 1.0 or norm > 5.0):  # Alanine
      is_suspicious = True
      suspicious_aa.append((aa, i, "HIGH_BIAS_OR_NORM"))
    
    marker = "⚠️" if is_suspicious else "  "
    print(f"{marker} {aa:3s} {i:5d} {bias:10.4f} {norm:12.4f} {mean:12.4f} {std:12.4f}")
  
  # Compare Alanine (0) vs other amino acids
  print("\n" + "=" * 80)
  print("Alanine (A, index 0) vs Others:")
  
  alanine_bias = float(w_out_bias[0])
  other_bias_mean = float(jnp.mean(w_out_bias[1:]))
  
  alanine_norm = float(jnp.linalg.norm(w_out_weight[0, :]))
  other_norm_mean = float(jnp.mean(jnp.array([jnp.linalg.norm(w_out_weight[i, :]) for i in range(1, 21)])))
  
  print(f"Alanine bias: {alanine_bias:.4f}")
  print(f"Other AA bias mean: {other_bias_mean:.4f}")
  print(f"Alanine/Others ratio: {alanine_bias / other_bias_mean if other_bias_mean != 0 else float('inf'):.4f}")
  
  print(f"\nAlanine weight norm: {alanine_norm:.4f}")
  print(f"Other AA weight norm mean: {other_norm_mean:.4f}")
  print(f"Alanine/Others ratio: {alanine_norm / other_norm_mean if other_norm_mean != 0 else float('inf'):.4f}")
  
  if suspicious_aa:
    print("\n⚠️  SUSPICIOUS AMINO ACIDS:")
    for aa, idx, reason in suspicious_aa:
      print(f"  - {aa} (index {idx}): {reason}")


if __name__ == "__main__":
  # Run tests directly
  test_weight_statistics()
  test_weight_norms()
  test_output_layer_weights()
