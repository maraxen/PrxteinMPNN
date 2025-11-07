"""Focused diagnostic for sequence recovery issues.

This test analyzes why sequence recovery is low (~5-6%) by:
1. Comparing unconditional vs conditional decoding
2. Checking if the model can recover its own input sequence
3. Identifying where the prediction diverges from ground truth
"""

from pathlib import Path

import jax
import jax.numpy as jnp

from prxteinmpnn.io.parsing import parse_input
from prxteinmpnn.io.weights import load_model
from prxteinmpnn.sampling import make_sample_sequences
from prxteinmpnn.utils.data_structures import Protein


def test_sequence_recovery_vs_random_baseline() -> None:
  """Test if model performs better than random guessing on sequence recovery."""
  model = load_model()

  # Load real protein structure
  pdb_path = Path(__file__).parent.parent / "data" / "1ubq.pdb"
  protein_tuple = next(parse_input(str(pdb_path)))
  protein = Protein.from_tuple(protein_tuple)

  # Use first 20 residues
  n_residues = 20
  coords = protein.coordinates[:n_residues]
  mask = protein.mask[:n_residues]
  res_idx = protein.residue_index[:n_residues]
  chain_idx = protein.chain_index[:n_residues]
  true_sequence = protein.aatype[:n_residues]

  print("\n=== Sequence Recovery Test ===")
  print(f"Testing {n_residues} residues")
  print(f"True sequence (AA indices): {true_sequence}")

  # Sample at low temperature (should be more confident)
  key = jax.random.PRNGKey(42)
  sample_fn = make_sample_sequences(model, sampling_strategy="temperature")
  sampled_aa_idx, logits, _ = sample_fn(
    key,
    coords,
    mask,
    res_idx,
    chain_idx,
    temperature=0.1,  # Low temp for confident predictions
  )

  print(f"\nSampled sequence (AA indices): {sampled_aa_idx}")

  # Calculate recovery rate
  matches = (sampled_aa_idx == true_sequence).astype(jnp.float32)
  recovery_rate = matches.mean() * 100

  print(f"\nSequence recovery: {recovery_rate:.1f}%")
  print(f"Matches by position: {matches}")

  # Analyze logits at true positions
  true_logits = logits[jnp.arange(n_residues), true_sequence]
  predicted_logits = logits[jnp.arange(n_residues), sampled_aa_idx]

  print(f"\nLogits for true amino acids (mean): {true_logits.mean():.3f}")
  print(f"Logits for predicted amino acids (mean): {predicted_logits.mean():.3f}")
  print(f"Difference: {(predicted_logits - true_logits).mean():.3f}")

  # Check if any position had true AA in top-3
  top3_predictions = jnp.argsort(logits, axis=-1)[:, -3:]
  true_in_top3 = jnp.any(top3_predictions == true_sequence[:, None], axis=1)
  top3_rate = true_in_top3.mean() * 100

  print(f"\nTrue AA in top-3 predictions: {top3_rate:.1f}%")

  # Random baseline would be ~5% (1/20)
  print(f"\nRandom baseline: 5.0%")
  print(f"Model performance: {recovery_rate:.1f}%")

  if recovery_rate < 10.0:
    print("\n⚠️  WARNING: Model performing near random baseline!")
  elif recovery_rate < 30.0:
    print("\n⚠️  WARNING: Model performing below expected 40-60%")
  else:
    print("\n✓ Model performing reasonably")


def test_conditional_scoring() -> None:
  """Test if model assigns high scores to the true sequence."""
  model = load_model()

  # Load real protein structure
  pdb_path = Path(__file__).parent.parent / "data" / "1ubq.pdb"
  protein_tuple = next(parse_input(str(pdb_path)))
  protein = Protein.from_tuple(protein_tuple)

  # Use first 10 residues
  n_residues = 10
  coords = protein.coordinates[:n_residues]
  mask = protein.mask[:n_residues]
  res_idx = protein.residue_index[:n_residues]
  chain_idx = protein.chain_index[:n_residues]
  true_sequence = protein.aatype[:n_residues]

  print("\n=== Conditional Scoring Test ===")
  print(f"Testing {n_residues} residues")
  print(f"True sequence (AA indices): {true_sequence}")

  # Score the true sequence
  true_sequence_onehot = jax.nn.one_hot(true_sequence, 21)

  _, logits_conditional = model(
    coords,
    mask,
    res_idx,
    chain_idx,
    decoding_approach="conditional",
    one_hot_sequence=true_sequence_onehot,
  )

  print(f"\nConditional logits shape: {logits_conditional.shape}")

  # What does the model predict when conditioned on true sequence?
  predicted_aa = logits_conditional.argmax(axis=-1)
  print(f"Predicted sequence: {predicted_aa}")

  # How often does conditional prediction match the input sequence?
  conditional_recovery = (predicted_aa == true_sequence).mean() * 100
  print(f"Conditional recovery: {conditional_recovery:.1f}%")

  # This should be VERY high (>90%) if the model is working correctly
  # The model should be able to "recover" the sequence it's being conditioned on
  if conditional_recovery < 50.0:
    print("\n⚠️  CRITICAL: Model cannot recover input sequence in conditional mode!")
    print("This suggests a bug in conditional decoder or sequence embedding")
  elif conditional_recovery < 90.0:
    print("\n⚠️  WARNING: Conditional recovery lower than expected")
  else:
    print("\n✓ Conditional decoder working correctly")


if __name__ == "__main__":
  test_sequence_recovery_vs_random_baseline()
  test_conditional_scoring()
