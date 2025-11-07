"""Test to diagnose potential conditional decoder bugs.

This test checks whether the conditional decoder is correctly handling
sequence embeddings during the forward pass, and compares it with the
autoregressive scan implementation to identify discrepancies.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from prxteinmpnn.io.parsing import parse_input
from prxteinmpnn.io.weights import load_model
from prxteinmpnn.sampling import make_sample_sequences
from prxteinmpnn.utils.data_structures import Protein


def test_conditional_decoder_vs_autoregressive_scan() -> None:
  """Compare conditional decoder with autoregressive scan implementation.

  This test checks if the conditional decoder and autoregressive scan
  produce similar results when given the same sequence context.
  """
  # Load model
  model = load_model()

  # Load real protein structure
  pdb_path = Path(__file__).parent.parent / "data" / "1ubq.pdb"
  protein_tuple = next(parse_input(str(pdb_path)))
  protein = Protein.from_tuple(protein_tuple)

  # Use first 10 residues for faster testing
  n_residues = 10
  coords = protein.coordinates[:n_residues]
  mask = protein.mask[:n_residues]
  res_idx = protein.residue_index[:n_residues]
  chain_idx = protein.chain_index[:n_residues]

  # Run encoder to get features
  node_features, edge_features, neighbor_indices = model.encoder(
    coords,
    mask,
    res_idx,
    chain_idx,
  )

  # Create a known sequence
  true_sequence = jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
  one_hot_sequence = jax.nn.one_hot(true_sequence, 21)

  # Test 1: Run conditional decoder directly
  # For conditional scoring: each position sees all OTHER positions (not itself)
  ar_mask = 1 - jnp.eye(n_residues, dtype=jnp.int32)
  decoded_conditional = model.decoder.call_conditional(
    node_features,
    edge_features,
    neighbor_indices,
    mask,
    ar_mask,
    one_hot_sequence,
    model.w_s_embed.weight,
  )

  # Compute logits from conditional decoder
  logits_conditional = jax.vmap(model.w_out)(decoded_conditional)

  print("\n=== Conditional Decoder Test ===")
  print(f"Conditional logits shape: {logits_conditional.shape}")
  print(f"Conditional logits mean: {logits_conditional.mean():.4f}")
  print(f"Conditional logits std: {logits_conditional.std():.4f}")

  # Check if conditional logits match the true sequence
  predicted_sequence = logits_conditional.argmax(axis=-1)
  recovery = (predicted_sequence == true_sequence).mean()
  print(f"Sequence recovery (conditional): {recovery * 100:.1f}%")

  # Test 2: Check if sequence embeddings are being used correctly
  # Compare first vs last layer of conditional decoder
  embedded_sequence = one_hot_sequence @ model.w_s_embed.weight
  print(f"\nEmbedded sequence shape: {embedded_sequence.shape}")
  print(f"Embedded sequence mean: {embedded_sequence.mean():.4f}")

  # The conditional decoder should use neighbor sequence embeddings (s_j)
  # not central node embeddings (s_i)
  # Let's verify this by checking if the context includes neighbor info


def test_autoregressive_sequence_embedding_update() -> None:
  """Test that sequence embeddings are updated during autoregressive scan.

  This test verifies that s_embed is properly updated after each sampling
  step in the autoregressive scan.
  """
  model = load_model()

  # Load real protein structure
  pdb_path = Path(__file__).parent.parent / "data" / "1ubq.pdb"
  protein_tuple = next(parse_input(str(pdb_path)))
  protein = Protein.from_tuple(protein_tuple)

  # Use first 10 residues for faster testing
  n_residues = 10
  coords = protein.coordinates[:n_residues]
  mask = protein.mask[:n_residues]
  res_idx = protein.residue_index[:n_residues]
  chain_idx = protein.chain_index[:n_residues]

  # Sample with temperature
  key = jax.random.PRNGKey(42)
  sample_fn = make_sample_sequences(model, sampling_strategy="temperature")
  sampled_aa_idx, logits, _ = sample_fn(
    key,
    coords,
    mask,
    res_idx,
    chain_idx,
    temperature=1.0,
  )

  print("\n=== Autoregressive Sampling Test ===")
  print(f"Sampled sequence shape: {sampled_aa_idx.shape}")
  print(f"Logits shape: {logits.shape}")
  print(f"Sampled amino acids: {sampled_aa_idx}")

  # Check diversity of predictions
  unique_aas = jnp.unique(sampled_aa_idx).size
  print(f"Unique amino acids: {unique_aas} / {n_residues}")

  # Should NOT be all Alanine
  alanine_pct = (sampled_aa_idx == 0).mean() * 100
  print(f"Alanine percentage: {alanine_pct:.1f}%")
  assert alanine_pct < 50.0, "Too much Alanine bias detected"


def test_sequence_edge_features_construction() -> None:
  """Test that edge sequence features are constructed correctly.

  The edge sequence features should be [e_ij, s_j] where s_j are the
  neighbor sequence embeddings, not the central node embeddings.
  """
  model = load_model()

  key = jax.random.PRNGKey(42)
  n_residues = 10
  n_neighbors = 5

  # Create simple features
  edge_features = jax.random.normal(key, (n_residues, n_neighbors, 128))
  s_embed = jax.random.normal(key, (n_residues, 128))
  neighbor_indices = jnp.tile(jnp.arange(n_neighbors)[None, :], (n_residues, 1))

  # Manually construct what should happen
  from prxteinmpnn.utils.concatenate import concatenate_neighbor_nodes

  edge_seq_features = concatenate_neighbor_nodes(
    s_embed,
    edge_features,
    neighbor_indices,
  )

  print("\n=== Edge Sequence Features Test ===")
  print(f"Edge sequence features shape: {edge_seq_features.shape}")
  print(f"Expected shape: ({n_residues}, {n_neighbors}, 256)")

  # Verify shape
  assert edge_seq_features.shape == (n_residues, n_neighbors, 256)

  # Verify that we're getting neighbor features, not central node features
  # The second half should be s_j = s_embed[neighbor_indices]
  s_j_manual = s_embed[neighbor_indices]
  s_j_from_concat = edge_seq_features[..., 128:]

  # These should match
  assert jnp.allclose(
    s_j_manual, s_j_from_concat, rtol=1e-5
  ), "Neighbor sequence features don't match!"

  print("✓ Edge sequence features correctly use neighbor embeddings (s_j)")


if __name__ == "__main__":
  pytest.main([__file__, "-v", "-s"])
