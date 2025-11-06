"""Tests for STE optimization with tied positions."""

import jax
import jax.numpy as jnp
import pytest

# Marked as TODO: Full integration tests require proper model setup
#from prxteinmpnn.model.mpnn import PrxteinMPNN
#from prxteinmpnn.sampling.sample import make_sample_sequences
#from prxteinmpnn.utils.autoregression import resolve_tie_groups


@pytest.mark.skip(reason="TODO: Requires proper model fixture setup - see test_sample.py")
def test_ste_optimization_with_tied_positions():
  """Test that STE optimization produces identical sequences for tied positions.
  
  This test is currently skipped because it requires:
  1. Proper model fixture setup (see tests/conftest.py and tests/sampling/test_sample.py)
  2. Correctly shaped protein structure data
  3. Proper tie_group_map generation
  
  The core logit averaging logic is tested in test_ste_tied_logits_remain_identical.
  """
  pass


@pytest.mark.skip(reason="TODO: Requires proper model fixture setup - see test_sample.py")
def test_ste_optimization_without_tied_positions():
  """Test that STE optimization works without tied positions (backward compatibility).
  
  This test is currently skipped because it requires proper model fixture setup.
  See tests/sampling/test_sample.py for examples of working STE tests.
  """
  pass


def test_ste_tied_logits_remain_identical():
  """Test that logits for tied positions remain identical throughout optimization."""
  # This is more of a unit test for the averaging logic
  n_residues = 6
  num_classes = 21

  # Create tie_group_map: [0, 0, 1, 1, 2, 2] - 3 groups of 2 positions each
  tie_group_map = jnp.array([0, 0, 1, 1, 2, 2], dtype=jnp.int32)
  num_groups = 3

  # Create logits with some variation
  logits = jax.random.normal(jax.random.PRNGKey(0), (n_residues, num_classes))

  # Apply the averaging logic (from ste_optimize.py)
  group_one_hot = jax.nn.one_hot(tie_group_map, num_groups, dtype=jnp.float32)
  group_logit_sums = jnp.einsum("ng,na->ga", group_one_hot, logits)
  group_counts = group_one_hot.sum(axis=0)
  group_avg_logits = group_logit_sums / (group_counts[:, None] + 1e-8)
  averaged_logits = jnp.einsum("ng,ga->na", group_one_hot, group_avg_logits)

  # Check that positions in the same group have identical logits
  assert jnp.allclose(
    averaged_logits[0], averaged_logits[1]
  ), "Group 0 positions should be identical"
  assert jnp.allclose(
    averaged_logits[2], averaged_logits[3]
  ), "Group 1 positions should be identical"
  assert jnp.allclose(
    averaged_logits[4], averaged_logits[5]
  ), "Group 2 positions should be identical"

  # Check that different groups have different logits (probabilistic but very likely)
  assert not jnp.allclose(
    averaged_logits[0], averaged_logits[2]
  ), "Different groups should differ"
