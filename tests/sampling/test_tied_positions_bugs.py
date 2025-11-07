"""Tests to verify and demonstrate tied-position bugs."""

import jax
import jax.numpy as jnp
import pytest

from prxteinmpnn.io.weights import load_model
from prxteinmpnn.sampling.sample import (
  make_encoding_sampling_split_fn,
  make_sample_sequences,
)
from prxteinmpnn.utils.autoregression import generate_ar_mask
from prxteinmpnn.utils.decoding_order import random_decoding_order


@pytest.fixture
def simple_structure():
  """Create a simple test structure with 5 residues."""
  n_residues = 5
  coords = jnp.ones((n_residues, 4, 3))
  mask = jnp.ones(n_residues)
  res_idx = jnp.arange(n_residues)
  chain_idx = jnp.zeros(n_residues, dtype=jnp.int32)
  return coords, mask, res_idx, chain_idx


def test_bug_temperature_strategy_tied_positions(simple_structure):
  """Test Bug #1: Temperature strategy with tied positions.
  
  This test verifies that the temperature sampling strategy correctly
  enforces tied positions to have identical amino acids.
  
  EXPECTED: Positions in the same tie group should have identical sequences.
  """
  coords, mask, res_idx, chain_idx = simple_structure
  n_residues = coords.shape[0]

  # Set up tied positions: [0, 1] tied, [2, 3] tied, [4] alone
  tie_group_map = jnp.array([0, 0, 1, 1, 2])
  num_groups = 3

  # Load model and create sampler
  model = load_model()
  sampler_fn = make_sample_sequences(
    model=model,
    decoding_order_fn=random_decoding_order,
    sampling_strategy="temperature",
  )

  # Sample multiple times to check consistency
  key = jax.random.key(42)
  keys = jax.random.split(key, 10)

  for test_key in keys:
    sampled_seq, logits, order = sampler_fn(
      test_key,
      coords,
      mask,
      res_idx,
      chain_idx,
      temperature=jnp.array(1.0),
      tie_group_map=tie_group_map,
      num_groups=num_groups,
    )

    # Check that tied positions have the same amino acid
    assert sampled_seq[0] == sampled_seq[1], (
      f"Bug confirmed: Positions 0 and 1 should be tied but got "
      f"seq[0]={sampled_seq[0]}, seq[1]={sampled_seq[1]}"
    )
    assert sampled_seq[2] == sampled_seq[3], (
      f"Bug confirmed: Positions 2 and 3 should be tied but got "
      f"seq[2]={sampled_seq[2]}, seq[3]={sampled_seq[3]}"
    )


def test_bug_split_sampling_ar_mask(simple_structure):
  """Test Bug #2: Split-sampling path has broken AR mask.
  
  This test verifies the bug in make_encoding_sampling_split_fn where
  generate_ar_mask is called with incorrect arguments.
  
  EXPECTED: The AR mask should allow tied positions to attend to each other
  and to all previous groups, but the bug causes tie_group_map to be passed
  as chain_idx, breaking the mask generation.
  """
  coords, mask, res_idx, chain_idx = simple_structure
  n_residues = coords.shape[0]

  # Set up tied positions
  tie_group_map = jnp.array([0, 0, 1, 1, 2])
  num_groups = 3

  # Create decoding order
  key = jax.random.key(42)
  decoding_order, _ = random_decoding_order(
    key, n_residues, tie_group_map, num_groups,
  )

  # Test the BUGGY call as it appears in the code
  buggy_ar_mask = generate_ar_mask(decoding_order, tie_group_map)

  # Test the CORRECT call with proper parameters
  correct_ar_mask = generate_ar_mask(
    decoding_order, None, tie_group_map, num_groups,
  )

  # The buggy mask should be different from the correct one
  assert not jnp.allclose(buggy_ar_mask, correct_ar_mask), (
    "Bug NOT confirmed: Masks are the same. "
    "This might mean the bug has been fixed or doesn't manifest in this case."
  )

  # The buggy mask incorrectly interprets tie_group_map as chain_idx
  # This means it creates a mask where positions can only attend to
  # positions in the same "chain" (actually the same tie group)
  # Let's verify this behavior

  # For the buggy mask, positions in different tie groups should NOT be able
  # to attend to each other, even if they come earlier in decoding order
  # Example: position 2 (group 1) should not attend to position 0 (group 0)
  # even though position 0 comes first in most decoding orders

  # This is the bug: the mask is too restrictive
  print(f"Buggy AR mask:\n{buggy_ar_mask}")
  print(f"Correct AR mask:\n{correct_ar_mask}")


def test_encoding_split_sample_fn_bug(simple_structure):
  """Test Bug #2 in the context of the full encoding/sampling split.
  
  This test verifies that the sample_fn in make_encoding_sampling_split_fn
  uses a broken AR mask due to incorrect function call. The sequences will
  still be tied (due to the group-wise sampling loop), but the logits used
  are computed with the wrong context.
  
  This test demonstrates the bug was present BEFORE the fix and should now
  show it's resolved.
  """
  coords, mask, res_idx, chain_idx = simple_structure
  n_residues = coords.shape[0]

  # Set up tied positions
  tie_group_map = jnp.array([0, 0, 1, 1, 2])
  num_groups = 3

  # Load model and create split functions
  model = load_model()
  encode_fn, sample_fn = make_encoding_sampling_split_fn(
    model_parameters=model,
    sampling_strategy="temperature",
  )

  # Encode once
  key = jax.random.key(42)
  encoding = encode_fn(
    key,
    coords,
    mask,
    res_idx,
    chain_idx,
    backbone_noise=None,
  )

  # Create decoding order for tied positions
  decoding_order, key = random_decoding_order(
    key, n_residues, tie_group_map, num_groups,
  )

  # Sample with tied positions
  sampled_seq = sample_fn(
    key,
    encoding,
    decoding_order,
    temperature=jnp.array(1.0),
    tie_group_map=tie_group_map,
    num_groups=num_groups,
  )

  # Check that tied positions have the same amino acid
  # After the fix, this should pass
  assert sampled_seq[0] == sampled_seq[1], (
    f"Positions 0 and 1 should be tied but got "
    f"seq[0]={sampled_seq[0]}, seq[1]={sampled_seq[1]}"
  )
  assert sampled_seq[2] == sampled_seq[3], (
    f"Positions 2 and 3 should be tied but got "
    f"seq[2]={sampled_seq[2]}, seq[3]={sampled_seq[3]}"
  )

  # The real bug is that the AR mask used inside sample_fn is computed incorrectly
  # We can't easily test this without refactoring, but the fix ensures
  # generate_ar_mask is called with correct parameters


def test_ar_mask_generation_with_chain_idx():
  """Test that shows the difference between chain_idx and tie_group_map.
  
  This demonstrates what happens when tie_group_map is mistakenly passed
  as chain_idx parameter.
  """
  n = 6
  decoding_order = jnp.arange(n)

  # Scenario: 3 groups, each with 2 positions
  tie_group_map = jnp.array([0, 0, 1, 1, 2, 2])
  num_groups = 3

  # Correct usage: AR mask for tied positions
  correct_mask = generate_ar_mask(
    decoding_order,
    chain_idx=None,
    tie_group_map=tie_group_map,
    num_groups=num_groups,
  )

  # Buggy usage: tie_group_map passed as chain_idx
  # This will interpret the groups as chain boundaries
  buggy_mask = generate_ar_mask(
    decoding_order,
    chain_idx=tie_group_map,  # WRONG!
    tie_group_map=None,
    num_groups=None,
  )

  print("\nCorrect AR mask (tied positions):")
  print(correct_mask)
  print("\nBuggy AR mask (tie_group_map as chain_idx):")
  print(buggy_mask)

  # In the correct mask, all positions can attend to previous groups
  # In the buggy mask, positions can only attend within their "chain" (group)

  # For example, position 2 (group 1) should attend to positions 0,1 (group 0)
  # in the correct mask
  assert correct_mask[2, 0] == 1, "Position 2 should attend to position 0"
  assert correct_mask[2, 1] == 1, "Position 2 should attend to position 1"

  # But in the buggy mask, it cannot (due to different "chain")
  assert buggy_mask[2, 0] == 0, "Buggy: Position 2 cannot attend to position 0"
  assert buggy_mask[2, 1] == 0, "Buggy: Position 2 cannot attend to position 1"
