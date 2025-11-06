"""Tests for autoregression utilities."""

import chex
import jax.numpy as jnp
import pytest

from prxteinmpnn.run.specs import RunSpecification
from prxteinmpnn.utils.autoregression import generate_ar_mask, resolve_tie_groups
from prxteinmpnn.utils.data_structures import Protein


def make_input(chain_id, residue_index):
  """Create a minimal Protein with batch_dim=1 for testing resolve_tie_groups."""
  n = len(chain_id)
  return Protein(
    coordinates=jnp.zeros((1, n, 4, 3)),  # batch_dim=1
    aatype=jnp.zeros((1, n), dtype=jnp.int32),  # Dummy amino acid types
    one_hot_sequence=jnp.zeros((1, n, 21), dtype=jnp.float32),  # Dummy one-hot
    mask=jnp.ones((1, n), dtype=jnp.bool_),
    chain_index=jnp.array(chain_id, dtype=jnp.int32)[None, :],  # Add batch dim
    residue_index=jnp.array(residue_index, dtype=jnp.int32)[None, :],  # Add batch dim
  )


def test_resolve_tie_groups_none():
  """Test resolve_tie_groups with tied_positions=None."""
  inp = make_input([0, 0, 1], [1, 2, 3])
  spec = RunSpecification(inputs=["dummy"], tied_positions=None)
  out = resolve_tie_groups(spec, inp)
  assert (out == jnp.arange(3)).all()


def test_resolve_tie_groups_direct():
  """Test resolve_tie_groups with tied_positions='direct'."""
  # N=6, L=3, K=2
  # Chain IDs [0,0,0, 1,1,1] indicate 2 structures concatenated
  inp = make_input([0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2])
  spec = RunSpecification(inputs=["dummy"], tied_positions="direct", pass_mode="inter")
  out = resolve_tie_groups(spec, inp)
  assert (out == jnp.array([0, 1, 2, 0, 1, 2])).all()


def test_resolve_tie_groups_auto():
  """Test resolve_tie_groups with tied_positions='auto'."""
  inp = make_input([0, 0, 1, 1], [0, 1, 0, 1])
  spec = RunSpecification(inputs=["dummy"], tied_positions="auto", pass_mode="inter")
  # structure_mappings: seq_pos 0 -> [0,2], seq_pos 1 -> [1,3]
  structure_mappings = [[0, 2], [1, 3]]
  out = resolve_tie_groups(spec, inp, structure_mappings)
  # Should assign same group to 0,2 and to 1,3
  assert out[0] == out[2]
  assert out[1] == out[3]
  assert len(jnp.unique(out)) == 2  # noqa: PLR2004


def test_resolve_tie_groups_explicit():
  """Test resolve_tie_groups with explicit tied positions."""
  inp = make_input([10000, 10001, 10000, 10001], [5, 10, 6, 11])  # noqa: PLR2004
  spec = RunSpecification(inputs=["dummy"], tied_positions=[[(10000, 5), (10001, 10)]])
  out = resolve_tie_groups(spec, inp)
  chain_ids = inp.chain_index[0]
  residue_indices = inp.residue_index[0]
  idx0 = jnp.where((chain_ids == 10000) & (residue_indices == 5))[0][0]  # noqa: PLR2004
  idx1 = jnp.where((chain_ids == 10001) & (residue_indices == 10))[0][0]  # noqa: PLR2004
  assert out[idx0] == out[idx1]
  # The other indices should be in different groups
  assert len(jnp.unique(out)) == 3  # noqa: PLR2004


@pytest.mark.parametrize(
    "decoding_order, expected_mask",
    [
        (
            jnp.array([0, 1, 2]),
            jnp.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]]),
        ),
        (
            jnp.array([2, 0, 1]),
            jnp.array([[1, 1, 1], [0, 1, 0], [0, 1, 1]]),
        ),
        (
            jnp.array([1, 2, 0]),
            jnp.array([[1, 0, 1], [1, 1, 1], [0, 0, 1]]),
        ),
    ],
)
def test_generate_ar_mask(decoding_order, expected_mask):
    """Test the generation of the autoregressive mask.

    Args:
        decoding_order: The order in which atoms are decoded.
        expected_mask: The expected autoregressive mask.

    Raises:
        AssertionError: If the output does not match the expected value.
    """
    mask = generate_ar_mask(decoding_order)
    chex.assert_trees_all_equal(mask, expected_mask)
    chex.assert_shape(mask, (len(decoding_order), len(decoding_order)))
    chex.assert_type(mask, int)