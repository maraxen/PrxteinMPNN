"""Tests for _sampling_helper._prepare_fixed_controls function.

Focus: Fixed mask application correctness, no double-application bug.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from aminx.host._sampling_helper import _prepare_fixed_controls
from aminx.run.specs import SamplingSpecification
from aminx.utils.data_structures import Protein


def _make_fake_protein(
    batch_size: int = 1, seq_len: int = 10
) -> Protein:
  """Create a minimal Protein namedtuple for testing."""
  return Protein(
      coordinates=jnp.zeros((batch_size, seq_len, 4, 3), dtype=jnp.float32),
      aatype=jnp.zeros((batch_size, seq_len), dtype=jnp.int32),
      atom_mask=jnp.ones((batch_size, seq_len, 37), dtype=jnp.float32),
      residue_index=jnp.arange(seq_len, dtype=jnp.int32)[None, :].repeat(
          batch_size, axis=0
      ),
      chain_index=jnp.zeros((batch_size, seq_len), dtype=jnp.int32),
      mask=jnp.ones((batch_size, seq_len), dtype=jnp.float32),
      mapping=None,
  )


class TestPrepareFixedControlsFixedMask:
  """Test suite for fixed_mask application in _prepare_fixed_controls."""

  def test_float_mask_1d_shape_broadcast(self):
    """Float mask (L,) should broadcast to (batch, L) exactly once.

    Verifies: fixed_mask with shape (seq_len,) is broadcast to (batch_size, seq_len)
    without double-application.
    """
    batch_size = 2
    seq_len = 10
    protein = _make_fake_protein(batch_size=batch_size, seq_len=seq_len)

    # 1D float mask: positions 3, 5, 7 are fixed
    fixed_mask_1d = np.array([0., 0., 0., 1., 0., 1., 0., 1., 0., 0.], dtype=np.float32)
    spec = SamplingSpecification(inputs=[], fixed_mask=fixed_mask_1d)

    # Call _prepare_fixed_controls
    fixed_mask_out, _ = _prepare_fixed_controls(spec, batched_ensemble=protein)

    # Verify shape
    assert fixed_mask_out.shape == (batch_size, seq_len)

    # Verify broadcast: both batches should have the same mask
    expected_mask = np.array([
        [0., 0., 0., 1., 0., 1., 0., 1., 0., 0.],
        [0., 0., 0., 1., 0., 1., 0., 1., 0., 0.],
    ], dtype=np.float32)
    np.testing.assert_array_equal(np.asarray(fixed_mask_out), expected_mask)

  def test_float_mask_2d_shape_no_broadcast(self):
    """Float mask (batch, L) should pass through without modification."""
    batch_size = 2
    seq_len = 10
    protein = _make_fake_protein(batch_size=batch_size, seq_len=seq_len)

    # 2D float mask: different mask per batch
    fixed_mask_2d = np.array(
        [
            [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 1., 0., 1., 0., 0.],
        ],
        dtype=np.float32,
    )
    spec = SamplingSpecification(inputs=[], fixed_mask=fixed_mask_2d)

    fixed_mask_out, _ = _prepare_fixed_controls(spec, batched_ensemble=protein)

    assert fixed_mask_out.shape == (batch_size, seq_len)
    np.testing.assert_array_equal(np.asarray(fixed_mask_out), fixed_mask_2d)

  def test_fixed_mask_and_fixed_positions_union(self):
    """fixed_mask and fixed_positions should be unioned (jnp.maximum).

    Verifies: when both are set, positions fixed by either should be 1.0 in output.
    """
    batch_size = 1
    seq_len = 10
    protein = _make_fake_protein(batch_size=batch_size, seq_len=seq_len)

    # fixed_mask: positions 2, 4 are fixed
    fixed_mask = np.array([0., 0., 1., 0., 1., 0., 0., 0., 0., 0.], dtype=np.float32)
    # fixed_positions: positions 4, 6, 8 are fixed (1D array means broadcast to all batches)
    fixed_positions = np.array([0., 0., 0., 0., 1., 0., 1., 0., 1., 0.], dtype=np.float32)

    spec = SamplingSpecification(inputs=[], fixed_mask=fixed_mask, fixed_positions=fixed_positions)

    fixed_mask_out, _ = _prepare_fixed_controls(spec, batched_ensemble=protein)

    # Expected: union of both (positions 2, 4, 6, 8 are fixed)
    expected = np.array([[0., 0., 1., 0., 1., 0., 1., 0., 1., 0.]], dtype=np.float32)
    np.testing.assert_array_equal(np.asarray(fixed_mask_out), expected)

  def test_idempotent_spec_application(self):
    """Applying same spec twice should produce identical result.

    Verifies: no double-application or accumulation of mask.
    """
    batch_size = 1
    seq_len = 10
    protein = _make_fake_protein(batch_size=batch_size, seq_len=seq_len)

    fixed_mask = np.array([0., 0., 1., 0., 1., 0., 0., 0., 0., 0.], dtype=np.float32)
    spec = SamplingSpecification(inputs=[], fixed_mask=fixed_mask)

    # Apply once
    fixed_mask_out1, _ = _prepare_fixed_controls(spec, batched_ensemble=protein)

    # Apply again
    fixed_mask_out2, _ = _prepare_fixed_controls(spec, batched_ensemble=protein)

    # Results should be identical
    np.testing.assert_array_equal(
        np.asarray(fixed_mask_out1), np.asarray(fixed_mask_out2)
    )

  def test_fixed_tokens_validation_with_mask(self):
    """Invalid fixed_tokens at masked positions should raise ValueError."""
    batch_size = 1
    seq_len = 10
    protein = _make_fake_protein(batch_size=batch_size, seq_len=seq_len)

    # fixed_mask: position 5 is fixed
    fixed_mask = np.array([0., 0., 0., 0., 0., 1., 0., 0., 0., 0.], dtype=np.float32)
    # fixed_tokens: position 5 has invalid token (25 > vocab size of 20)
    fixed_tokens = np.array([0, 0, 0, 0, 0, 25, 0, 0, 0, 0], dtype=np.int32)

    spec = SamplingSpecification(inputs=[], fixed_mask=fixed_mask, fixed_tokens=fixed_tokens)

    with pytest.raises(ValueError, match="fixed_tokens must be in"):
      _prepare_fixed_controls(spec, batched_ensemble=protein)

  def test_no_double_application_bug(self):
    """Verify that fixed_mask is NOT applied twice (the double-application bug).

    This tests the core issue: the old code had TWO separate blocks applying fixed_mask:
    1. Lines 448-456: direct numpy broadcast
    2. Lines 485-493: via _broadcast_per_structure + jnp.maximum

    This would cause the mask to be applied twice, which could break cumulative behavior.
    The fix consolidates into a single application using _broadcast_per_structure.
    """
    batch_size = 1
    seq_len = 10
    protein = _make_fake_protein(batch_size=batch_size, seq_len=seq_len)

    # A mask that should be applied exactly once
    fixed_mask = np.array([0., 0., 1., 0., 1., 0., 0., 0., 0., 0.], dtype=np.float32)
    spec = SamplingSpecification(inputs=[], fixed_mask=fixed_mask)

    fixed_mask_out, _ = _prepare_fixed_controls(spec, batched_ensemble=protein)

    # The mask should be broadcast and set exactly once, not accumulated
    # Before fix: mask might be (1, 10) with values potentially > 1 if applied twice
    # After fix: mask is (1, 10) with values in [0, 1] (no double-application)
    expected = np.array([[0., 0., 1., 0., 1., 0., 0., 0., 0., 0.]], dtype=np.float32)
    np.testing.assert_array_equal(np.asarray(fixed_mask_out), expected)

    # Verify that all values are <= 1.0 (sanity check for double-application)
    assert np.all(np.asarray(fixed_mask_out) <= 1.0), "Mask values exceed 1.0 (possible double-application)"
