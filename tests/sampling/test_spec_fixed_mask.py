"""Tests: spec.fixed_mask flows through _prepare_fixed_controls.

Invariant: when SamplingSpecification.fixed_mask[i] = 1.0, the resulting
fixed_mask_for_vmap tensor must have 1.0 at position i — regardless of
whether fixed_positions is also set.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from aminx.host._sampling_helper import _prepare_fixed_controls
from aminx.run.specs import SamplingSpecification


class _FakeBatch:
    """Minimal stand-in for a batched Protein with coordinates attribute."""

    def __init__(self, batch_size: int, seq_len: int) -> None:
        self.coordinates = np.zeros((batch_size, seq_len, 4, 3), dtype=np.float32)


N = 10
BATCH = 1


@pytest.fixture
def fake_batch():
    return _FakeBatch(BATCH, N)


def _base_spec(**kwargs):
    return SamplingSpecification(inputs="test.pdb", **kwargs)


def test_fixed_mask_no_fixed_positions(fake_batch):
    """fixed_mask alone (no fixed_positions) must be reflected in the output."""
    mask = np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32)
    spec = _base_spec(fixed_mask=mask)
    fm, _ft = _prepare_fixed_controls(spec, batched_ensemble=fake_batch)
    result = np.array(fm[0])
    np.testing.assert_array_equal(
        result,
        mask,
        err_msg="fixed_mask not reflected in output when fixed_positions is unset",
    )


def test_fixed_mask_union_with_fixed_positions(fake_batch):
    """fixed_mask and fixed_positions together must union (OR) their masked positions."""
    pos_mask = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)  # position 0
    fm_mask = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)   # position 2
    expected = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)

    spec = _base_spec(fixed_positions=pos_mask, fixed_mask=fm_mask)
    fm, _ft = _prepare_fixed_controls(spec, batched_ensemble=fake_batch)
    result = np.array(fm[0])
    np.testing.assert_array_equal(
        result,
        expected,
        err_msg="fixed_mask + fixed_positions union not correct",
    )


def test_fixed_mask_none_unchanged(fake_batch):
    """When fixed_mask is None, output must be all zeros (no fixed positions set)."""
    spec = _base_spec()
    fm, _ft = _prepare_fixed_controls(spec, batched_ensemble=fake_batch)
    result = np.array(fm[0])
    np.testing.assert_array_equal(
        result,
        np.zeros(N, dtype=np.float32),
        err_msg="Output should be all zeros when neither fixed_mask nor fixed_positions is set",
    )


def test_fixed_mask_does_not_affect_fixed_tokens(fake_batch):
    """Setting fixed_mask must not alter fixed_tokens output."""
    mask = np.ones(N, dtype=np.float32)
    spec = _base_spec(fixed_mask=mask)
    _fm, ft = _prepare_fixed_controls(spec, batched_ensemble=fake_batch)
    result = np.array(ft[0])
    np.testing.assert_array_equal(
        result,
        np.zeros(N, dtype=np.int32),
        err_msg="fixed_tokens should stay all-zeros when only fixed_mask is set (no fixed_tokens spec)",
    )
