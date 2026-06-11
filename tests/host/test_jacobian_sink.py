"""Tests for Jacobian io_callback accumulation sink."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from aminx.host.output_sinks import (
  _dispatch_jacobian_io,
  active_jacobian_sink,
  jacobian_sink_session,
  stage_jacobian_io,
)


def test_dispatch_jacobian_io_stages_to_active_sink() -> None:
  """io_callback handler appends host Jacobian arrays to the active sink."""
  jac = np.zeros((4, 21), dtype=np.float32)
  with jacobian_sink_session() as sink:
    _dispatch_jacobian_io(0, jac)
    staged = sink.take_all()
  assert len(staged) == 1
  np.testing.assert_array_equal(staged[0], jac)


def test_stage_jacobian_io_drains_after_barrier() -> None:
  """``stage_jacobian_io`` + ``effects_barrier`` populates the session sink."""
  jac = jnp.ones((3, 21), dtype=jnp.float32)
  with jacobian_sink_session() as sink:
    _ = stage_jacobian_io(jnp.int32(7), jac)
    jax.effects_barrier()
    staged = sink.take_all()
  assert len(staged) == 1
  assert staged[0].shape == (3, 21)


def test_active_jacobian_sink_none_outside_session() -> None:
  assert active_jacobian_sink() is None
