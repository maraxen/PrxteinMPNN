"""Phase 5g — ordering / barrier smoke tests for ``io_callback`` streaming."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from prxteinmpnn.utils._vendored_callbacks import BoundedCallbackHandler, async_indexed_stream


def test_async_indexed_stream_covers_range() -> None:
  idx = list(async_indexed_stream(10, chunk_size=3))
  assert idx == [(0, 3), (3, 3), (6, 3), (9, 1)]


def test_bounded_callback_handler_drain_order() -> None:
  seen: list[int] = []

  def sink(x: int) -> None:
    seen.append(x)

  h = BoundedCallbackHandler(sink, max_pending=4)
  for i in range(3):
    h.submit(i)
  h.drain()
  assert seen == [0, 1, 2]


def test_bounded_callback_handler_max_pending() -> None:
  def sink(_: int) -> None:
    return

  h = BoundedCallbackHandler(sink, max_pending=1)
  h.submit(0)
  with pytest.raises(RuntimeError, match="max_pending"):
    h.submit(1)


@pytest.mark.parity_fast
def test_io_callback_ordered_false_with_effects_barrier() -> None:
  """Callbacks complete before barrier returns; host observes deterministic completion."""
  events: list[int] = []

  def host_a(_: jnp.ndarray) -> None:
    events.append(1)

  def host_b(_: jnp.ndarray) -> None:
    events.append(2)

  @jax.jit
  def f() -> jax.Array:
    jax.experimental.io_callback(host_a, None, jnp.int32(0), ordered=False)
    jax.experimental.io_callback(host_b, None, jnp.int32(0), ordered=False)
    return jnp.float32(0.0)

  out = f()
  jax.block_until_ready(out)
  jax.effects_barrier()
  assert events == [1, 2]
