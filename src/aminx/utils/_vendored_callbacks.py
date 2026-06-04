"""Host-side helpers for JAX ``io_callback`` streaming (Phase **5g** scaffolding).

Roadmap §3.6 / Phase **5g** calls for vendored jaxbeans-style streaming with:

- **Never** ``ordered=True`` on ``io_callback``: it pins callbacks to program order on the host and forces extra synchronization; keep ``ordered=False`` explicitly at call sites and rely on :func:`jax.effects_barrier` where you need drain / staging semantics.
- :func:`jax.effects_barrier` at sink boundaries to drain host effects deterministically
- bounded queues / backpressure when host sinks fall behind

This module stays **dependency-light** (no jaxbeans import); patterns mirror
``aminx/sampling/ste_optimize.py`` and ``aminx/run/multistate_pools.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Generic, TypeVar

T = TypeVar("T")


class BoundedCallbackHandler(Generic[T]):
  """FIFO backlog with an explicit max depth (host-side backpressure primitive).

  Unlike an unbounded list, this raises when producers outrun consumers — use
  alongside ``io_callback`` producers once the sink is wired.
  """

  __slots__ = ("_max_pending", "_pending", "_sink")

  def __init__(self, sink: Callable[[T], None], *, max_pending: int = 256) -> None:
    self._sink = sink
    self._max_pending = max_pending
    self._pending: list[T] = []

  def submit(self, item: T) -> None:
    if len(self._pending) >= self._max_pending:
      msg = "BoundedCallbackHandler: max_pending exceeded"
      raise RuntimeError(msg)
    self._pending.append(item)

  def drain(self) -> None:
    """Flush pending items to the sink (call after ``jax.effects_barrier``)."""
    while self._pending:
      self._sink(self._pending.pop(0))


def async_indexed_stream(num_items: int, *, chunk_size: int = 1) -> Iterator[tuple[int, int]]:
  """Yield ``(start, count)`` slices covering ``range(num_items)`` on the host.

  JAX integration still routes payloads through ``jax.experimental.io_callback``;
  this helper only schedules chunk boundaries for Python-side loops that wrap JIT stages.
  """
  if chunk_size < 1:
    msg = "chunk_size must be >= 1"
    raise ValueError(msg)
  start = 0
  while start < num_items:
    count = min(chunk_size, num_items - start)
    yield start, count
    start += count
