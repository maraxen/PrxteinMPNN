"""Shared host utilities for sampling/scoring streaming (Phase 5f).

Centralizes ``jax.effects_barrier`` at sink boundaries, chunk iteration aligned with
:func:`~aminx.utils._vendored_callbacks.chunk_int_range`, and
structure-iterator sizing for ``io_callback`` scalar markers.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import jax

from aminx.utils._vendored_callbacks import chunk_int_range


def sink_barrier() -> None:
  """Synchronize ``io_callback(..., ordered=False)`` effects before host sink drains."""
  jax.effects_barrier()


def structure_batch_count_for_iterator(protein_iterator: Any) -> int:  # noqa: ANN401
  """Return ``len(iterator)`` for structure-batch markers, or ``-1`` if not sized."""
  try:
    return len(protein_iterator)
  except TypeError:
    return -1


def iter_streaming_chunks(total: int, chunk_size: int) -> Iterator[tuple[int, int]]:
  """Yield ``(chunk_start, chunk_count)`` covering ``[0, total)`` (same semantics as manual range)."""
  yield from chunk_int_range(total, chunk_size=chunk_size)


class StreamingBatchHost:
  """Namespace for host streaming helpers used by ``run/sampling.py`` and ``run/scoring.py``."""

  sink_barrier = staticmethod(sink_barrier)
  structure_batch_count = staticmethod(structure_batch_count_for_iterator)
  iter_chunks = staticmethod(iter_streaming_chunks)
