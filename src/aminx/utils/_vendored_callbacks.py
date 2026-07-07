"""Host-side helpers for JAX ``io_callback`` streaming (Phase **5g** scaffolding).

Roadmap §3.6 / Phase **5g** calls for jaxbeans-style streaming with:

- **Never** ``ordered=True`` on ``io_callback``: it pins callbacks to program order on the host and forces extra synchronization; keep ``ordered=False`` explicitly at call sites and rely on :func:`jax.effects_barrier` where you need drain / staging semantics.
- :func:`jax.effects_barrier` at sink boundaries to drain host effects deterministically

This module stays **dependency-light** (no jaxbeans import, reimplemented in that style);
patterns mirror ``aminx/sampling/ste_optimize.py`` and ``aminx/run/multistate_pools.py``.

EPIC #1541 P4 scoping (2026-07-06, see
``.praxia/docs/specs/260706_epic1541-p4-runner-hostsinks-scoping.md``) found this module's
``BoundedCallbackHandler``/``async_indexed_stream`` names collided with unrelated,
differently-behaved classes of the same name in ``xtrax.io``/``xtrax.engine.io`` (xtrax's
back an async training-loop callback dispatcher; these back host-side JIT-boundary
``io_callback`` chunk scheduling). ``BoundedCallbackHandler`` had zero call sites anywhere
in aminx and was removed rather than renamed; ``async_indexed_stream`` was renamed to
``chunk_int_range`` to remove the collision for its one real caller
(``host/streaming_host.py``).
"""

from __future__ import annotations

from collections.abc import Iterator


def chunk_int_range(num_items: int, *, chunk_size: int = 1) -> Iterator[tuple[int, int]]:
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
