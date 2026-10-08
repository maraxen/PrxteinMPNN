"""``_dispatch_axis`` routes xtrax's ``ChunkedMap`` (named ``SafeMap`` before
xtrax 0.4.0a11) to chunked mapping with the strategy's own tile.

Dispatch is by class name, so a renamed class silently fell through to the
fallback branch, which also calls ``_safe_map`` but with
``batch_size_fallback``. The tile reaching ``_safe_map`` is what tells the two
apart.
"""

import jax.numpy as jnp
from xtrax.tiling import ChunkedMap

from aminx.host import kernel_dispatch
from aminx.tiling.strategy import SafeMap as AminxSafeMap


def _capture(monkeypatch):
  seen = []

  def fake_safe_map(body, xs, *, batch_size):
    seen.append(batch_size)
    return body(xs)

  monkeypatch.setattr(kernel_dispatch, "_safe_map", fake_safe_map)
  return seen


def test_xtrax_chunked_map_uses_its_batch_size(monkeypatch):
  seen = _capture(monkeypatch)
  kernel_dispatch._dispatch_axis(ChunkedMap(batch_size=3), lambda x: x, jnp.arange(6), batch_size_fallback=99)
  assert seen == [3]


def test_aminx_safe_map_uses_its_tile(monkeypatch):
  seen = _capture(monkeypatch)
  kernel_dispatch._dispatch_axis(AminxSafeMap(tile=5), lambda x: x, jnp.arange(6), batch_size_fallback=99)
  assert seen == [5]


def test_unknown_strategy_uses_fallback(monkeypatch):
  """Negative control: an unrecognised strategy really does reach the fallback."""

  class Other:
    batch_size = 3

  seen = _capture(monkeypatch)
  kernel_dispatch._dispatch_axis(Other(), lambda x: x, jnp.arange(6), batch_size_fallback=99)
  assert seen == [99]
