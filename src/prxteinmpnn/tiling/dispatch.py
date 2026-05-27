"""Library-side axis dispatch factory contract (Task S6-A2).

Converts AxisStrategy instances into typed iterators, with rejection logic
for invalid axis/strategy pairs. Part of the composable_jax library surface.
"""

from __future__ import annotations

from prxteinmpnn.tiling.errors import TilingError
from prxteinmpnn.tiling.strategy import AxisStrategy, SafeMap, Scan, Vmap


class DispatchRejected(TilingError):
  """Raised when an AxisStrategy is rejected for a given axis.

  Example: Scan on a heterogeneous axis (state) cannot be used because
  jax.lax.scan requires static carry shape, which incompatible with
  variable-geometry state elements.
  """


def make_axis_dispatch(strategy: AxisStrategy, *, axis: str = "state") -> object:
  """Dispatch an AxisStrategy to a typed iterator.

  Converts a strategy (Vmap, SafeMap, Scan) into a corresponding iterator
  (VmapIterator, SafeMapIterator, JaxScanIterator). Enforces topological
  constraints: e.g., Scan on a heterogeneous axis is rejected.

  Parameters
  ----------
  strategy : AxisStrategy
      One of Vmap, SafeMap, or Scan.
  axis : str, optional
      Name of the axis being dispatched. Default "state". Used to detect
      heterogeneous axes (e.g., state is heterogeneous; Scan is invalid there).

  Returns
  -------
  object
      A MapIterator (VmapIterator or SafeMapIterator) or ScanIterator
      (JaxScanIterator). Concrete types are imported at call time.

  Raises
  ------
  DispatchRejected
      If strategy is Scan and axis is heterogeneous (e.g., axis="state").
  """
  # Lazy import to avoid circular dependency with iterator.py.
  # iterator.py may not exist at dispatch time (parallel task in Wave A).
  from prxteinmpnn.tiling.iterator import (
    JaxScanIterator,
    SafeMapIterator,
    VmapIterator,
  )

  # Reject Scan on heterogeneous axes (state is the canonical heterogeneous axis).
  if isinstance(strategy, Scan):
    if axis == "state":
      raise DispatchRejected(
        f"Cannot use Scan strategy on {axis} axis: {axis} axis contains "
        "heterogeneous (variable-shape) state elements. Scan requires "
        "static carry shape across all iterations.",
      )
    # For future non-state heterogeneous axes, add more checks here.

  # Dispatch by strategy type.
  if isinstance(strategy, Vmap):
    return VmapIterator()
  if isinstance(strategy, SafeMap):
    return SafeMapIterator(tile=strategy.tile)
  if isinstance(strategy, Scan):
    return JaxScanIterator()
  # Exhaustiveness check: should never reach here if AxisStrategy is sealed.
  raise TypeError(f"Unknown strategy type: {type(strategy)}")


__all__ = ["DispatchRejected", "make_axis_dispatch"]
