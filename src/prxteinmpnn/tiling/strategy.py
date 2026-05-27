"""Axis iteration strategy types for BatchPlanner.

Three strategies control how a mapped axis is iterated:
- Vmap: jax.vmap — fully parallel, materializes the full axis.
- SafeMap: jax.lax.map with tile chunking — stateless, memory-bounded.
- Scan: jax.lax.scan with carry — sequential with rolling cross-talk.

AxisStrategy is a sealed union; use isinstance() guards, not if/elif on strings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

C = TypeVar("C")  # carry type
X = TypeVar("X")  # per-step input
Y = TypeVar("Y")  # per-step output


@runtime_checkable
class ScanTransition(Protocol, Generic[C, X, Y]):
  """Transition function for carry-bearing axis scan.

  Must satisfy: (carry, x) -> (carry, y)
  where carry has fixed shape across all iterations of the axis.
  """

  def __call__(self, carry: C, x: X) -> tuple[C, Y]: ...


@dataclass(frozen=True)
class Vmap:
  """Iterate axis via jax.vmap — fully parallel.

  All elements are materialized simultaneously. Use when memory budget allows
  and elements are independent (no cross-talk required).
  """


@dataclass(frozen=True)
class SafeMap:
  """Iterate axis via jax.lax.map with tile chunking — stateless.

  Elements are processed in tiles of `tile` elements at a time.
  No carry state; elements are independent. Use for memory-constrained
  axes where vmap would OOM.
  """

  tile: int


@dataclass(frozen=True)
class Scan(Generic[C, X, Y]):
  """Iterate axis via jax.lax.scan with carry — rolling cross-talk.

  `init` is the initial carry value; must have static shape at JAX trace time.
  `transition(carry, x) -> (carry, y)` is called once per axis element.
  `ordered_sinks=True` means any Sink/Tap on this axis uses ordered=True
  in io_callback, preserving step order for downstream writers.

  CONSTRAINT: Scan is invalid on heterogeneous=True axes (variable-shape
  elements cannot be scanned — jax.lax.scan requires static carry shape).
  BatchPlanner.plan() and make_inference_plan() both enforce this.
  """

  init: Any  # C — initial carry; may contain JAX arrays (traced)
  transition: ScanTransition  # (C, X) -> (C, Y)
  ordered_sinks: bool = True


# Sealed union — all three are concrete and exhaustive.
AxisStrategy = Vmap | SafeMap | Scan

__all__ = ["AxisStrategy", "SafeMap", "Scan", "ScanTransition", "Vmap"]
