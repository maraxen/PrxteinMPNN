"""Structural protocols for tiling axis-iteration strategies.

Defines the callable-shaped interfaces used by BatchPlanner axis strategies
(``Scan``, ``DedupGather``) in ``aminx.tiling.strategy``. Kept separate from
``types.protocols``/``types.stages`` since these are generic, batch-planning
internals rather than model or inference-pipeline protocols.
"""

from __future__ import annotations

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


@runtime_checkable
class DedupFn(Protocol):
  """In-trace static gather: select K_bucket unique elements from xs."""

  def __call__(self, xs: Any, unique_indices: Any) -> Any: ...


@runtime_checkable
class GatherFn(Protocol):
  """In-trace scatter: expand K_bucket unique results back to N positions."""

  def __call__(self, ys_unique: Any, index_map: Any) -> Any: ...


__all__ = ["DedupFn", "GatherFn", "ScanTransition"]
