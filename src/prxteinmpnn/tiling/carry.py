"""CarrySpec — declare a carry-bearing scan on a named axis.

Used by SamplingSpecification (and custom experiment specs) to indicate which
axes should use jax.lax.scan with a carry, rather than safe_map (stateless).
BatchPlanner.plan() reads CarrySpec list in Phase 0 and pre-demotes matching
axes to Scan(init, transition) decisions before Phases 1 and 2.

CONSTRAINT: Heterogeneous axes (shapes vary per element) cannot be scanned —
jax.lax.scan requires static carry shape. CarrySpec rejects known heterogeneous
axis names eagerly; the planner validator enforces this at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from prxteinmpnn.tiling.strategy import ScanTransition

# Known heterogeneous axis names — Scan is structurally impossible on these.
_HETEROGENEOUS_AXIS_NAMES: frozenset[str] = frozenset({"n_states", "n_structures"})


@dataclass(frozen=True)
class CarrySpec:
  """Declare carry-bearing scan on a named axis.

  Attributes:
      axis_name: Name of the axis (must match AxisSpec.name in the planner,
          e.g. "n_noises", "n_samples", "n_temperatures").
      init: Initial carry value. May contain JAX arrays (traced leaves).
          Shape must be static at JAX trace time.
      transition: (carry, x) -> (carry, y) function. Must be a ScanTransition.
      ordered_sinks: If True, any Sink/Tap on this axis uses ordered=True
          in io_callback (step-ordered guarantees). Default: True.

  Raises:
      ValueError: If axis_name is a known heterogeneous axis.
  """

  axis_name: str
  init: Any
  transition: ScanTransition
  ordered_sinks: bool = True

  def __post_init__(self) -> None:
    if self.axis_name in _HETEROGENEOUS_AXIS_NAMES:
      msg = (
        f"Cannot create CarrySpec for axis '{self.axis_name}': "
        f"this axis is heterogeneous (element shapes vary) and cannot "
        f"be scanned with jax.lax.scan, which requires static carry shape. "
        f"Heterogeneous axes must use SafeMap. "
        f"Known heterogeneous axes: {sorted(_HETEROGENEOUS_AXIS_NAMES)}"
      )
      raise ValueError(msg)


__all__ = ["CarrySpec"]
