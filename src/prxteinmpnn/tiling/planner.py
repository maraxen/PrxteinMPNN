from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from collections.abc import Callable

  from prxteinmpnn.tiling.carry import CarrySpec

from prxteinmpnn.tiling.strategy import AxisStrategy, SafeMap, Scan, Vmap

logger = logging.getLogger(__name__)

# TODO: introduce bucketing management in here


def ceil_to_granularity(n: int, g: int) -> int:
  """Round n up to the nearest multiple of g. If g <= 1 or n == 0, returns n unchanged."""
  if g <= 1 or n == 0:
    return n
  return ((n + g - 1) // g) * g


def estimate_memory_theoretical(
  decisions: list[AxisDecision],
  base_shape_bytes: float,
  activation_multiplier: float,
) -> float:
  """Estimate peak memory for a set of axis decisions.

  Axes with batch_size=0 (vmap) contribute their full cardinality to the
  memory product. Axes with batch_size>0 (safe_map) contribute only their
  tile size — one tile is live at a time.

  activation_multiplier must be supplied by the caller (no default).
  """
  product = 1
  for d in decisions:
    if d.batch_size == 0:
      product *= d.axis.cardinality
    else:
      product *= d.batch_size
  return base_shape_bytes * product * activation_multiplier


@dataclass(frozen=True)
class AxisSpec:
  """Describes one mappable axis for the batch planner."""

  name: str
  axis_index: int  # lower = innermost; greedy loop demotes in ascending order
  cardinality: int  # typical/max size of this axis
  default_batch_size: int  # 0 = vmap; positive = safe_map tile size
  tile_granularity: int  # safe_map tile sizes are rounded up to multiples of this
  heterogeneous: bool  # if True, element shapes may vary; vmap invalid; always safe_map
  doc: str
  dedup_eligible: bool = False  # if True, may be assigned DedupGather strategy


@dataclass(frozen=True)
class AxisDecision:
  """Planner output for one axis."""

  axis: AxisSpec
  batch_size: int  # 0 = vmap; positive = safe_map tile size
  reasoning: str
  strategy: AxisStrategy = dataclasses.field(default_factory=Vmap)


@dataclass(frozen=True)
class BatchPlan:
  """Planner output for a full set of axes."""

  decisions: list[AxisDecision]
  total_memory_estimate: float
  axes_by_index: dict[int, AxisSpec]
  budget_exceeded: bool

  def exceeded_budget(self) -> bool:
    """True when even minimum-tile decisions exceed the budget. Never raises."""
    return self.budget_exceeded

  def decision_for(self, name: str) -> AxisDecision:
    for d in self.decisions:
      if d.axis.name == name:
        return d
    raise KeyError(name)

  def log_summary(self) -> None:
    parts = [f"{d.axis.name}=bs:{d.batch_size}" for d in self.decisions]
    status = "EXCEEDED" if self.budget_exceeded else "ok"
    logger.debug(
      "BatchPlan [%s]: %s | estimate=%.1f bytes",
      status,
      ", ".join(parts),
      self.total_memory_estimate,
    )


@dataclass(frozen=True)
class BatchPlanner:
  """Host-side planner: decides vmap vs safe_map tile size per axis.

  estimate_memory is injected so the theoretical estimator can be swapped
  for an HLO-backed empirical model without changing this class.
  """

  axes: list[AxisSpec]
  budget_bytes: float
  estimate_memory: Callable[..., float]
  carries: list[CarrySpec] = dataclasses.field(default_factory=list)
  dedup_specs: list = dataclasses.field(default_factory=list)

  def plan(self) -> BatchPlan:

    sorted_axes = sorted(self.axes, key=lambda a: a.axis_index)
    carry_by_name = {c.axis_name: c for c in self.carries}

    # Phase 0: pre-demote axes with declared CarrySpec to Scan (before budget loop)
    phase0_decisions: list[AxisDecision] = []
    phase0_names: set[str] = set()
    for ax in sorted_axes:
      if ax.name in carry_by_name:
        cs = carry_by_name[ax.name]
        scan_strategy = Scan(
          init=cs.init,
          transition=cs.transition,
          ordered_sinks=cs.ordered_sinks,
        )
        phase0_decisions.append(
          AxisDecision(
            axis=ax,
            batch_size=1,
            reasoning=f"carry-bearing scan (CarrySpec declared for '{ax.name}')",
            strategy=scan_strategy,
          ),
        )
        phase0_names.add(ax.name)

    remaining = [ax for ax in sorted_axes if ax.name not in phase0_names]

    # Phase 0b: pre-demote dedup_eligible axes with declared DedupSpec to DedupGather
    dedup_by_name = {ds.axis_name: ds for ds in self.dedup_specs}
    phase0b_decisions: list[AxisDecision] = []
    dedup_names: set[str] = set()
    for ax in sorted_axes:
      if ax.name in dedup_by_name and ax.name not in phase0_names:
        ds = dedup_by_name[ax.name]
        dg_strategy = ds.to_dedup_gather()
        phase0b_decisions.append(
          AxisDecision(
            axis=ax,
            batch_size=ds.k,
            reasoning=f"dedup-gather (DedupSpec declared for '{ax.name}', k={ds.k}, k_bucket={dg_strategy.k_bucket})",
            strategy=dg_strategy,
          ),
        )
        dedup_names.add(ax.name)

    remaining = [ax for ax in remaining if ax.name not in dedup_names]

    # Phase 1: pre-demote heterogeneous axes (shapes vary; vmap invalid)
    decisions: list[AxisDecision] = list(phase0_decisions) + list(phase0b_decisions)
    het_names: set[str] = set()
    for ax in remaining:
      if ax.heterogeneous:
        tile = max(1, ax.tile_granularity)
        decisions.append(
          AxisDecision(
            axis=ax,
            batch_size=tile,
            reasoning="heterogeneous axis: element shapes vary; safe_map required",
            strategy=SafeMap(tile=tile),
          ),
        )
        het_names.add(ax.name)

    # Phase 2: greedy budget loop for homogeneous axes (innermost-first)
    homogeneous = [ax for ax in remaining if not ax.heterogeneous]
    hom_decisions: list[AxisDecision] = [
      AxisDecision(
        axis=ax,
        batch_size=0,
        reasoning="vmap (homogeneous, within budget)",
        strategy=Vmap(),
      )
      for ax in homogeneous
    ]
    for i, ax in enumerate(homogeneous):
      current = decisions + hom_decisions
      if self.estimate_memory(current) <= self.budget_bytes:
        break
      tile = max(1, ax.tile_granularity)
      hom_decisions[i] = AxisDecision(
        axis=ax,
        batch_size=tile,
        reasoning=f"demoted to safe_map tile={tile}: estimate exceeded budget",
        strategy=SafeMap(tile=tile),
      )

    all_decisions = decisions + hom_decisions

    # Post-plan eligibility guard: DedupGather may only be assigned to dedup_eligible axes.
    from prxteinmpnn.tiling.strategy import DedupGather as _DedupGather
    from prxteinmpnn.tiling.errors import TilingError

    for d in all_decisions:
      if isinstance(d.strategy, _DedupGather) and not d.axis.dedup_eligible:
        raise TilingError(
            f"Axis '{d.axis.name}' has dedup_eligible=False but was assigned "
            f"DedupGather strategy. Set dedup_eligible=True on the AxisSpec to allow dedup."
        )

    final_estimate = self.estimate_memory(all_decisions)
    exceeded = final_estimate > self.budget_bytes

    return BatchPlan(
      decisions=all_decisions,
      total_memory_estimate=final_estimate,
      axes_by_index={ax.axis_index: ax for ax in self.axes},
      budget_exceeded=exceeded,
    )
