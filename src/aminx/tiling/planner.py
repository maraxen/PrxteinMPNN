from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from collections.abc import Sequence

  from xtrax.tiling import AxisDecision


def estimate_memory_theoretical(
  decisions: Sequence[AxisDecision],
  base_shape_bytes: float,
  activation_multiplier: float,
) -> float:
  """Estimate peak memory for a set of axis decisions.

  Vmap-strategy axes contribute their full cardinality to the memory
  product; all other strategies (SafeMap, Scan, DedupGather) contribute
  only their decided tile size (d.batch_size) -- one tile is live at a
  time.

  Checks type(d.strategy).__name__ == "Vmap" rather than d.batch_size == 0:
  xtrax.tiling.BatchPlanner's joint-budget mode (which this function is
  used with, as a MemoryBudget.estimate callable -- see host/plan.py's
  _plan_with_joint_budget) sets batch_size=spec.default_batch_size even for
  Vmap decisions, never 0 (EPIC #1541 T-PLANNER.2 finding, 2026-07-06). A
  d.batch_size == 0 check -- this module's own retired local BatchPlanner's
  convention -- would silently underestimate memory for every Vmap axis
  under xtrax's convention.

  activation_multiplier must be supplied by the caller (no default).

  Note: this is the one survivor of aminx.tiling.planner's original
  BatchPlanner/AxisSpec/AxisDecision/BatchPlan/ceil_to_granularity/
  resolve_safe_map_tile -- those migrated to xtrax.tiling.BatchPlanner's
  joint-budget mode (see host/plan.py's _plan_with_joint_budget and
  .praxia/docs/specs/260706_epic1541-planner-joint-budget-migration.md).
  This function's own math is unchanged and is now invoked through xtrax's
  engine rather than this module's retired copy of the demotion loop.
  """
  product = 1
  for d in decisions:
    if type(d.strategy).__name__ == "Vmap":
      product *= d.spec.cardinality
    else:
      product *= d.batch_size
  return base_shape_bytes * product * activation_multiplier
