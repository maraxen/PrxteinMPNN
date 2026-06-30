"""Plan audit for aminx.tiling.BatchPlanner, via xtrax.eda.

aminx's BatchPlan/AxisDecision/AxisSpec satisfy xtrax.eda's structural
Protocols (AxisDecisionLike/AxisSpecLike/BatchPlanLike) directly -- no
conversion to xtrax-native objects needed (AxisDecision.spec is a property
alias for the existing `axis` field; see planner.py). This module just
re-exports xtrax.eda's audit functions for discoverability from aminx.tiling.

Example
-------
>>> from aminx.tiling.eda import explain
>>> plan = my_batch_planner.plan()
>>> stats = explain(plan)
>>> for axis in stats["axes"]:
...     print(axis["name"], axis["strategy"], axis["reasoning"])

Known limitation: aminx has no Bucket AxisStrategy variant (bucketing is
handled separately via aminx.tiling.bucketing + host/plan.py, not as a
BatchPlanner strategy choice) -- bucket_stats will always be empty for
aminx plans. This is accurate, not a bug.
"""

from __future__ import annotations

from xtrax.eda import explain_plan as explain
from xtrax.eda import extract_plan_stats as extract_stats

__all__ = ["explain", "extract_stats"]
