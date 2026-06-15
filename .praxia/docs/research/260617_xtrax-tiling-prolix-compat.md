---
title: xtrax.tiling CORE — prolix 6-axis planner compatibility
date: 260617
sprint: 260617_poe-rs3-pt-stagebundle-tiling-val
item: "#1599"
xtrax_sha: 8cbbe12
---

# xtrax.tiling / prolix compat — D7 falsifiable validation

## Verdict: PARTIAL PASS

**D7-part1 (planner accepts 6-axis plan): PASS**
xtrax.AxisSpec + xtrax.BatchPlanner can replicate prolix's 6-axis plan. Both heterogeneous
axes (n_mols, n_conformers) return batch_size > 0. All 4 homogeneous axes get Vmap strategy.

**D7-part2 (CORE non-leaky): FAIL**
`DedupGather` is exported in `xtrax.tiling.__all__`, violating the D4/D5 CORE/OPTIONAL split.
`test_no_dedup_gather` fails as expected. Gap must close before R1 DoD.

## Test results (xtrax SHA 8cbbe12)

```
tests/tiling/test_prolix_compat.py::TestProlixPlannerCompat::test_planner_accepts_6_axes         PASS
tests/tiling/test_prolix_compat.py::TestProlixPlannerCompat::test_n_mols_batch_size_positive     PASS
tests/tiling/test_prolix_compat.py::TestProlixPlannerCompat::test_n_conformers_batch_size_positive PASS
tests/tiling/test_prolix_compat.py::TestProlixPlannerCompat::test_homogeneous_axes_use_vmap_strategy PASS
tests/tiling/test_prolix_compat.py::TestCoreNonLeakyAssertions::test_no_io_callback_sink         PASS
tests/tiling/test_prolix_compat.py::TestCoreNonLeakyAssertions::test_no_dedup_gather             FAIL
```

## API surface mismatches

| Prolix field | xtrax equivalent | Notes |
|---|---|---|
| `default_batch_size=0` (vmap) | `batch_size=cardinality` | must set to full cardinality to get Vmap |
| `default_batch_size=1` (safe_map) | `batch_size=1` | triggers SafeMap via cardinality > 1 |
| `tile_granularity` | `granularity` | direct mapping |
| `axis_index` | (absent) | positional in plan() call list |
| `doc` | (absent) | not in xtrax.AxisSpec |
| `BatchPlan.decision_for(name)` | (absent) | test uses local helper `_decision_for` |
| `BatchPlanner(axes, budget_bytes, estimate_memory)` | `BatchPlanner(memory_estimator=None)` | xtrax takes specs at plan() call time |

## Gaps before R1 DoD

1. **DedupGather in CORE** (D4/D5 blocker): Move `DedupGather` to `xtrax.tiling.ext` or
   `xtrax.tiling.dedup`. Must not appear in `xtrax.tiling.__all__` under CORE import.

2. **BatchPlan.decision_for()** missing: prolix consumers need `plan.decision_for(name)`.
   Either add to BatchPlan or require callers to iterate `decisions` directly.

3. **Heterogeneous pre-demotion semantics**: xtrax incidentally SafeMap-assigns heterogeneous
   axes because `cardinality > batch_size=1`. prolix Phase 1 is an explicit pre-demotion pass.
   Incidental behavior sufficient for D7 but semantically weaker. Track as T2.5b.

## Test file

`/home/marielle/projects/xtrax/tests/tiling/test_prolix_compat.py`
