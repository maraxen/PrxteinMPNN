---
title: BatchPlanner/AxisSpec stays on aminx.tiling.planner — not migrated to xtrax.tiling.plan
status: Accepted
date: 2026-07-06
related_epic: "#1541 (aminx→xtrax tiling refactor), gate #1556 (T2.GATE)"
---

# Decision

`aminx.tiling.planner.BatchPlanner` / `AxisSpec` (`src/aminx/tiling/planner.py`) will **not** be
migrated to `xtrax.tiling.plan.BatchPlanner` / `AxisSpec` as part of EPIC #1541's P3 call-site
migration. The planner stays local, indefinitely, by design — not deferred, not blocked.

This applies only to the planning *algorithm* (`BatchPlanner.plan()`). The dispatch-layer flip
(T2.4's `make_axis_dispatch_via_xtrax`, already landed in `factory.py`) is unaffected and unrelated;
this decision does not reopen or roll back that work.

## Why

Scoping this migration piece (recon, 2026-07-06) surfaced a real algorithmic gap, not a structural
one:

- **xtrax's `BatchPlanner._decide_strategy`** decides each axis *independently*: cardinality vs.
  `default_batch_size`, with an optional per-axis `memory_estimator(spec) -> bytes` checked against
  the *hardware's* `jax.devices()[0].memory_stats()['bytes_limit']`.
- **aminx's `BatchPlanner.plan()`** runs a *joint, multi-axis greedy demotion loop* (Phase 2) against
  a *caller-supplied* `budget_bytes` (`headroom × device_limit − param_bytes`), demoting axes from
  Vmap→SafeMap one at a time in `axis_index` order until the **combined product** across all axes'
  decisions fits. `estimate_memory_theoretical` takes the whole decision list, not one axis.

These are different algorithms. They agree when every axis independently fits its budget, and
diverge under real multi-axis memory pressure — exactly the case the planner exists to handle. No
adapter can translate one into the other; only reimplementing aminx's loop on top of xtrax's
per-axis primitive, or upstreaming the joint-budget algorithm into xtrax itself, would preserve
behavior. Neither is a translate-and-wrap job like T2.4 was.

Two secondary findings reinforce this:

1. `plan_bucketed()` (`host/plan.py:780`) does `dataclasses.replace(planner, axes=modified_axes)` —
   relies on aminx's `BatchPlanner` being a frozen dataclass with `axes` as a field. xtrax's
   `BatchPlanner` is a plain class; axes are passed to `.plan(specs)`, not held as a field.
   `dataclasses.replace` on it is a hard `TypeError`.
2. `host/kernel_dispatch.py`'s `_dispatch_axis` is a **second, independent** consumer of
   `AxisDecision.strategy` — it hand-rolls its own `isinstance(strategy, Vmap/SafeMap/Scan/DedupGather)`
   handling against `aminx.tiling.strategy` classes directly (via `aminx.utils.safe_scan`), and never
   goes through `tiling/dispatch.py` / T2.4's adapter at all. Migrating the planner alone wouldn't
   flip anything real here without also touching this second dispatcher.

This follows the same precedent already set in `_validate_plan_topology` (`host/plan.py:257`): generic,
reusable checks (Rules 1-2) delegate to `xtrax.stages.validate_plan_topology`; the domain-specific
check (Rule 3, STEDecode/UnconditionalDecodeStep) stays local. The joint-budget demotion loop is
aminx's own memory-budgeting concern — genuinely useful, but not a generic tiling primitive xtrax's
architecture provides today.

## What this closes out

- The "planner" line item in EPIC #1541 / P3's remaining-pieces list is resolved as **won't-migrate**,
  not pending.
- `AxisSpec`'s cosmetic field drift (aminx has `axis_index`/`doc`; xtrax has `bucket_boundaries`/`role`)
  is not worth chasing on its own — xtrax's `bucket_boundaries` would be a free win (aminx's planner
  has a literal `# TODO: introduce bucketing management`, never built) but isn't reason enough to
  migrate the whole planner.
- Remaining P3 pieces (per the EPIC #1541 handoff): the 5 direct `aminx.tiling.iterator` construction
  sites, then `carry`/`carry_shape`/`dedup`, then `bucketing`/`pad`.

## Reopening this decision

If xtrax later grows an equivalent joint-budget, caller-supplied-budget demotion algorithm (upstream
feature, not scoped here), revisit. Until then, `aminx.tiling.planner` is the intended long-term home
for this logic, not a shim awaiting migration.
