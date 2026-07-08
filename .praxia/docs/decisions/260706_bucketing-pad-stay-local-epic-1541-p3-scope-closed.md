---
title: bucketing.py / pad.py stay local — EPIC #1541 P3 scoping closed
status: Accepted
date: 2026-07-06
related_epic: "#1541 (aminx→xtrax tiling refactor), gate #1556 (T2.GATE)"
---

# Decision

`aminx.tiling.bucketing` (`BucketingConfig`, `select_bucket`, `group_by_bucket`,
`BucketAssignment`) and `aminx.tiling.pad` (`pad_bundle`) stay on `aminx.tiling`,
by design, indefinitely. This closes out the last named item in EPIC #1541 P3's
remaining-pieces list ("bucketing/pad").

## bucketing.py

Two consumers, both already resolved as local for reasons unrelated to bucketing
itself:

- `host/plan.py`'s `plan_bucketed()` uses `group_by_bucket`/`BucketAssignment` to
  produce per-bucket `aminx.tiling.planner.BatchPlan` objects — a companion to the
  planner, which already stays local (`260706_planner-stays-on-aminx-tiling-by-design.md`).
  Same reasoning as `CarrySpec`/`DedupSpec`
  (`260706_carryspec-dedupspec-stay-local-carryshape-migrates.md`).
- `inference/bundle_builder.py` uses `select_bucket`/`BucketingConfig` standalone
  (no `BatchPlanner` involvement) to pick a bucket ceiling before calling `pad_bundle`.
  xtrax's `select_bucket(length, boundaries)` is a structurally compatible drop-in
  here (same "smallest boundary >= length" contract, different signature: boundaries
  tuple vs a `BucketingConfig` wrapper) — but migrating it alone has no payoff: its
  inseparable partner, `pad_bundle`, cannot migrate (see below), so the two would
  still call into two different libraries for one atomic decision+pad step. Not
  worth the churn for a one-line helper with no operational gain.

## pad.py

`pad_bundle(bundle, target_length)` was never a real migration candidate. It is
irreducibly aminx/MPNN domain logic: hand-written, per-field, per-axis padding of
`InferenceBundle`'s ~15 named fields (`geometry.coords` padded on axis 1, `wave.group_ids`
padded on axis 0, etc. — each field has different padding semantics). xtrax's
`bucketize(xs, bucket_size)` is generic but pads *every* pytree leaf's *leading* axis
uniformly — wrong for most of `InferenceBundle`'s fields, which need axis-1 (not
axis-0) padding. There is no way to express `pad_bundle`'s actual behavior as a
`bucketize()` call. Same category as `host/plan.py`'s Rule 3 (STEDecode check):
pure domain logic that happens to live under `aminx.tiling`, not a tiling primitive.

## EPIC #1541 P3 scoping: closed

All items from the P3 remaining-pieces list are now resolved:

| Item | Outcome |
|---|---|
| Dispatch layer (`factory.py`) | Migrated (T2.4 + T2.GATE, prior PRs) |
| Direct iterator construction (5 sites) | Migrated (PR #88) |
| `CarryShape` | Migrated (this session) |
| Planner (`BatchPlanner`/`AxisSpec`) | Won't-migrate — algorithmic gap |
| `CarrySpec` / `DedupSpec` | Won't-migrate — companions to the local planner |
| `bucketing.py` / `pad.py` | Won't-migrate / not applicable — companions to the local planner, or pure domain logic |
| `aminx.tiling.errors` (`TilingError`) | Out of scope from the start |

## Consequence: `aminx.tiling` will NOT be deleted

The original spec's stated endgame — "once all pieces flip, `aminx.tiling` can be
deleted and backlog #1483 closed" — does not hold. Confirmed permanent residents of
`aminx.tiling`: `planner.py`, `axes.py`, `carry.py`, `dedup.py`, `bucketing.py`,
`pad.py`, `strategy.py` (aminx's own `Vmap`/`SafeMap`/`Scan`/`DedupGather` vocabulary,
which T2.4's `_strategy_to_xtrax()` translates *from* — required for as long as the
local planner constructs these), `errors.py`, and `dispatch.py` (holds both the
now-production-dead-but-deliberately-retained legacy `make_axis_dispatch` — kept as
the T2.GATE parity baseline — and the T2.4 translation adapter).

Only `iterator.py` and `carry_shape.py` have zero remaining production call sites
(confirmed via grep, 2026-07-06) — but neither can be *deleted*: `iterator.py` is
still imported by `dispatch.py`'s legacy `make_axis_dispatch` (retained deliberately
as the parity/regression baseline the T2.GATE test suites compare against), and both
modules are still directly unit-tested for their own sake (`test_iterator.py`,
`test_carry_shape.py`).

**Recommend**: update backlog `#1483` to reflect this — either close it as
"resolved, not as originally scoped" (the flip is complete; the deletion goal was
never achievable given the planner's real requirements) or retitle it to track only
the two now-production-orphaned-but-still-tested modules, if there's a future
appetite to decide those tests/modules' fate. Not acted on here — a backlog/scope
call, not a code change.
