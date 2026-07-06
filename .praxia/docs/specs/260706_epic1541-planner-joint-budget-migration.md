# Spec: migrate aminx.tiling.planner onto xtrax's joint-budget BatchPlanner

**Task:** 260706_epic1541-tiling-migration · **Status:** DRAFT, pending review · **Scope:** aminx only; consumes xtrax 0.4.0a1's joint-budget `BatchPlanner` as a fixed, already-shipped dependency. Companion to `xtrax`'s own spec, `.praxia/docs/specs/260706_joint-budget-batch-planner.md` (that repo).

**Supersedes-in-part**: `.praxia/docs/decisions/260706_planner-stays-on-aminx-tiling-by-design.md` (the algorithmic-gap blocker that decision recorded is closed by xtrax 0.4.0a1's `MemoryBudget`/`budget=` mode) and the `CarrySpec`/`DedupSpec` portion of `.praxia/docs/decisions/260706_carryspec-dedupspec-stay-local-carryshape-migrates.md` (their blocker — aminx's planner not consulting `heterogeneous_axes` — is closed once the planner routes through xtrax's `plan()`, which already checks it in Phase 0). Both docs are being annotated with a pointer to this spec rather than rewritten; the historical reasoning stands as "why this was blocked as of 2026-07-06," this spec is "why and how it's unblocked."

## Motivation

xtrax 0.4.0a1 (published to PyPI 2026-07-06, tag `v0.4.0a1`) ships a joint-budget
mode on `BatchPlanner`: pass `budget=MemoryBudget(bytes, estimate)` and the
planner runs the same greedy multi-axis demotion algorithm aminx's local
`BatchPlanner` hand-rolls today — start every eligible axis at `Vmap`, demote
one at a time (in spec order) until the whole-plan estimate fits, raise
`BudgetInfeasibleError` if it never does. This was the one piece of
`aminx.tiling.planner` that had no xtrax equivalent; recon on 2026-07-06 (see
the superseded decision doc) also found aminx's planner's Phase 0 blindly
pre-demotes `CarrySpec`-declared axes to `Scan` with **no** heterogeneous-axis
check, relying entirely on `CarrySpec.__post_init__`'s own eager (and
xtrax-incompatible) validation to catch the mistake. xtrax's `plan()` already
checks `heterogeneous_axes` in Phase 0 — routing aminx's planning through it
closes that gap too, for free, as a consequence of the planner move rather than
separate work.

Net: this spec migrates the planner, and `CarrySpec`/`DedupSpec` ride along.
`bucketing.py`'s `group_by_bucket`/`BucketAssignment` and `pad.py`'s
`pad_bundle` are **not** revisited by this spec — nothing about them changed;
they stay resolved as in `260706_bucketing-pad-stay-local-epic-1541-p3-scope-closed.md`.

## Current state (what's being replaced)

`src/aminx/tiling/planner.py`'s `BatchPlanner` (frozen dataclass, `axes`/`budget_bytes`/
`estimate_memory`/`carries`/`dedup_specs` as constructor fields, `.plan()` takes no
args) implements, in order: Phase 0 (CarrySpec → Scan, **no heterogeneity check**),
Phase 0b (DedupSpec → DedupGather), Phase 1 (heterogeneous axes → SafeMap), Phase 2
(greedy joint-budget demotion, sorted by `axis_index`, against `budget_bytes: float`
via `estimate_memory: Callable[[list[AxisDecision]], float]`). Returns a `BatchPlan`
with `decision_for(name)`, `log_summary()`, `exceeded_budget()` convenience methods
and a `budget_exceeded: bool` field — the plan is always returned, even over budget.

Production call graph: `host/plan.py`'s `make_sampling_planner()` (builds axes from
the `aminx.tiling.axes` registry, computes `budget_bytes = limit * headroom -
param_bytes`, calls the local planner) and `plan_bucketed()` (re-plans per bucket via
`dataclasses.replace(planner, axes=modified_axes)`); `host/kernel_dispatch.py`'s
`_dispatch_axis` (reads `.decision_for(AxisNames.X).strategy`, `isinstance`-checks
against `aminx.tiling.strategy.{Vmap,SafeMap,Scan,DedupGather}`); `run/specs.py`'s
`SamplingSpecification.carry_specs`/`dedup_specs` fields (typed against the local
`CarrySpec`/`DedupSpec`).

## Target xtrax API (0.4.0a1, verified against the published wheel, not the spec prose)

```python
# xtrax.tiling
AxisSpec(name, cardinality, default_batch_size, tile_granularity=1,
         heterogeneous=False, dedup_eligible=False, bucket_boundaries=None,
         role=AxisRole.KNOWN)                                    # no axis_index, no doc

MemoryBudget(bytes: int, estimate: Callable[[Sequence[AxisDecision]], int])
# __post_init__ rejects bool/non-int bytes, non-positive bytes, non-callable estimate

BatchPlanner(memory_estimator=None, carry_specs=None, dedup_specs=None,
             heterogeneous_axes=None, budget: MemoryBudget | None = None)
# budget and memory_estimator mutually exclusive -> ValueError at construction
.plan(specs: Sequence[AxisSpec]) -> BatchPlan   # BatchPlan = just `decisions: tuple[AxisDecision, ...]`
# no .decision_for()/.log_summary()/.exceeded_budget(), no budget_exceeded field

BudgetInfeasibleError(Exception)   # plain Exception, not a TilingError

# xtrax.tiling.estimators (helpers, not required, see "not adopted" below)
device_memory_budget(fraction=0.9, device=None) -> int
lowered_memory_estimate(fn, *abstract_args) -> int

CarrySpec(axis_name, init, transition, ordered_sinks=True)   # unchanged from 0.3.1: no eager heterogeneous check
DedupSpec(axis_name, unique_indices, index_map, k, dedup_fn=None, gather_fn=None)  # __post_init__ ADDS an index_map range check aminx's version lacks
```

## Behavior changes callers must accept (enumerated so nothing is silently swallowed)

1. **`budget_bytes` becomes an `int`.** aminx currently computes a `float`
   (`limit * headroom - param_bytes`). `MemoryBudget.__post_init__` rejects
   non-int. Fix: `int(...)` at the call site in `make_sampling_planner`, floor
   not round (never *increase* the effective budget).
2. **Silent over-budget plans become `BudgetInfeasibleError`.** Confirmed via
   grep (2026-07-06): no production code reads `budget_exceeded`/
   `exceeded_budget()` today — it's set and never inspected outside
   `planner.py` itself and test boilerplate. This is a strict tightening (a
   real problem that used to be silently ignored now raises), not a behavior
   regression, but it is a **new exception type in the call path** that
   `make_sampling_planner`'s callers were not written to expect. Translate at
   the boundary into an aminx-owned exception (see below) so existing
   `except TilingError` callers, if any, keep working; audit callers for a
   bare call with no handler and decide whether to add one.
3. **No `.decision_for(name)` method.** Replace with a module-level
   `decision_for(plan: BatchPlan, name: str) -> AxisDecision` helper
   (aminx-side, small, ~5 lines) at every call site (`extract_batch_sizes`,
   `kernel_dispatch.py`'s ~5 uses).
4. **No `.log_summary()`.** If this diagnostic is worth keeping, reimplement
   as an aminx-side function over `BatchPlan.decisions` directly; otherwise
   drop it (confirm no test/log-scraping depends on its exact format first).
5. **`dataclasses.replace(planner, axes=...)` breaks outright** —
   `BatchPlanner` is a plain class in xtrax, not a frozen dataclass, and
   `axes` isn't a constructor field (it's a `.plan(specs)` argument). Rewrite
   `plan_bucketed()` to reconstruct the modified axes list and call
   `planner.plan(modified_axes)` directly; the `BatchPlanner` instance itself
   (carry/dedup/heterogeneous/budget config) doesn't need per-bucket
   reconstruction, only the axes list passed to `.plan()`.
6. **`kernel_dispatch.py`'s `isinstance(strategy, DedupGather)` must target
   `xtrax.tiling.strategy.DedupGather`**, since `DedupSpec.to_dedup_gather()`
   will now construct xtrax's class. Same for any `isinstance` against
   `Vmap`/`SafeMap`/`Scan` in `_dispatch_axis` — audit all four.
7. **`AxisSpec` loses `axis_index` and `doc`.** Ordering becomes "pass specs
   pre-sorted" (aminx's `axes.py` `ALL_AXES` list is already in the right
   order — verify, don't assume). `doc` strings move to comments/a parallel
   lookup if still wanted for tooling; they're not load-bearing.
8. **xtrax's `DedupSpec.__post_init__` adds an `index_map` range check
   `[0, k)`** aminx's lacks — strictly additive validation, not a behavior
   change any caller could be relying on the absence of (a missing check
   can't be a documented contract).

## Not adopted in this migration (explicitly deferred)

- **`lowered_memory_estimate`** (XLA-compile-based real memory measurement) —
  strictly more accurate than aminx's current closed-form
  `estimate_memory_theoretical`, but wiring it in requires compiling a
  representative tile of the actual decode computation per candidate decision
  set (the estimator docstring itself flags this as "own slice" work). This
  spec keeps `estimate_memory_theoretical`'s existing math verbatim, now
  invoked through xtrax's engine instead of aminx's copy of the loop — a
  behavior-preserving delegation, not a numerical change. Adopting the
  compiler-backed estimator is a legitimate follow-up, tracked separately, not
  bundled into a migration whose job is to not change what the planner
  decides.
- **`device_memory_budget()`** — doesn't support subtracting `param_bytes`
  (only a flat `fraction`), and its fail-loud-on-no-memory-stats contract
  differs from aminx's current silent-4GiB-fallback. Keep aminx's existing
  `try/except` device-limit read for now; revisit once a caller actually needs
  the stricter contract.

## Migration plan

Mirrors the T2.4/T2.GATE naming already used in this epic.

**T-PLANNER.0 — dependency pin.** `pyproject.toml`: `xtrax==0.4.0a1` (exact
pin, per the alpha-consumer instruction — plain `>=` ranges resolve to 0.3.1
until 0.4.0 leaves prerelease). Confirm `uv lock` resolves without
`--prerelease` flags being required project-wide (an exact pin should not need
that; verify, don't assume).

**T-PLANNER.1 — `axes.py`.** Rewrite the 10 `AxisSpec` instances against
xtrax's dataclass (drop `axis_index`, `doc`; keep `ALL_AXES` list order as the
implicit ordering contract, verified against the current `axis_index` values
before deleting them).

**T-PLANNER.2 — `host/plan.py` core.** Rewrite `make_sampling_planner` to
build `MemoryBudget(bytes=int(...), estimate=estimate_memory_theoretical)` and
call `xtrax.tiling.BatchPlanner(budget=..., carry_specs=..., dedup_specs=...,
heterogeneous_axes=...)`. Add the `decision_for()` helper; update
`extract_batch_sizes`. Translate `BudgetInfeasibleError` into an
aminx-owned exception at this boundary (new `PlanBudgetInfeasibleError(TilingError)`
or reuse `PlanTopologyError` if semantically close enough — decide during
implementation, not preregistered here).

**T-PLANNER.3 — `plan_bucketed()`.** Rewrite the `dataclasses.replace`
pattern per behavior-change #5 above.

**T-PLANNER.4 — CarrySpec/DedupSpec cascade.** Swap `run/specs.py`'s imports
to `xtrax.tiling`. Update `kernel_dispatch.py`'s `isinstance` checks (behavior
change #6). Decide the fate of `aminx.tiling.carry.py`/`dedup.py` (delete now
that nothing production-side imports them, or keep as thin re-exports for one
release — lean delete, since `iterator.py`'s precedent this session was to
leave orphaned-but-tested modules in place only when something *else*
(`dispatch.py`'s legacy function) still needs them; nothing needs
`carry.py`/`dedup.py` once this lands). Update the 3 test files that construct
aminx's local `CarrySpec`/`DedupSpec` directly
(`test_carry_spec.py`, `test_planner_phase0.py`, `test_sampling_planner_carry_dedup.py`).

**T-PLANNER.GATE — parity.** Not a re-run of xtrax's own AC1–AC13 (those are
already verified upstream, per the xtrax spec, against xtrax's synthetic unit
tests) — this gate verifies aminx's *specific* registry/config produces
identical plans old-vs-new:
- Every `make_sampling_planner` scenario exercised by today's test suite
  (`test_planner_phase0.py`, `test_bucketing.py`, `test_bucketed_plan.py`,
  `test_safe_map_tile_resolution.py`), run through both the old local planner
  (kept temporarily, not yet deleted) and the new xtrax-backed one; assert
  identical `(strategy type, batch_size)` per axis.
- At least one scenario that would have silently returned
  `budget_exceeded=True` under the old planner — confirm the new path raises
  the translated exception instead, with a message containing the budget,
  final estimate, and per-axis strategy state (per xtrax's AC7).
- `plan_bucketed()` round-trip on a representative multi-bucket batch.
- Full existing suite (`tests/tiling/`, `tests/host/`, plus anything touching
  `kernel_dispatch.py`/sampling) green.

**Flip.** Only after the gate passes: delete `aminx.tiling.planner.py`'s
implementation (or reduce it to the thin `make_sampling_planner` wiring if
that's where it ends up living — TBD during implementation whether
`host/plan.py` absorbs it directly or a slimmed `planner.py` remains as a
one-function shim), delete `carry.py`/`dedup.py`, remove the old
`estimate_memory_theoretical`-in-a-hand-rolled-loop code path.

## Definition of done

- All behavior changes above are either preserved exactly (budget math,
  demotion order, divisibility warnings) or deliberately and visibly changed
  (fail-loud infeasibility) — no silent semantic drift.
- Full test suite green; parity gate scenarios pass.
- `aminx.tiling` module inventory after this lands: `axes.py` (data, now
  xtrax-typed), `strategy.py`+`dispatch.py`'s `_strategy_to_xtrax()` (status
  TBD — once the planner constructs xtrax-native strategies directly, audit
  whether anything still constructs aminx-native `Vmap`/`SafeMap`/`Scan`/
  `DedupGather`; if not, these become deletable too, which is a further
  reduction beyond this spec's stated scope but worth checking for once this
  lands), `bucketing.py`/`pad.py` (unchanged, per the closed decision),
  `errors.py` (unchanged, out of scope).
- Backlog `#1483` updated to reflect the now-more-accurate endgame (not
  drafted here — a call for after this lands, once the `strategy.py`
  question above is actually answered rather than speculated).

## Off-ramp

If T-PLANNER.GATE surfaces a real behavioral divergence that can't be closed
(e.g., xtrax's demotion order contract turns out to interact with aminx's
specific axis registry in some unanticipated way), stop at the gate — do not
flip. The old local planner is not deleted until the gate passes; this spec's
phases are individually revertable up through T-PLANNER.4.
