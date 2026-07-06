---
title: CarrySpec/DedupSpec stay on aminx.tiling — CarryShape migrates to xtrax.tiling
status: Superseded (in part) — see 260706_epic1541-planner-joint-budget-migration.md
date: 2026-07-06
related_epic: "#1541 (aminx→xtrax tiling refactor), gate #1556 (T2.GATE)"
---

> **2026-07-06, later same day**: the `CarrySpec`/`DedupSpec` portion of this
> decision is superseded — their blocker (aminx's planner not consulting
> `heterogeneous_axes`) is closed once the planner itself routes through
> xtrax's `BatchPlanner.plan()` (which already checks it in Phase 0), per
> `../specs/260706_epic1541-planner-joint-budget-migration.md`. The
> `CarryShape` portion of this decision (migrated, unaffected) still stands.

# Decision

Of EPIC #1541 P3's "carry/carry_shape/dedup" remaining piece, only `CarryShape`
(`src/aminx/tiling/carry_shape.py`) migrates to `xtrax.tiling.CarryShape`. `CarrySpec`
(`src/aminx/tiling/carry.py`) and `DedupSpec`/`get_k_bucket` (`src/aminx/tiling/dedup.py`)
stay on `aminx.tiling`, by design, indefinitely — same treatment as the planner
(`260706_planner-stays-on-aminx-tiling-by-design.md`).

## CarryShape — migrated

Pure decode-mode metadata (name/shape/dtype + `.materialize()`), unrelated to
`BatchPlanner`. Byte-identical to `xtrax.tiling.CarryShape` (only a cosmetic
return-type annotation differs: `Shaped[jax.Array, ...]` vs `jax.Array`). Same shape
as the iterator.py flip: clean drop-in, not an adapter.

Flipped in `factory.py` and `autoregressive.py` (the only 2 production call sites).
`test_autoregressive.py` updated to import from `xtrax.tiling` too — `CarryShape` is a
concrete frozen dataclass, not a `Protocol`, so (unlike `MapIterator`/`ScanIterator`)
`ty` treats aminx's and xtrax's versions as distinct nominal types even though the
fields are identical; confirmed via `ty check` (`Expected xtrax.tiling.carry_shape.CarryShape,
found tiling.carry_shape.CarryShape` on `AutoregressiveDecode(wave_carry=...)` until fixed).
`test_carry_shape.py` (which tests the local module directly, not through
`AutoregressiveDecode`) is untouched — same reasoning as `test_iterator.py` in the
iterator migration.

## CarrySpec — stays local

`CarrySpec`'s only consumer is `aminx.tiling.planner.BatchPlanner.plan()`'s Phase 0
(already decided to stay local). aminx's `CarrySpec.__post_init__` eagerly validates
`axis_name` against a hardcoded `_HETEROGENEOUS_AXIS_NAMES = frozenset({"n_states",
"n_structures"})` and raises `ValueError` at construction time
(`test_carry_spec.py::test_carry_spec_rejects_heterogeneous_axis_name` /
`test_carry_spec_rejects_other_known_heterogeneous` lock this in).

xtrax's `CarrySpec` has no such check — it explicitly delegates heterogeneous-axis
rejection to `BatchPlanner.plan()`'s `heterogeneous_axes` parameter, checked at
plan-time. But aminx's own planner (Phase 0) blindly pre-demotes any axis with a
matching `CarrySpec` to `Scan` with no heterogeneity check at all — the *only* guard
against `Scan`-on-heterogeneous for CarrySpec-declared axes today is `CarrySpec`'s own
eager `__post_init__`. Migrating to xtrax's version would silently drop this guard
with no replacement (aminx's planner doesn't consult a `heterogeneous_axes` set), and
would break the two tests above outright (xtrax's `CarrySpec` raises nothing).

## DedupSpec — stays local

Same reasoning, different mechanism. `DedupSpec`'s only consumer is
`aminx.tiling.planner.BatchPlanner.plan()`'s Phase 0b. Its `to_dedup_gather()` method
constructs a `DedupGather` strategy by importing directly from
`aminx.tiling.strategy` (`from aminx.tiling.strategy import DedupGather, ...`).
Swapping to xtrax's `DedupSpec` would make `to_dedup_gather()` construct
`xtrax.tiling.strategy.DedupGather` instead — which would then flow into aminx's
(local) planner's `AxisDecision.strategy` output. `host/kernel_dispatch.py:76`'s
`isinstance(strategy, DedupGather)` checks against aminx's own class
(`from aminx.tiling.strategy import DedupGather` at line 58) — an xtrax-native
instance would silently fail that check and break dedup dispatch at runtime, with
no error until someone actually exercises the dedup path.

(xtrax's `DedupSpec` does have one additive safety improvement aminx's lacks — an
`index_map` range check in `[0, k)` — but that's not enough reason to take on the
class-identity break above. If we ever want that specific extra check, port it
into aminx's own `DedupSpec.__post_init__` directly; no migration needed for that.)

## What this closes out

- EPIC #1541 P3's "carry/carry_shape/dedup" line item is resolved: `CarryShape`
  done (migrated), `CarrySpec`/`DedupSpec` won't-migrate (same status as the planner).
- Remaining P3 piece: `bucketing`/`pad`.
- `aminx.tiling.errors` (`TilingError`) remains explicitly out of scope, as before.

## Reopening this decision

If `aminx.tiling.planner`'s Phase 0/0b logic is ever rewritten to consult a
`heterogeneous_axes`-style parameter (mirroring xtrax's), or to construct strategies
generically rather than importing `aminx.tiling.strategy` directly, revisit whether
`CarrySpec`/`DedupSpec` can migrate too. Until then they're companions to a
local-by-design planner, not shims awaiting migration.
