---
created: 260616
task: R7-1 (#1926)
status: gate-delivered
---

# AxisSpec Field Mapping — R7-1 Gate

Canonical field comparison across xtrax and aminx AxisSpec implementations.
Naming decisions here gate RS-6 (R6-1) field names.

## Source Definitions

- **xtrax AxisSpec**: `xtrax/src/xtrax/tiling/plan.py:27–49`
- **aminx AxisSpec**: `aminx/src/aminx/tiling/planner.py:49–60`

## Three-Column Field Table

| xtrax field | aminx field | canonical decision |
|---|---|---|
| `name: str` | `name: str` | `name` |
| `cardinality: int` | `cardinality: int` | `cardinality` |
| `batch_size: int` | `default_batch_size: int` | **`default_batch_size`** |
| `granularity: int = 1` | `tile_granularity: int` | **`tile_granularity`** |
| `heterogeneous: bool = False` | `heterogeneous: bool` | `heterogeneous` |
| `dedup_eligible: bool = False` | `dedup_eligible: bool = False` | `dedup_eligible` |
| `bucket_boundaries: tuple[int, ...] \| None = None` | *(absent)* | `bucket_boundaries` (xtrax-only) |
| *(absent)* | `axis_index: int` | `axis_index` (aminx-only, ordering metadata) |
| *(absent)* | `doc: str` | `doc` (aminx-only, free-text) |

**Correction vs sprint plan template:** `dedup_eligible` is present in aminx AxisSpec at `planner.py:60` (`bool = False` default) — the sprint template incorrectly listed it as absent in aminx. Both implementations agree on this field name and default.

## Naming Decisions

### Conflict 1 — `batch_size` vs `default_batch_size`

**Canonical: `default_batch_size`**

Evidence: `xtrax/tiling/plan.py:89` defines `AxisDecision.batch_size: int` as the **output** field for the resolved tile size. Using `default_batch_size` on the AxisSpec **input** eliminates reader ambiguity between "spec hint" and "planner output." Semantic: `0 = vmap`, positive = safe_map tile size — the `default_` qualifier makes this explicit.

### Conflict 2 — `granularity` vs `tile_granularity`

**Canonical: `tile_granularity`**

Evidence: the codebase has multiple granularity concepts (bucket granularity, sequence granularity). `tile_granularity` names the safe_map alignment role precisely. aminx planner.py comment: "safe_map tile sizes are rounded up to multiples of this." Keeps usage unambiguous across future RS-7+ work.

## RS-6 RunSpec Axis Field Names

**GATE DECISION for Track A (R6-1):** Use ONLY these canonical kwarg names in `xtrax/run/spec.py` and any `AxisSpec` instantiation within the RS-6 sprint:

- `default_batch_size=` — canonical (replaces xtrax `batch_size=`; avoids collision with `AxisDecision.batch_size` output field)
- `tile_granularity=` — canonical (replaces xtrax `granularity=`; disambiguates from bucket/sequence granularity)

All other fields are unambiguous and carry forward unchanged: `name`, `cardinality`, `heterogeneous`, `dedup_eligible`, `bucket_boundaries` (xtrax-only), `axis_index` (aminx-only), `doc` (aminx-only).

## Future Work (R7-2)

The xtrax `AxisSpec` class itself still uses the old names (`batch_size`, `granularity`). Migrating xtrax internals to canonical names requires updating all xtrax call sites, planner logic, tests, and publishing a new xtrax version. This is **R7-2** — separate task, not in this sprint.
