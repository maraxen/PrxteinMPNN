# Epic: RunSpecification Refactor + xtrax.run Foundation

**Date:** 260615 | **Status:** Specced, sprint-ready  
**Session:** 260616 design session — full conversation grounding in transcript

---

## Overview

Three-phase epic to (1) fix the RunSpecification inheritance hierarchy, (2) promote the Fuse/Tap/Sink boundary protocols to xtrax and establish xtrax.run, and (3) converge AxisSpec implementations across aminx/prolix/xtrax. These land in stages because phase 2 changes the xtrax public surface and phase 3 requires behavioral-parity tests.

**Critic pass:** c9ce19bb — 5 must-fix gaps filtered to 4 real changes (F2/M1 rejected: wrong premises — RunSpecification is @dataclass not eqx.Module; no mypy). See session for full critic output.

---

## Decided Design

### Three-stage runtime pattern (already exists, needs naming)

```
RunSpecification          ← serializable config, plain @dataclass, TOML-roundtrippable
    ↓ build_run_spec()
RunSpec (eqx.Module)      ← execution shape, typed sub-configs
    ↓ prep_protein_stream_and_model()
RuntimeBundle             ← materialized: loaded inputs + model + BatchPlan + sinks (unnamed today)
```

### Clean base (post-refactor)
RunSpecification base carries:
- Model identity: `model_weights`, `model_version`, `model_family`, `checkpoint_id`, `model_local_path`, `checkpoint_registry_path`
- Input reference: `inputs` (long-term: `inputs_path: Path`)
- Input constraints: `tied_positions`, `pass_mode`, `tie_group_map`, `structure_mapping`, `multi_state_temperature`, `ar_mask`
- **Conditioning** (moved from SamplingSpec): `sidechain_conditioning: bool = False`, `fixed_mask: ArrayLike | None = None`
- Resource config: `host_resource_allocation_strategy`, `ram_budget_mb`, `max_workers`, `n_devices`
- Sequence handling: `max_length`, `truncation_strategy`
- Data: `use_preprocessed`, `preprocessed_index_path`, `split`, `max_buffer_size`
- Misc: `random_seed`, `chain_id`, `model`, `altloc`, `decoding_order_fn`, `conformational_states`, `cache_path`, `output_dir`, `overwrite_cache`

**NOT on base:** campaign/grid fields (already on SamplingSpec), `output_h5_path` (deprecated, per-subclass for now), fn fields (on specific subclasses).

### Noise field design (RS-6)
Replace `backbone_noise`/`backbone_noise_mode`/`estat_noise`/`estat_noise_mode`/`vdw_noise`/`vdw_noise_mode`/`use_electrostatics`/`use_vdw` with:

```python
@dataclass(frozen=True)
class FeatureNoiseBundle:
    feature: Literal["backbone", "electrostatics", "vdw"]
    levels: Sequence[float]
    mode: Literal["direct", "thermal"] = "direct"
    noise_fn: NoiseFn | None = None  # callable override; replaces noise_type Literal
```

`noise: list[FeatureNoiseBundle]` on base replaces 8 scattered fields. Feature type + params + fn in one cohesive unit.

**FeatureNoiseBundle is a frozen dataclass** (not eqx.Module). It lives on `RunSpecification` (plain @dataclass), not on `RunSpec` (eqx.Module). `build_run_spec` transforms it into appropriate eqx.Module-aware structures as needed.

**Gate for R6-2:** A field mapping table documenting all 8 old noise fields → FeatureNoiseBundle attributes must be produced before R6-2 implementation begins. Path: `.praxia/docs/research/260616_noise-field-map.md`. Same gate pattern as R7-1 AxisSpec mapping. The table must account for all 8 fields (`backbone_noise`, `backbone_noise_mode`, `estat_noise`, `estat_noise_mode`, `vdw_noise`, `vdw_noise_mode`, `use_electrostatics`, `use_vdw`) — silent field drops are the failure mode.

### xtrax module structure (post RS-6)
```
xtrax.stages.boundaries   ← NEW FILE: Fuse/Tap/Sink/AxisBoundary (promoted from aminx)
                             Reconcile with FuseFn in stages/protocols.py
xtrax.run                 ← NEW MODULE
  RunSpec(eqx.Module)       seed, axes: list[AxisSpec], carry_specs, boundaries
  SinkSpec                  after_axis: str, factory: SinkFactory
  SinkFactory               tagged union: JsonlSink, NullSink, CallbackSink (H5Sink deprecated)
  InputResolver             Protocol[Spec, Bundle]
  SinkFn / AggregationFn    Protocols
```

Dependency order: `run → {stages, tiling}` — acyclic.

### Literal → Callable migration (RS-6)
Literal stays on serializable spec. Resolved to Callable at materialize stage via registry.
Pattern already correct at `host/plan.py:649` (`make_stage_set`).

```python
# on subclasses that need it (Scoring, Sampling, Jacobian — not Inspection)
encoding_aggregation_fn: AggregationFn = jnp.mean   # replaces average_encoding_mode Literal
decode_fn: DecodeFn | None = None                    # replaces decode_fn: Any
# backward compat shim:
class AveragingMode(enum.Enum):
    MEAN = "mean"
    def to_fn(self) -> AggregationFn: return jnp.mean
```

### Logical grouping (RS-6)
One level of sub-config. No deeper nesting.
- `noise: list[FeatureNoiseBundle]` — replaces 8 noise fields
- `model: ModelConfig` — identity fields
- `resources: ResourceConfig` — host allocation fields
- `SinkConfig` per-subclass — bridge from `output_h5_path` to `SinkSpec`
- Conditioning fields remain flat (sidechain_conditioning, fixed_mask, ar_mask, tied_positions, multi_state_*)

---

## Sprint 260618 — Hygiene P1

### Item H1: Move `sidechain_conditioning` + `fixed_mask` to RunSpecification base

**Files:** `aminx/run/specs.py`

Remove from `SamplingSpecification` (lines 316, 336). Add to `RunSpecification` defaulted-field block (145–225), **before** `run_spec = field(init=False)` at line 226:

```python
# Conditioning
sidechain_conditioning: bool = False
fixed_mask: ArrayLike | None = None
```

**Safe because:** `build_run_spec` reads via `getattr(..., "sidechain_conditioning", False)` (MRO-agnostic). `spec_json.py:186` already includes `fixed_mask` in array-coercion set.

**Gate:** `uv run pytest` passes. Add round-trip test: construct `ScoringSpecification(fixed_mask=<array>)`, assert field survives JSON round-trip.

**LOC:** ~8

---

### Item H2: Fix chain_mask hardcode (`_sampling_helper.py:346`)

**Files:** `aminx/host/_sampling_helper.py`

Replace hardcoded `chain_mask = jnp.ones((batch_size, seq_len), dtype=jnp.float32)` with:

```python
if spec.fixed_mask is not None:
    fixed_mask_np = _broadcast_per_structure(
        spec.fixed_mask, batch_size=batch_size, expected_len=seq_len,
        dtype=jnp.float32, name="fixed_mask"
    )
    chain_mask = 1.0 - fixed_mask_np  # _broadcast_per_structure guarantees float32
else:
    chain_mask = jnp.ones((batch_size, seq_len), dtype=jnp.float32)
```

**Critical:** Do NOT write `1 - spec.fixed_mask` directly. `spec.fixed_mask` is raw user input — possibly 1-D, unpadded, not yet a jnp array. Use `_broadcast_per_structure` (already exists at lines 448–457 in `_prepare_fixed_controls`). Convention: `chain_mask` uses `1=designable, 0=fixed`; `fixed_mask` uses `1=fixed` — hence complement.

**Gate:** `fixed_mask=None` → all-ones of `(batch_size, seq_len)`; `fixed_mask=jnp.ones((1,8))` → all-zeros broadcast to `(batch_size, 8)`; `assert chain_mask.dtype == jnp.float32` (guard against silent uint8→float64 promotion from un-cast mask inputs).

**LOC:** ~15

---

### Item H3: Fix `_sync_run_spec` double-fire

**Files:** `aminx/run/specs.py`

`build_run_spec` currently fires 2–3× per construction (base `__post_init__` line 259, then each subclass override lines 302, 413, 441, 483). Fix with guard flag:

```python
# on RunSpecification base, in defaulted block before run_spec:
_run_spec_dirty: bool = field(init=False, default=True, repr=False)

def _sync_run_spec(self) -> None:
    if not self._run_spec_dirty:
        return
    object.__setattr__(self, "_run_spec_dirty", False)
    object.__setattr__(self, "run_spec", build_run_spec(self))
```

Each subclass `__post_init__` **suppresses** the base call by setting False before `super()`, runs its own validation, then re-enables and fires once:
```python
def __post_init__(self) -> None:
    object.__setattr__(self, "_run_spec_dirty", False)  # suppress base's _sync_run_spec call
    super().__post_init__()
    # ... subclass validation ...
    object.__setattr__(self, "_run_spec_dirty", True)   # re-enable
    self._sync_run_spec()                               # fires once, post-validation
```

Base's `_sync_run_spec()` call sees `dirty=False` and returns early. Most-derived class's call sets `dirty=True` then fires the build. **Critical:** setting `dirty=True` before `super()` (the opposite order) would cause the BASE to fire before subclass validation runs — producing a run_spec built from a partially-constructed spec.

**Gate:** Mock `build_run_spec`, assert exactly 1 call for `SamplingSpecification(...)`.

**LOC:** ~20

---

### Item H4: Promote Fuse/Tap/Sink/AxisBoundary to `xtrax/stages/boundaries.py`

**Files:**
- `xtrax/src/xtrax/stages/boundaries.py` (new — canonical definition)
- `xtrax/src/xtrax/stages/__init__.py` (add exports)
- `xtrax/src/xtrax/stages/protocols.py` (deprecate FuseFn; add `__getattr__` shim)
- `aminx/src/aminx/types/boundaries.py` (become re-export shim: `from xtrax.stages.boundaries import Fuse, Tap, Sink, AxisBoundary`)
- `aminx/src/aminx/types/stages.py:27` (update import)

**Protocol signatures (preserve exactly):**
```python
@runtime_checkable
class Fuse(Protocol, Generic[S, O]):
    def __call__(self, stacked: S) -> O: ...

@runtime_checkable
class Tap(Protocol, Generic[T]):
    ordered: bool
    def __call__(self, x: T) -> T: ...

@runtime_checkable
class Sink(Protocol, Generic[T]):
    ordered: bool
    def __call__(self, x: T) -> None: ...

class AxisBoundary(eqx.Module):
    fuse: Fuse | None = eqx.field(static=True, default=None)
    tap: Tap | None = eqx.field(static=True, default=None)
    sink: Sink | None = eqx.field(static=True, default=None)
```

**FuseFn reconciliation:** `stages/protocols.py:39` has `FuseFn(Protocol[PerItem, Combined])` — same concept, different name. Resolution: `Fuse` in `boundaries.py` is the canonical name; `FuseFn` in `protocols.py` becomes a deprecated alias via module-level `__getattr__`:

```python
# protocols.py — after H4 lands
import warnings
def __getattr__(name: str):
    if name == "FuseFn":
        warnings.warn(
            "FuseFn is deprecated; import Fuse from xtrax.stages.boundaries instead.",
            DeprecationWarning, stacklevel=2,
        )
        from xtrax.stages.boundaries import Fuse
        return Fuse
    raise AttributeError(name)
```

Do NOT silently merge or leave both as independent definitions. The DeprecationWarning is required — passive spec comments are insufficient; the inconsistency must be machine-detectable during the H4→R6-4 coexistence window.

**Do NOT** move the topology validator (`host/plan.py:263–297`) into boundaries.py — protocol declaration and enforcement are separate layers.

**Scope of H4 FuseFn reconciliation — aminx FuseFn is OUT OF SCOPE:** `aminx/types/stages.py:68` defines a SEPARATE `FuseFn(Protocol[PerItem, Combined])` for logit transform functions (not a boundary protocol — it has an extra `bias: PerItem | None` parameter and is used by `LogitTransformFn`, `ARLogitTransformFn`, `TieGroupFuseFn`). H4 does NOT touch, deprecate, or rename this aminx FuseFn. Only `xtrax/stages/protocols.py:39`'s `FuseFn` (the boundary protocol) is reconciled.

**Gate:** `from xtrax.stages import Fuse, Tap, Sink, AxisBoundary` in both xtrax and aminx. `isinstance(concrete_sink, Sink)` returns True. `from aminx.types.boundaries import Fuse` succeeds without DeprecationWarning (shim is clean). `from xtrax.stages.protocols import FuseFn  # noqa: TID251` raises DeprecationWarning (the `TID251` suppression is required in the gate test because `pyproject.toml:142` bans deep imports from `xtrax.stages.protocols` — the test must deliberately exercise the deprecated path). aminx tests pass unchanged.

**LOC:** ~50 (new file ~30, plumbing ~20)

---

## Sprint RS-6 — xtrax.run + Fn Migration + Grouping

### R6-1: Create `xtrax/run/` module
- `run/spec.py` — `RunSpec(eqx.Module)`: `seed`, `axes`, `carry_specs`, `boundaries`
- `run/sink.py` — `SinkSpec`, `SinkFactory`, `SinkFn`, `NullSink`, `CallbackSink`, `JsonlSink`
- `run/resolver.py` — `InputResolver`: callable protocol `(spec: RunSpec, bundle: RuntimeBundle) -> FeatureBatch`; `RunContext`; `RuntimeBundle` (plain dataclass wrapping `iterator: IterDataset, model: eqx.Module` — model crosses jit via `InferencePlan.encode`/`.decode` `@filter_jit`); `FeatureBatch` (named return type — a concrete named type, NOT `dict[str, Array]` or `Any`; exact fields defined at R6-1 implementation)

**InputResolver contract (must not deviate):**
- Signature: `(spec: RunSpec, bundle: RuntimeBundle) -> FeatureBatch` — NO generic TypeVar parametrization over Spec+Bundle
- Subclass-specific resolvers dispatch via `@singledispatch` on `spec` type, not Generic Protocol subclassing
- Ruling out explicitly: `InputResolver[Spec, Bundle]` — generic Protocols with TypeVars in contravariant position require covariance, which breaks structural subtyping for resolver callables
- `run/protocols.py` — `AggregationFn`, `DecodeFn`, `NoiseFn`, `FeatureBatch` (named return type of InputResolver)
- aminx `RunSpec` (eqx.Module at `aminx/run/spec.py:114`) extends `xtrax.run.RunSpec` — eqx.Module inheritance is valid; aminx's RunSpec adds aminx-specific sub-configs (IOConfig, ResourceConfig, MultistateConfig, LigandConfig, etc.) on top of the xtrax base (seed, axes, carry_specs, boundaries)

**RuntimeBundle materializer (distinct from InputResolver):** `Protocol` with `__call__(self, spec: RunSpec) -> RuntimeBundle` — loads model + dataset from the spec and returns the materialized bundle. This is the PRIOR stage to InputResolver: materializer produces the bundle, then InputResolver uses `(spec, bundle) -> FeatureBatch`. Not yet named in the xtrax.run module surface; naming TBD at R6-1 implementation.

**AxisSpec naming gate (conditional sequencing rule):** If R7-1 field mapping table is complete before RS-6 RunSpec field names are committed, use `*_axis: AxisSpec` names directly in RS-6 — no `*_batch_size` interim names. If R7-1 is not complete at that point, use `*_batch_size` as temporary placeholder names AND treat the RS-7 rename as a declared breaking-change sprint boundary (not an incremental patch). Interim `*_batch_size` names are actively bad when multiple downstream projects build against xtrax.run immediately — they guarantee a cross-project rename in RS-7. Goal: ship with final names if at all possible. Do NOT use interim names as a default.

### R6-2: Logical grouping on RunSpecification
- `FeatureNoiseBundle` list replaces 8 noise fields
- `ModelConfig`, `ResourceConfig` sub-configs
- `SinkConfig` per-subclass (bridge from `output_h5_path` to `SinkSpec`)
- `output_h5_path` becomes deprecated alias for one sprint

### R6-3: Fn migration on subclasses
- `average_encoding_mode: Literal` → `encoding_aggregation_fn: AggregationFn`
- `decode_fn: Any` → `decode_fn: DecodeFn | None`
- `AveragingMode` enum with `to_fn()` shim
- `build_run_spec` bridge → `RunSpec.from_spec(spec)` classmethod with typed reads

### R6-4: FuseFn/Fuse unification
Resolve `FuseFn` (stages/protocols.py) vs `Fuse` (stages/boundaries.py). Deprecate one, re-export as the other.

---

## Sprint RS-7 — AxisSpec Convergence

### R7-1: Three-column mapping table (GATE — must precede R7-2)
Produce `.praxia/docs/research/260616_axisspec-field-map.md`: exact field comparison across xtrax/prolix/aminx. Include `bucket_boundaries` (xtrax only).

**Additional R7-1 deliverable:** Resolve the `tile_granularity` vs `granularity` naming conflict. xtrax already has a `granularity` field; prolix uses `tile_granularity`. The table must include a naming decision column and the chosen name must be committed as part of R7-1. Do not defer to R7-2 — if R7-2 implements the wrong name, AxisSpec's public API needs a second touch.

### R7-2: Extend xtrax AxisSpec
Add: `axis_index: int | None = None`, `tile_granularity: int = 1` (unify with existing `granularity` field — document distinction or merge).

### R7-3: Behavioral parity tests
Prolix `default_batch_size=0` → Vmap. xtrax `cardinality <= batch_size` → Vmap. Tests must verify identical decisions for equivalent inputs.

### R7-4: Migration sequence
1. prolix migrates off `prolix.tiling.AxisSpec` → `xtrax.tiling.AxisSpec`
2. aminx migrates off local planner → xtrax planner
3. Preserve `_BatchPlanWrapper` behavior (keeps `BatchPlan` out of JIT signature)

### R7-5: `*_batch_size` → `*_axis: AxisSpec`
`samples_batch_size`, `noise_batch_size`, `temperature_batch_size` → named `AxisSpec` fields. Runners use `BatchPlanner().plan([...])`.

---

## Acceptance Criteria

| Item | Gate |
|------|------|
| H1 | pytest passes; round-trip test for ScoringSpec with fixed_mask; test asserts `sidechain_conditioning` and `fixed_mask` are on `RunSpecification` base (not only SamplingSpec) |
| H2 | fixed_mask=None → all-ones; fixed_mask set → complement, correct shape; `assert chain_mask.dtype == jnp.float32` passes |
| H3 | build_run_spec called exactly once per SamplingSpecification construction (guard flag or idempotency — both valid; RunSpecification is plain @dataclass, not eqx.Module) |
| H4 | `from xtrax.stages import Fuse, Tap, Sink, AxisBoundary` works; isinstance checks pass; `from aminx.types.boundaries import Fuse` works without DeprecationWarning; `from xtrax.stages.protocols import FuseFn` raises DeprecationWarning; aminx tests pass unchanged |
| R6-1 | `from xtrax.run import RunSpec, SinkSpec, InputResolver, FeatureBatch`; InputResolver signature is `(spec: RunSpec, bundle: RuntimeBundle) -> FeatureBatch`; aminx tests pass |
| R6-2 | Noise field mapping table (`.praxia/docs/research/260616_noise-field-map.md`) committed before any R6-2 code |
| R7-1 | Field-map table committed (with naming decision for `tile_granularity` vs `granularity`) before any R7-2 work begins |
| R7-3 | Parity tests pass before R7-4 migration begins |
| R7-5 | aminx/prolix both import from xtrax.tiling.plan; no local AxisSpec copies |
