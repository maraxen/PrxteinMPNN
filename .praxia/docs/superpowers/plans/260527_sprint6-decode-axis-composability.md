# Sprint 6: Composable Decode Axis-Iteration Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans`. Steps use checkbox (`- [x]`) syntax.
> **CRITICAL CONSTRAINT:** Sprint 6 must be developed as a **reference implementation for a future stand-alone `composable_jax` library**. The library-side surface must be domain-neutral; the app-side mode classes carry MPNN-specific kernels. `test_library_surface.py` enforces the import-graph boundary.

> **Status: COMPLETE ✅** — All 14 tasks implemented. Commits on `refactor-full`. Verified 2026-06-02.

**Task ID:** `260527_s6-decode-axis-composability`
**Branch:** `refactor-full` | **Builds on:** Sprint 5 (`ee871f6d`) | **Version:** v3 (post oracle round 1)

---

## Glossary (read first)

Two distinct concepts that sound similar:

- **`decode_step`** — A `ConditionalDecodeStep | UnconditionalDecodeStep | None` `eqx.Module` field on `StageSet` (existing, Sprint 4). Wraps the *inner kernel* call. NOT touched by Sprint 6 (already on StageSet, stays on StageSet).
- **`decode_fn`** — A resolved mode-class instance (`ConditionalDecode`, `UnconditionalDecode`, `AutoregressiveDecode`, `STEDecode`) — *the dispatch result of Sprint 6*. Lives on `InferencePlan`, NOT on `StageSet`.

Two distinct concepts with similar shape:

- **`CarrySpec`** (Sprint 5, in `tiling/carry.py`) — fields `axis_name`, `init`, `transition`, `ordered_sinks`. Consumed by `BatchPlanner` Phase 0 to pre-demote axes to Scan. **Has actual carry value (`init`), not metadata.**
- **`CarryShape`** (NEW Sprint 6, in `tiling/carry_shape.py`) — metadata-only struct: `name`, `shape: tuple[int, ...]`, `dtype: jnp.dtype`. Used by mode classes that need to declare wave-axis carry shape **without** binding to a traced JAX value at class construction time. The init-value is materialized inside `__call__` from the metadata.

---

## Goal

Extend Sprint 5's `AxisStrategy` + variant-class composability pattern to the **four decode paths** (conditional, unconditional, autoregressive, STE) so that:

1. Each decode mode is a **single typed `eqx.Module`** (`ConditionalDecode`, `UnconditionalDecode`, `AutoregressiveDecode`, `STEDecode`) — no `if/elif` branching inside `__call__`.
2. Each mode owns one or more **injected iterators** (`MapIterator` for stateless axes, `ScanIterator` for carry axes). The state-axis (S) strategy is bound at factory time as a `state_iterator` field — linear, not multiplicative.
3. The AR wave-axis carry is reified as a **`CarryShape` + `ScanIterator` field pair** on `AutoregressiveDecode`. The mode class materializes the actual init-array inside `__call__` from the metadata.
4. The AR scatter-scan (`driver.py:437-443`) stays post-hoc inside `AutoregressiveDecode.__call__`, AFTER the wave iterator returns. The wave iterator handles only the sequence-carrying main scan.
5. The STE path reproduces its **tied-group einsum averaging** (currently `optimize_ste.py:197-207`, inside `update_step`) explicitly inside `STEDecode.__call__` — this is STE-specific behavior, not a general stage_set slot.
6. STE consumes the user's `stage_set`, projected to `logit_transform` only for the inner score call (the tied-group averaging is handled by STEDecode itself, not the projection).
7. New `decoder_sink: tuple[DecoderSinkFn, ...]` slot on `StageSet` mirrors `encoder_sink`. (Pre/post-process slots deferred — no concrete use case yet.)
8. `decode_fn` is resolved onto `InferencePlan`, NOT onto `StageSet`.
9. The planner validator (S5-C7) extends with **two** new rules (down from v2's three — Rule 3 dropped per oracle round 1):
   - `Scan` on state axis → `DispatchRejected` (from library), with mode-name context wrapped at app-layer
   - `STEMode` paired with `StageSet.decode_step` being a `UnconditionalDecodeStep` → `PlanTopologyError`

End state: **4 decode mode classes** + **3 iterator classes** (Vmap/SafeMap/Scan, library-side) + **1 STE wrapper** dispatched by `make_decode_fn(model, mode, strategy)`. Factory is split: domain-neutral `tiling/dispatch.py` (rejection logic, type signature) + app-side `inference/decode/factory.py` (concrete (mode, strategy) → mode-class table).

---

## Architecture

```
tiling/                            ← LIBRARY-side (extractable into composable_jax)
├── strategy.py                    ← (existing) Vmap | SafeMap | Scan
├── carry.py                       ← (existing) CarrySpec — for planner Phase 0
├── carry_shape.py                 ← NEW  CarryShape metadata (name, shape, dtype)
├── iterator.py                    ← NEW  MapIterator / ScanIterator Protocols
│                                         + VmapIterator / SafeMapIterator / JaxScanIterator
└── dispatch.py                    ← NEW  make_axis_dispatch(strategy, axis) -> Iterator
                                          + DispatchRejected (extends PlanTopologyError)

types/                             ← LIBRARY-side (existing)
├── stages.py                      ← (existing) StageSet — ADD decoder_sink ONLY
└── boundaries.py                  ← (existing)

inference/decode/                  ← APP-side (MPNN-specific)
├── __init__.py                    ← public re-exports
├── mode.py                        ← ConditionalMode | UnconditionalMode |
│                                    AutoregressiveMode | STEMode (sealed union)
├── protocols.py                   ← DecodeScoreFn | ARDecodeFn | STEDecodeFn (3 protocols, not 1)
│                                  + DecoderSinkFn
├── _kernel.py                     ← pure helpers: _decode_one_step, _project_logits,
│                                    _apply_logit_transform, _tied_group_einsum_average
├── _base.py                       ← _ConditionalDecodeBase (eqx.Module, ABC)
├── conditional.py                 ← ConditionalDecode(_ConditionalDecodeBase)
├── unconditional.py               ← UnconditionalDecode(eqx.Module)
├── autoregressive.py              ← AutoregressiveDecode(eqx.Module) — owns
│                                    state_iterator, wave_iterator, wave_carry: CarryShape
├── ste.py                         ← STEDecode(eqx.Module) — owns inner: _ConditionalDecodeBase,
│                                    iterations: int (reproduces tied-group einsum internally)
└── factory.py                     ← make_decode_fn(model, mode, strategy)
                                       calls tiling.dispatch.make_axis_dispatch + wraps mode context

host/
└── plan.py                        ← InferencePlan.decode_fn (resolved here)
                                     + _validate_plan_topology extended (two new rules)

inference/driver.py                ← shrinks to ≤4K; legacy _decode_* deleted;
                                     decode() and infer_topology() remain as thin routers
```

### Sealed unions (in `inference/decode/mode.py`, APP-side)

```python
@dataclass(frozen=True)
class ConditionalMode: ...
@dataclass(frozen=True)
class UnconditionalMode: ...
@dataclass(frozen=True)
class AutoregressiveMode:
    """Wave axis is always Scan internally — a structural invariant, not a knob.
    
    There is no W-axis BatchPlanner decision; the user does not pass a wave_iterator.
    """
@dataclass(frozen=True)
class STEMode:
    inner_mode: ConditionalMode = ConditionalMode()
    iterations: int = 100

DecodeMode = ConditionalMode | UnconditionalMode | AutoregressiveMode | STEMode
```

### Iterator protocols (in `tiling/iterator.py`, LIBRARY-side)

```python
@runtime_checkable
class MapIterator(Protocol):
    """Stateless axis iteration: (fn, xs) -> ys. No carry."""
    def __call__(self, fn: Callable, xs: Any, *, in_axes: Any = 0) -> Any: ...

@runtime_checkable
class ScanIterator(Protocol):
    """Carry-bearing axis iteration: (fn, init, xs) -> (final_carry, ys)."""
    def __call__(self, fn: Callable, init: Any, xs: Any) -> tuple[Any, Any]: ...

class VmapIterator(eqx.Module): ...           # MapIterator via jax.vmap (no fields)
class SafeMapIterator(eqx.Module):            # MapIterator via safe_map
    tile: int = eqx.field(static=True)
class JaxScanIterator(eqx.Module): ...        # ScanIterator via jax.lax.scan (no fields)
```

**Treedef invariant** (oracle REC-1): `tree_structure(ConditionalDecode(state_iterator=VmapIterator()))` ≠ `tree_structure(ConditionalDecode(state_iterator=SafeMapIterator(tile=4)))`. This is the **intended** semantics — switching strategy must trigger re-JIT. Tests in Task 1 assert this.

### Decode protocols (in `inference/decode/protocols.py`, APP-side)

> **Naming note (oracle round 2):** `ScoreFn` is already a top-level type alias used by `InferencePlan.score` (Sprint 2 invariant, CLAUDE.md). The decode-specific Conditional/Unconditional scoring protocol is therefore named `DecodeScoreFn` to avoid shadowing.

```python
DecodeScoreFn = Callable[..., Logits]          # ConditionalDecode + UnconditionalDecode
ARDecodeFn    = Callable[..., SampleResult]    # AutoregressiveDecode (carries wave)
STEDecodeFn   = Callable[..., tuple[Any, ...]] # STEDecode (returns (sequence, logits_a, logits_b))
DecoderSinkFn = Callable[..., None]             # io_callback hook, mirrors EncoderSinkFn
```

`InferencePlan.decode_fn: DecodeScoreFn | ARDecodeFn | STEDecodeFn`.

### Forbidden cells (planner-enforced)

|                    | Vmap (state) | SafeMap (state) | Scan (state) |
|--------------------|:------------:|:---------------:|:------------:|
| ConditionalMode    |      YES     |       YES       |      NO      |
| UnconditionalMode  |      YES     |       YES       |      NO      |
| AutoregressiveMode |      YES†    |      YES†       |      NO      |
| STEMode            |      YES     |       YES       |      NO      |

†For `AutoregressiveMode`, the listed strategy applies to the inner state-axis (S). The wave-axis (W) is internal and always `JaxScanIterator`. There is **no user-facing W-strategy knob** (Risk D-3 v3).

Scan-on-state is rejected because state geometries are heterogeneous (Sprint 5 invariant in `CarrySpec.__post_init__`).

### Composable-JAX patterns applied

- **Pattern 1** — 4 mode classes (variant per call-signature), not 8 (variant per axis-strategy).
- **Pattern 4** — `MapIterator`/`ScanIterator` are protocol wrappers around `safe_map`/`jax.lax.scan`/`jax.vmap`.
- **Pattern 5** — Concrete iterators injected at factory time. Protocols are static-only (no `@runtime_checkable` on the concrete eqx.Modules).
- **Pattern 6** — `make_decode_fn` runs once at `make_inference_plan` time; the resolved mode class is stored on `InferencePlan.decode_fn`.

---

## Tech Stack

- Python 3.12+, JAX, Equinox (`eqx.Module`, `eqx.field(static=True)`, ABC mix-in), pytest, `uv run pytest`
- Reuses Sprint 5: `AxisStrategy`, `safe_map`, `safe_scan`, `CarrySpec`, `AxisBoundary`, `PlanTopologyError`

---

## Oracle Risk Summary (updated post round 1)

- **Risk D-1 (kernel divergence):** All shared math lives in `_kernel.py` pure functions; mode classes own iterator orchestration only.
- **Risk D-2 (STE tied-group einsum):** Currently in `optimize_ste.py:197-207` (post-update, inside `update_step`); NOT in `score_conditional`. **Mitigation:** `STEDecode.__call__` reproduces this einsum block explicitly, using `_tied_group_einsum_average` helper in `_kernel.py`. Fixture coverage at Task 10 includes tied positions with S>1 and num_groups<L to catch silent regressions.
- **Risk D-3 (AR wave-axis internal):** Wave-axis Scan is a structural invariant on `AutoregressiveDecode`; no user-facing knob exists. The validator does NOT need a "reject external W-strategy" rule — it's unreachable by API design. Documented in `AutoregressiveMode` docstring.
- **Risk D-4 (variant explosion):** 4 mode classes × injected iterator field. K/W axis extensions add iterator fields, not class multiplication.
- **Risk D-5 (library extraction):** Library surface narrowed: only `tiling/*.py` + `types/stages.py` (existing) + `types/boundaries.py` (existing). All decode-domain types (`mode.py`, `protocols.py`, `_kernel.py`, mode classes, factory) are app-side.
- **Risk D-6 (back-compat / driver.py removal):** Wave E sequencing — retire after callers migrated. Broader grep audit per oracle REC-6.
- **Risk D-7 (STE↔Conditional peer-file coupling):** `_ConditionalDecodeBase(eqx.Module, ABC)` mediates; both `ConditionalDecode` and `STEDecode.inner` consume the base.
- **Risk D-8 (`decode_fn` placement):** On `InferencePlan`, NOT `StageSet`. Glossary above clarifies vs `decode_step`.
- **Risk D-9 (deferred hooks):** Pre/post-process hooks excluded; only `decoder_sink` added.
- **Risk D-10 (`CarrySpec` API mismatch — NEW v3):** Sprint 5's `CarrySpec` has fields `axis_name`, `init`, `transition`, `ordered_sinks` — not the (`name`, `shape`, `dtype`, `init_value()`) API v2 assumed. **Mitigation:** introduce new metadata-only `CarryShape(name, shape, dtype)` struct in `tiling/carry_shape.py` (Task 3); `AutoregressiveDecode.wave_carry: CarryShape` holds metadata, materializes the actual init-array inside `__call__`. `CarrySpec` remains untouched for planner Phase-0 consumption.
- **Risk D-11 (AR scatter scan — NEW v3):** `decode_ar` has TWO scans (main wave scan at `driver.py:427` + scatter scan at `driver.py:443`). **Mitigation:** Scatter scan stays post-hoc inside `AutoregressiveDecode.__call__` (after `wave_iterator`); the iterator only carries the sequence. `wave_carry: CarryShape(name="sequence", shape=(L,), dtype=int32)`.
- **Risk D-12 (validator Rule 1 circular — NEW v3):** Oracle CONCERN-5: Task 11 Rule 1 claimed `make_axis_dispatch` enforces mode-specific messages, but `make_axis_dispatch` is library-neutral. **Mitigation:** `make_decode_fn` (Task 11) wraps `DispatchRejected` with a re-raise that adds `f"{type(mode).__name__}"` context. Validator test calls `make_decode_fn` end-to-end.
- **Risk D-13 (deprecation warning vs hard-cut — NEW v3, oracle REC-5):** STE legacy path on `refactor-full` branch is unreleased. **Mitigation:** hard-require `stage_set` parameter; no `DeprecationWarning`; grep-audit + migrate all callers in same PR.

---

## File Map

### New files

| File | Responsibility | Library-side? |
|---|---|:--:|
| `src/prxteinmpnn/tiling/iterator.py` | `MapIterator` / `ScanIterator` protocols + 3 concrete iterators | YES |
| `src/prxteinmpnn/tiling/dispatch.py` | `make_axis_dispatch(strategy, axis)`; `DispatchRejected` | YES |
| `src/prxteinmpnn/tiling/carry_shape.py` | `CarryShape(name, shape, dtype)` metadata struct | YES |
| `src/prxteinmpnn/inference/decode/__init__.py` | Public re-exports | NO |
| `src/prxteinmpnn/inference/decode/mode.py` | `DecodeMode` sealed union | NO |
| `src/prxteinmpnn/inference/decode/protocols.py` | `DecodeScoreFn`, `ARDecodeFn`, `STEDecodeFn`, `DecoderSinkFn` | NO |
| `src/prxteinmpnn/inference/decode/_kernel.py` | Pure helpers (including `_tied_group_einsum_average`) | NO |
| `src/prxteinmpnn/inference/decode/_base.py` | `_ConditionalDecodeBase(eqx.Module, ABC)` | NO |
| `src/prxteinmpnn/inference/decode/conditional.py` | `ConditionalDecode(_ConditionalDecodeBase)` | NO |
| `src/prxteinmpnn/inference/decode/unconditional.py` | `UnconditionalDecode(eqx.Module)` | NO |
| `src/prxteinmpnn/inference/decode/autoregressive.py` | `AutoregressiveDecode(eqx.Module)` | NO |
| `src/prxteinmpnn/inference/decode/ste.py` | `STEDecode(eqx.Module)` | NO |
| `src/prxteinmpnn/inference/decode/factory.py` | `make_decode_fn(model, mode, strategy)` + mode-context wrap | NO |

### Modified files

| File | What changes |
|---|---|
| `src/prxteinmpnn/types/stages.py` | Add `decoder_sink: tuple[DecoderSinkFn, ...] = ()` as `eqx.field(static=True)`. **`decode_step` stays — already present; not modified.** **`decode_fn` does NOT live here.** |
| `src/prxteinmpnn/host/plan.py` | `InferencePlan` gains `decode_fn: DecodeScoreFn \| ARDecodeFn \| STEDecodeFn`. `make_inference_plan` calls `make_decode_fn` once. `_validate_plan_topology` adds two new rules (mode-name-wrapped DispatchRejected; STE+Unconditional decode_step rejection). |
| `src/prxteinmpnn/inference/driver.py` | Shrinks from 18.0K → ≤4K; `_decode_conditional`, `_decode_unconditional`, `decode_ar` deleted; `decode()` and `infer_topology()` become ≤10-LOC routers via `plan.decode_fn`. |
| `src/prxteinmpnn/inference/optimize_ste.py` | `make_optimize_sequence_fn` **requires** `stage_set: StageSet` (no `None` default; oracle REC-5). Constructs `STEDecode` from the input stage_set + projection. |
| `src/prxteinmpnn/host/kernel_dispatch.py` | Strategy resolution routed through `tiling/dispatch.py` `make_axis_dispatch`. |

### Test files

| File | What it covers |
|---|---|
| `tests/tiling/test_iterator.py` | Protocol conformance; numerical equivalence; **treedef-invariant tests** (REC-1) |
| `tests/tiling/test_dispatch.py` | Happy path + `DispatchRejected` on (state, Scan) |
| `tests/tiling/test_carry_shape.py` | `CarryShape` metadata-only behavior; `materialize(L)` returns a JAX array of declared shape/dtype |
| `tests/inference/decode/test_mode.py` | `DecodeMode` sealed union: frozen, equality, defaults, isinstance |
| `tests/types/test_stages_decoder_sink.py` | `decoder_sink` slot defaults to `()`, is static, preserves treedef invariant |
| `tests/inference/decode/test_kernel.py` | `_decode_one_step`, `_project_logits`, `_tied_group_einsum_average` parity with current `driver.py` + `optimize_ste.py` outputs |
| `tests/inference/decode/test_conditional.py` | `ConditionalDecode` parity with `_decode_conditional`; Vmap and SafeMap iterators |
| `tests/inference/decode/test_unconditional.py` | Same for unconditional |
| `tests/inference/decode/test_autoregressive.py` | `AutoregressiveDecode` parity with `decode_ar`; wave-scan invariant; **scatter-scan post-hoc** (Risk D-11); CarryShape round-trip |
| `tests/inference/decode/test_ste.py` | `STEDecode` parity with current `optimize_sequence_fn`; **fixture includes tied positions S>1 num_groups<L** (Risk D-2); stage_set projection |
| `tests/inference/decode/test_factory.py` | `make_decode_fn` table; forbidden pairs → `DispatchRejected` with mode-name context |
| `tests/host/test_plan_topology_decode.py` | Validator rejects: (mode, Scan) with mode-name in message; STE + UnconditionalDecodeStep |
| `tests/inference/decode/test_driver_shim.py` | Post-deletion `driver.decode()` routes through `plan.decode_fn` |
| `tests/inference/decode/test_library_surface.py` | Import-graph lint (AST walk, **includes TYPE_CHECKING blocks** per REC-3, with explicit negative test) |

---

## Wave A — Foundation types (Tasks 1–6, parallel-safe)

### Task 1: `MapIterator` / `ScanIterator` + concrete iterators + treedef tests

**Files:** Create `tiling/iterator.py`, `tests/tiling/test_iterator.py`

- [x] **Step 1.1: Write tests:**
  - Protocol conformance: each concrete iterator satisfies its corresponding `runtime_checkable` Protocol via `isinstance`.
  - Numerical equivalence: `VmapIterator()(lambda x: x*2, jnp.arange(4)) == [0,2,4,6]`; same for `SafeMapIterator(tile=2)`; `JaxScanIterator()(lambda c, x: (c+x, c+x), 0, jnp.arange(4))` returns `(6, jnp.array([0,1,3,6]))`.
  - **Treedef invariant (REC-1):** `tree_structure(SomeWrapper(iter=VmapIterator())) != tree_structure(SomeWrapper(iter=SafeMapIterator(tile=4)))`. Document this is intended (re-JIT on strategy switch).
- [x] **Step 1.2: Implement** — `VmapIterator()` (no fields) wraps `jax.vmap`; `SafeMapIterator(tile: int = eqx.field(static=True))` wraps `safe_map`; `JaxScanIterator()` wraps `jax.lax.scan`.
- [x] **Step 1.3: Run tests — pass.**
- [x] **Step 1.4: Commit:** `feat(S6-A1): add MapIterator/ScanIterator + Vmap/SafeMap/JaxScan iterators with treedef tests`

### Task 2: `make_axis_dispatch` (library-side factory contract)

**Files:** Create `tiling/dispatch.py`, `tests/tiling/test_dispatch.py`

- [x] **Step 2.1: Write tests:**
  - Happy path: `make_axis_dispatch(Vmap())` → `VmapIterator()`; `make_axis_dispatch(SafeMap(tile=4))` → `SafeMapIterator(tile=4)`.
  - Reject path: `make_axis_dispatch(Scan(...), axis="state")` raises `DispatchRejected` with message containing "state axis" and "heterogeneous".
- [x] **Step 2.2: Implement** — `DispatchRejected(PlanTopologyError)`; `make_axis_dispatch(strategy, *, axis: str = "state")` dispatches by `isinstance` on `strategy`. Raises `DispatchRejected` for `Scan` on heterogeneous axes (state is the canonical heterogeneous axis).
- [x] **Step 2.3: Commit:** `feat(S6-A2): add make_axis_dispatch library-side factory contract`

### Task 3: `CarryShape` metadata struct

**Files:** Create `tiling/carry_shape.py`, `tests/tiling/test_carry_shape.py`

- [x] **Step 3.1: Write tests:**
  - Frozen dataclass; equality by value.
  - `CarryShape(name="sequence", shape=(L,), dtype=jnp.int32).materialize()` returns `jnp.zeros((L,), dtype=jnp.int32)`.
  - Distinct from `CarrySpec`: assert `tiling.carry.CarrySpec` and `tiling.carry_shape.CarryShape` are different types.
- [x] **Step 3.2: Implement:**
  ```python
  @dataclass(frozen=True)
  class CarryShape:
      name: str
      shape: tuple[int, ...]
      dtype: Any  # jnp.dtype
      def materialize(self) -> jax.Array:
          return jnp.zeros(self.shape, dtype=self.dtype)
  ```
- [x] **Step 3.3: Commit:** `feat(S6-A3): add CarryShape metadata struct (Risk D-10 mitigation)`

### Task 4: `DecodeMode` sealed union + 3 decode protocols

**Files:** Create `inference/decode/mode.py`, `inference/decode/protocols.py`, `tests/inference/decode/test_mode.py`

- [x] **Step 4.1: Write tests** — 6 tests:
  - Each mode is a frozen dataclass.
  - Equality by value.
  - `AutoregressiveMode()` has no W-axis fields (oracle CONCERN-6).
  - `STEMode().inner_mode == ConditionalMode()`.
  - `isinstance(x, DecodeMode)` for all four.
  - 3 protocols (`DecodeScoreFn`, `ARDecodeFn`, `STEDecodeFn`) are `Callable` aliases distinguishable by return-type annotations (mypy/pyright check, not runtime).
- [x] **Step 4.2: Implement** `mode.py` — sealed union of 4 modes as frozen dataclasses.
- [x] **Step 4.3: Implement** `protocols.py` — 3 Callable type aliases + `DecoderSinkFn`. NO `@runtime_checkable` (Pattern 5 skill caution).
- [x] **Step 4.4: Commit:** `feat(S6-A4): add DecodeMode union + 3 decode protocols (app-side)`

### Task 5: `_ConditionalDecodeBase` ABC + shared kernel helpers

**Files:** Create `inference/decode/_base.py`, `inference/decode/_kernel.py`, `tests/inference/decode/test_kernel.py`

- [x] **Step 5.1: Write golden-snapshot tests:**
  - `_decode_one_step(model, node_f, edge_f, nei, mask, ar_mask, sequence_oh) -> (L, H)` matches current `driver.py:_decode_conditional` inner body to machine precision on a fixture.
  - `_project_logits(model, decoded) -> (S, L, V)` matches.
  - **NEW (Risk D-2):** `_tied_group_einsum_average(logits, tie_group_map, num_groups) -> averaged_logits` matches the einsum block in `optimize_ste.py:197-207` on a fixture with tied positions.
- [x] **Step 5.2: Extract** the bodies into pure functions in `_kernel.py`.
- [x] **Step 5.3: Implement `_ConditionalDecodeBase(eqx.Module, ABC)`** — abstract method `__call__(self, key, enc, bundle, config, stage_set) -> Logits`. Provides shared `_apply_logit_transform` helper.
- [x] **Step 5.4: Commit:** `feat(S6-A5): add _ConditionalDecodeBase ABC + pure kernel helpers (incl. tied-group einsum)`

### Task 6: `StageSet.decoder_sink` slot

**Files:** Modify `types/stages.py`, create `tests/types/test_stages_decoder_sink.py`

- [x] **Step 6.1: Write test** — assert `StageSet.decoder_sink` defaults to `()`, is `eqx.field(static=True)`, and treedef leaves count is unchanged from pre-Sprint-6 (S5 Risk 1 invariant).
- [x] **Step 6.2: Add slot.** Update `make_stage_set` signature accordingly.
- [x] **Step 6.3: Commit:** `feat(S6-A6): add decoder_sink slot to StageSet`

---

## Wave B — Non-AR mode classes (Tasks 7–8, after Wave A)

### Task 7: `ConditionalDecode`

**Files:** Create `inference/decode/conditional.py`, `tests/inference/decode/test_conditional.py`

- [x] **Step 7.1: Write parity tests** — `ConditionalDecode(model=m, state_iterator=VmapIterator())` matches `driver.py:_decode_conditional(m, ...)` on fixtures: **S=1, S=4, S=8** (oracle round-1 fixture mandate). Same with `state_iterator=SafeMapIterator(tile=2)`. Tolerance ≤1e-6.
- [x] **Step 7.2: Implement `ConditionalDecode(_ConditionalDecodeBase)`** — fields `model: Any = eqx.field(static=True)`, `state_iterator: MapIterator`. `__call__(self, key, enc, bundle, config, stage_set)` calls `self.state_iterator(per_state_fn, encoder_outputs)`. NO branching on `state_iterator` type.
- [x] **Step 7.3: Run parity tests.**
- [x] **Step 7.4: Commit:** `feat(S6-B7): add ConditionalDecode mode class`

### Task 8: `UnconditionalDecode`

**Files:** Create `inference/decode/unconditional.py`, `tests/inference/decode/test_unconditional.py`

- [x] **Step 8.1: Write parity tests** — same fixture coverage as Task 7.
- [x] **Step 8.2: Implement** — same shape as Task 7; `__call__` does not consume `sequence_oh`.
- [x] **Step 8.3: Commit:** `feat(S6-B8): add UnconditionalDecode mode class`

---

## Wave C — AutoregressiveDecode (Task 9, after Wave B)

### Task 9: `AutoregressiveDecode` with `CarryShape` and post-hoc scatter

**Files:** Create `inference/decode/autoregressive.py`, `tests/inference/decode/test_autoregressive.py`

The AR case has TWO scans in current code: the wave-scan with sequence carry (`driver.py:427`) + the post-hoc scatter-scan that maps per-wave logits back to per-position (`driver.py:443`). Sprint 6 fuses **only the wave-scan into the iterator**; the scatter stays post-hoc inside `__call__` (Risk D-11).

- [x] **Step 9.1: Write parity tests:**
  - `AutoregressiveDecode(model=m, state_iterator=VmapIterator(), wave_iterator=JaxScanIterator(), wave_carry=CarryShape("sequence", (L,), jnp.int32), decoding_order_fn=...)` matches `driver.py:decode_ar(m, ...)` on fixtures (S=1, S=4) with tied positions and untied positions. Tolerance ≤1e-6.
  - S-axis parity: Vmap vs SafeMap inside scan body matches.
  - CarryShape round-trip: `wave_carry.materialize().shape == (L,)`, `.dtype == jnp.int32`.
- [x] **Step 9.2: Implement `AutoregressiveDecode(eqx.Module)`** — fields:
  - `model: Any = eqx.field(static=True)`
  - `decoding_order_fn: DecodingOrderFn = eqx.field(static=True)`
  - `state_iterator: MapIterator`
  - `wave_iterator: ScanIterator` (always `JaxScanIterator()` from factory; field exists for type-symmetry)
  - `wave_carry: CarryShape = eqx.field(static=True)`
  
  `__call__`:
  1. Materialize `init = self.wave_carry.materialize()` (shape (L,))
  2. Construct `scan_body(sequence, wave_idx) -> (new_seq, per_wave_logits)` using `self.state_iterator(per_state_decode, encoder_outputs)` for the per-step S iteration.
  3. Call `final_sequence, logits_stack = self.wave_iterator(scan_body, init, jnp.arange(n_waves))`.
  4. **Post-hoc scatter** (NOT in iterator): `logits = jax.lax.scan(scatter_logits, jnp.zeros((L, V)), jnp.arange(n_waves))[0]` — preserves current `driver.py:443` behavior.
  5. Return `SampleResult(final_sequence, logits, ...)`.
- [x] **Step 9.3: Tie-group integration** — `stage_set.ar_logit_transform` and `stage_set.tie_group_fuse` called from kernel helpers (already vmapped over L); pass-through unchanged.
- [x] **Step 9.4: Run all parity tests.**
- [x] **Step 9.5: Commit:** `feat(S6-C9): add AutoregressiveDecode with CarryShape + post-hoc scatter`

---

## Wave D — STE + factory + validator (Tasks 10–12, after Wave C)

### Task 10: `STEDecode` composing `_ConditionalDecodeBase` (reproduces tied-group einsum)

**Files:** Create `inference/decode/ste.py`, `tests/inference/decode/test_ste.py`

- [x] **Step 10.1: Write parity tests** — `STEDecode(inner=ConditionalDecode(model=m, state_iterator=VmapIterator()), iterations=10, optimizer=optax.adam(1e-3))` matches current `make_optimize_sequence_fn(...)` on fixtures including:
  - Untied positions (S=4, L=20)
  - **Tied positions: `tie_group_map=[0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9]`, `num_groups=10`, S=4, L=20** (Risk D-2 fixture mandate from oracle CONCERN-2)
  - Single state (S=1, both tied and untied)
  - Tolerance ≤1e-5.
- [x] **Step 10.2: Write stage_set projection test** — when input `stage_set` has `ar_logit_transform` and `tie_group_fuse` set, the inner's `stage_set` (after `_project_stage_set_for_ste`) contains only `logit_transform`.
- [x] **Step 10.3: Implement `STEDecode(eqx.Module)`** — fields:
  - `inner: _ConditionalDecodeBase` (a `ConditionalDecode` instance bound to a `MapIterator`)
  - `iterations: int = eqx.field(static=True)`
  - `optimizer: Any = eqx.field(static=True)`
  
  `__call__`:
  1. Construct projected stage_set (`_project_stage_set_for_ste`) — only `logit_transform`.
  2. `loss_fn = make_loss_fn(self.inner, projected_stage_set, ...)` — pure closure.
  3. `value_and_grad = jax.value_and_grad(loss_fn)`.
  4. `update_step` body:
     a. Compute grads via `value_and_grad`.
     b. `optimizer.update(grads, opt_state)` → `optax.apply_updates`.
     c. **Reproduce tied-group einsum** (Risk D-2) via `_tied_group_einsum_average(next_logits, tie_group_map, num_groups)` from `_kernel.py`.
     d. Apply fixed positions.
  5. `jax.lax.fori_loop(0, self.iterations, update_step, init_state)`.
  6. Final STE pass; return `(sequence, logits_a, logits_b)`.
  - Optional `jax.checkpoint` on the loss_fn (gated by env var or config flag).
- [x] **Step 10.4: Run parity tests** including the tied-positions fixture.
- [x] **Step 10.5: Commit:** `feat(S6-D10): add STEDecode (reproduces tied-group einsum per Risk D-2)`

### Task 11: `make_decode_fn` factory (mode-context wrapping)

**Files:** Create `inference/decode/factory.py`, `tests/inference/decode/test_factory.py`

- [x] **Step 11.1: Write factory tests:**
  - Happy path: `(ConditionalMode(), Vmap())` → `ConditionalDecode(state_iterator=VmapIterator())` (assert iterator field type).
  - `(UnconditionalMode(), SafeMap(tile=4))` → `UnconditionalDecode(state_iterator=SafeMapIterator(tile=4))`.
  - `(AutoregressiveMode(), Vmap())` → `AutoregressiveDecode` with `wave_iterator: JaxScanIterator()`, `wave_carry: CarryShape("sequence", ...)`.
  - `(STEMode(inner_mode=ConditionalMode(), iterations=50), Vmap())` → `STEDecode(inner=ConditionalDecode(state_iterator=VmapIterator()), iterations=50)`.
  - **Reject path with mode-name context (Risk D-12):** `make_decode_fn(model, ConditionalMode(), Scan(...))` raises `DispatchRejected` with message matching `"ConditionalMode.*state axis.*heterogeneous"`.
- [x] **Step 11.2: Implement** — dispatch by `isinstance(mode, ...)`. Each branch calls `make_axis_dispatch(strategy, axis="state")` first and wraps any `DispatchRejected` to add `f"in mode {type(mode).__name__}"` context, then constructs the mode class.
- [x] **Step 11.3: Implement `_project_stage_set_for_ste(stage_set: StageSet) -> StageSet`** — returns a StageSet with only `logit_transform` populated; called by the `STEMode` branch.
- [x] **Step 11.4: Commit:** `feat(S6-D11): add make_decode_fn factory with mode-name-wrapped DispatchRejected`

### Task 12: Planner validator extension

**Files:** Modify `host/plan.py:_validate_plan_topology`, create `tests/host/test_plan_topology_decode.py`

The validator gets **two** new rules (Rule 3 from v2 dropped per oracle CONCERN-6 — there's no W-axis BatchPlanner knob to validate).

- [x] **Step 12.1: Write tests:**
  - **Rule 1** (via `make_decode_fn` end-to-end): `make_decode_fn(model, ConditionalMode(), Scan(init=jnp.zeros(()), transition=lambda c,x:(c,x)))` raises `DispatchRejected` with message containing both `"ConditionalMode"` AND `"heterogeneous"`.
  - **Rule 2** (validator-direct): a plan whose `stage_set.decode_step` is `UnconditionalDecodeStep(...)` and whose `decode_fn` is `STEDecode(...)` triggers `PlanTopologyError` at `_validate_plan_topology` call time.
- [x] **Step 12.2: Implement Rule 2** in `_validate_plan_topology`:
  ```python
  if isinstance(plan.decode_fn, STEDecode) and isinstance(plan.stage_set.decode_step, UnconditionalDecodeStep):
      raise PlanTopologyError("STEMode requires ConditionalDecodeStep; got UnconditionalDecodeStep")
  ```
  Rule 1 is enforced by `make_decode_fn` (factory level); validator tests confirm the message.
- [x] **Step 12.3: Commit:** `feat(S6-D12): extend plan topology validator (two new rules; W-axis rule dropped per oracle)`

---

## Wave E — Wire through plan + retire driver.py (Tasks 13–14, after Wave D)

### Task 13: `InferencePlan.decode_fn` + hard-cut STE wiring

**Files:** Modify `host/plan.py`, `inference/optimize_ste.py`, tests in `tests/host/`

- [x] **Step 13.1: Write integration test** — `plan.sample(...)` and `plan.score(...)` route through `plan.decode_fn`; instrument the mode class (or assert via PyTree leaf inspection) that it was called.
- [x] **Step 13.2: Update `InferencePlan`** — add `decode_fn: DecodeScoreFn | ARDecodeFn | STEDecodeFn`. `make_inference_plan` calls `make_decode_fn(model, mode, strategy)` once and stores the result.
- [x] **Step 13.3: Update `make_optimize_sequence_fn`** (oracle REC-5: hard-cut):
  - **Require** `stage_set: StageSet` (NO `None` default).
  - Construct `STEDecode` from the input stage_set via the factory.
  - No `DeprecationWarning`. Calling without `stage_set` raises `TypeError` (Python signature).
- [x] **Step 13.4: grep + migrate** all callers of `make_optimize_sequence_fn` in `src/`, `tests/`, and `scripts/` to pass `stage_set`.
- [x] **Step 13.5: Run full suite** — `uv run pytest -q --tb=short` must pass.
- [x] **Step 13.6: Commit:** `feat(S6-E13): InferencePlan.decode_fn; hard-cut STE to require stage_set`

### Task 14: Retire `driver.py` decode functions + library-surface lint

**Files:** Modify `inference/driver.py`, sweep, create `tests/inference/decode/test_library_surface.py`

- [x] **Step 14.1: Broad grep audit** (oracle REC-6):
  ```bash
  grep -rn "_decode_conditional\|_decode_unconditional\|^def decode_ar\|getattr.*decode\|setattr.*decode\|driver\.decode_ar\|import_module.*driver\|__import__.*driver" src/ tests/ scripts/
  ```
  Record every caller; migrate to `plan.score()` / `plan.sample()` / `plan.decode_fn(...)`.
- [x] **Step 14.2: Verify** `host/plan.py:driver=driver_module.decode` router is preserved (it's the thin shim that survives).
- [x] **Step 14.3: Delete** `_decode_conditional`, `_decode_unconditional`, `decode_ar` from `driver.py`. Keep `decode()` and `infer_topology()` as ≤10-LOC routers.
- [x] **Step 14.4: Write `test_library_surface.py`** (REC-3):
  - AST-walks every `.py` file under `tiling/`, `types/` (excluding `types/stages.py` which is a shared bridge).
  - Treats `if TYPE_CHECKING:` imports the same as runtime imports.
  - Asserts no `from prxteinmpnn.{inference,model,sampling,scoring,run,host,io}.* import ...` exists in any library-side file.
  - Explicit negative test: a deliberately-bad file (or string fixture) containing `from prxteinmpnn.inference.driver import decode_ar` is asserted to be **detected** by the lint.
  - Allowed: `jax`, `jax.numpy`, `equinox`, `jaxtyping`, `optax`, stdlib.
- [x] **Step 14.5: Run full suite — 0 failures required.**
- [x] **Step 14.6: Commit:** `refactor(S6-E14): retire driver.py decode functions; add library-surface lint with TYPE_CHECKING coverage`

---

## Self-Review Checklist

**Spec coverage:**
- [x] `MapIterator`/`ScanIterator` + 3 concrete iterators + treedef tests (Task 1; REC-1)
- [x] `make_axis_dispatch` library-side factory contract (Task 2)
- [x] `CarryShape` metadata struct (Task 3; CONCERN-1)
- [x] `DecodeMode` + 3 decode protocols (Task 4; REC-2, REC-4)
- [x] `_ConditionalDecodeBase` ABC + `_kernel.py` incl. `_tied_group_einsum_average` (Task 5; CONCERN-2)
- [x] `StageSet.decoder_sink` slot only (Task 6)
- [x] `ConditionalDecode` (Task 7)
- [x] `UnconditionalDecode` (Task 8)
- [x] `AutoregressiveDecode` with `CarryShape` + post-hoc scatter (Task 9; CONCERN-3)
- [x] `STEDecode` reproducing tied-group einsum (Task 10; CONCERN-2)
- [x] `make_decode_fn` factory with mode-context wrap (Task 11; CONCERN-5)
- [x] Planner validator: 2 rules (Rule 1 message-level via factory; Rule 2 STE+Unconditional) (Task 12; CONCERN-6 drop)
- [x] `InferencePlan.decode_fn` + hard-cut STE (Task 13; REC-5)
- [x] `driver.py` retired + library-surface lint (Task 14; REC-3, REC-6)

**Risks addressed:**
- [x] D-1: Shared `_kernel.py` prevents variant drift
- [x] D-2: `STEDecode` reproduces tied-group einsum + fixture coverage at Task 10
- [x] D-3: No user-facing W-axis knob — invariant only, not validator rule
- [x] D-4: 4 mode classes × injected iterator (linear)
- [x] D-5: Library surface limited to `tiling/` + existing types
- [x] D-6: Wave E sequencing + broader grep audit
- [x] D-7: `_ConditionalDecodeBase` mediates STE↔Conditional
- [x] D-8: `decode_fn` on `InferencePlan`, not `StageSet`
- [x] D-9: Pre/post-process hooks deferred
- [x] D-10: `CarryShape` introduced; `CarrySpec` unchanged
- [x] D-11: Scatter scan stays post-hoc in `AutoregressiveDecode.__call__`
- [x] D-12: `make_decode_fn` wraps `DispatchRejected` with mode-name context
- [x] D-13: Hard-cut STE, no DeprecationWarning

**Invariants preserved (CLAUDE.md):**
- [x] `InferenceBundle` and sub-bundles: not touched
- [x] `SamplerFn`/`ScoreFn` top-level signatures: not touched
- [x] Kernel math: relocated only, not rewritten
- [x] `make_stage_set` single construction site
- [x] Sprint 5 primitives consumed unchanged (`CarrySpec` API NOT modified — new `CarryShape` is additive)

**Wave sequencing:**
- Wave A (Tasks 1–6): parallel-safe foundation
- Wave B (Tasks 7–8): requires Wave A
- Wave C (Task 9): requires Wave B
- Wave D (Tasks 10–12): requires Wave C
- Wave E (Tasks 13–14): requires Wave D; Task 14 is destructive — last to land

---

## Changelog

- **v1 (draft)** — 2026-05-27 — Initial draft for code-architecture-advisor + oracle review.
- **v2 (post architecture-advisor)** — 2026-05-27 — 8 variants → 4 mode classes × iterator field; factory split; `decode_fn` to InferencePlan; AR wave-carry reified; `_ConditionalDecodeBase` ABC; pre/post hooks dropped.
- **v3.1 (post oracle round 2 — APPROVED)** — 2026-05-27 — Renamed `ScoreFn` → `DecodeScoreFn` per oracle round-2 caveat (avoids shadowing the existing top-level `ScoreFn` from Sprint 2 COMP-8). No structural changes; spec is implementation-ready.
- **v3 (post oracle round 1)** — 2026-05-27 — Resolved 6 oracle CONCERNs + 6 RECs:
  - C-1 → New `CarryShape` metadata struct (Task 3); v2's `CarrySpec(name=..., shape=...)` API was hallucinated
  - C-2 → STE reproduces tied-group einsum explicitly (Task 10); fixture coverage mandated
  - C-3 → AR scatter-scan stays post-hoc (not fused into iterator); `wave_carry` shape stays `(L,)`
  - C-4 → Glossary added distinguishing `decode_step` (StageSet) from `decode_fn` (InferencePlan)
  - C-5 → Validator Rule 1 reframed as mode-context-wrapped `DispatchRejected` from factory
  - C-6 → W-axis validator rule (v2 Rule 3) dropped — unreachable by API design
  - R-1 → Treedef-invariant tests added to Task 1
  - R-2 → `decode_mode.py` moved to app-side (`inference/decode/mode.py`); protocols also app-side
  - R-3 → Library-surface lint handles TYPE_CHECKING; explicit negative test added
  - R-4 → 3 decode protocols (ScoreFn / ARDecodeFn / STEDecodeFn) instead of 1
  - R-5 → STE hard-cut (no DeprecationWarning); `stage_set` required
  - R-6 → grep audit broadened to `scripts/`, `getattr`, `setattr`, `import_module`, `__import__`
  - Task count: 13 → 14 (CarryShape became its own task)
