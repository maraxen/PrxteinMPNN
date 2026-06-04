# Specification: Heterogeneous Dedup-Batching (`DedupGather` strategy)

> task_id: 260603_het-batch-dedup · sprint: 22 · backlog: #930 · **status: v2.1 — RE-GATE PASS (editorial fixes applied)**
> Authored by specification-specialist. DESIGN ONLY — no implementation this sprint.
> v1 returned UNANIMOUS NEEDS_WORK; v2 resolved every design fix; re-gate returned oracle PASS + arch-advisor PASS + plan-auditor NEEDS_WORK (editorial only). The 3 re-gate fixes — `eq=False` on DedupGather, Task 3 gate test-path, 2a→2b sequencing — are applied below. (v1 preserved in git at 7c5fecb.)

## Overview

Add a fourth `AxisStrategy` variant, `DedupGather`, that collapses N logical axis elements to K unique physical elements, runs the expensive body exactly K times, then scatters results back to all N positions — expressed as a general, JIT-compatible, in-trace static gather/scatter on the existing tiling stack. The mechanism is proven correct by spike (see Spike Evidence).

**Non-goals for this sprint:**
- No changes to `InferencePlan.encode` / `.decode` call sites beyond `_dispatch_axis` (unified paths only).
- No automatic dedup-eligibility inference — caller declares it explicitly.
- Initial eligible set is `{n_structures}` only.
- No threading of `DedupSpec` into `SamplingSpecification` until a caller exists (mirror the `CarrySpec` / `BatchPlanner.carries` precedent).

---

## Spike Evidence

**Mechanism proven at commits 1457251 / c67e5e4.** Script: `scripts/spikes/dedup_encode_kvn_spike.py` + `.bth.toml`.

Under the real `jax.lax.map` path (backend of `SafeMap(tile=1)`):
- `io_callback` runtime counter fired K=3 on an N=6 batch with K=3 unique (body ran K not N); also 3-vs-9.
- Gathered output bit-identical to naive: max abs diff 0.00e+00.
- `@jax.jit` compiled cleanly, no `TracerError`.

The mechanism (`xs[unique_indices]` gather in-trace, `ys_unique[index_map]` scatter in-trace, host numpy computes index arrays once at plan-build) is general — not restricted to any dispatch-layer position. The decisive v1 gate objection ("host-side dedup impossible because the structure axis is heterogeneous → SafeMap(tile=1) → lax.map, traced") is resolved by performing gather AND scatter fully in-trace via static integer arrays.

---

## Acceptance Criteria

1. `DedupGather` is a member of the `AxisStrategy` sealed union (`strategy.py:73` union extended).
2. `_dispatch_axis` (`kernel_dispatch.py`) handles `isinstance(strategy, DedupGather)` without fallthrough.
3. `make_axis_dispatch` (`tiling/dispatch.py`) handles `DedupGather` (raises `DispatchRejected` with a clear message) — no `raise TypeError(Unknown strategy type)` fallthrough on `DedupGather`.
4. An axis with `dedup_eligible=True` can be assigned `DedupGather` by a caller; `dedup_eligible=False` axes raise `TilingError` at `BatchPlanner.plan()` construction time (before any JAX trace).
5. Round-trip test: `DedupGather` output bit-identical (float32, `jnp.array_equal`) or within `atol=1e-6` (bfloat16) to the naive per-element path on a batch with known duplicates.
6. Instrumented test: under `jax.lax.map`, `io_callback` fires `k_bucket` times when N > K.
7. `DedupGather` compiles under `jax.jit`; `xs_unique.shape[0] == k_bucket` after the gather.
8. `DedupBundle` is NOT part of the public API (dropped — fix H).

---

## Fixer Tasks

### Task 1: `DedupFn`, `GatherFn` protocols and `DedupGather` dataclass in `strategy.py`

Add to `src/aminx/tiling/strategy.py`:
1. `DedupFn` protocol — `(xs: PyTree, unique_indices: Int[Array, "K"]) -> PyTree`. Default: `jax.tree.map(lambda x: x[unique_indices], xs)`. JIT-compatible static gather; called in-trace.
2. `GatherFn` protocol — `(ys_unique: PyTree, index_map: Int[Array, "N"]) -> PyTree`. Default: `jax.tree.map(lambda y: y[index_map], ys_unique)`. JIT-compatible scatter; in-trace.
3. `DedupGather` frozen `@dataclass`:
   - `unique_indices: np.ndarray` — `(K_bucket,)` int32 host numpy. Indices into the N-batch selecting K_bucket (padded) unique entries. Computed once at plan-build; never a dynamic JIT value.
   - `index_map: np.ndarray` — `(N,)` int32 host numpy, values in `[0, K_bucket)`. `index_map[i]=k` → position i takes unique slot k. Flat, not padded.
   - `k: int` — raw unique count (pre-padding).
   - `k_bucket: int` — padded static K-bucket (>= k).
   - `dedup_fn: DedupFn` / `gather_fn: GatherFn` — default to the helpers above.

**eqx.partition rationale (fix J):** `DedupGather` is a plain frozen dataclass, not an `eqx.Module`. `AxisDecision`/`BatchPlan` are plain frozen dataclasses, never pytree-registered, never JIT args; strategy is closure-captured at each `_dispatch_axis` call. `dedup_fn`/`gather_fn` are Python callables that must be JIT-compatible because the inner body is traced. `unique_indices`/`index_map` are host numpy → `jnp.asarray` inside `_dispatch_axis` at trace time → static integer arrays in the compiled program. No `eqx.partition` needed. **Declare it `@dataclass(frozen=True, eq=False)`** (re-gate fix): the `np.ndarray` fields would make the synthesized `__eq__`/`__hash__` raise (ambiguous truth value / unhashable ndarray) — every other AxisStrategy variant is eq/hash-safe today, so keep the union consistent and compare via `isinstance` + `np.array_equal`.

**Files:** `strategy.py` (modify), `tiling/__init__.py` (exports).
**Gate:** `uv run pytest tests/tiling/test_strategy.py -q` + new `test_dedup_gather_stores_fields`, `test_dedup_gather_in_union`.
**Ordering:** Before Task 2b and Task 3.

### Task 2a: `dedup_eligible` flag on `AxisSpec` + eligibility guard in `plan()`

Add `dedup_eligible: bool = False` to `AxisSpec` (`planner.py:50-59`). Only `N_STRUCTURES` (`axes.py:48`) gets `dedup_eligible=True`; the other 9 constructions keep the default. (Verified: 10 AxisSpec constructions in axes.py.)

In `BatchPlanner.plan()` (`planner.py:115`): post-phase validation pass — for each `AxisDecision` with `DedupGather`, assert `decision.axis.dedup_eligible`, else `TilingError`. Construction-time, pre-trace.

**Eligibility rationale (fix K):** eligibility is now about SEMANTIC validity (entries can be duplicated and carry no independent per-element state), not dispatch-layer position. `n_structures`: eligible (heterogeneous batches with repeated backbones; encode deterministic per structure). `n_noises`: NOT eligible this sprint (noise values are all-distinct; gate flagged v1's inclusion as wrong). `n_samples`: ineligible (distinct PRNG draws). Flag stays for future axes.

**Phase-1 exclusion:** Phase 0b (Task 2b) adds DedupGather axes to `dedup_names`; Phase 1's heterogeneous demotion skips `dedup_names`, mirroring `phase0_names` (`planner.py:141`).

**Files:** `planner.py:50-59`, `planner.py:115-191`, `axes.py:48`.
**Gate:** `uv run pytest tests/tiling/test_planner_phase0.py -q` + `test_dedup_spec_rejected_on_non_eligible_axis`.
**Ordering:** After Task 1.

### Task 2b: `DedupSpec` and K-bucketing in `dedup.py`

Create `src/aminx/tiling/dedup.py`.

```python
@dataclass(frozen=True)
class DedupSpec:
    axis_name: str
    unique_indices: np.ndarray   # (K,) int32 — raw unique positions
    index_map: np.ndarray        # (N,) int32 — inverse map
    k: int                       # = len(unique_indices)
    dedup_fn: DedupFn | None = None
    gather_fn: GatherFn | None = None
```

`__post_init__`: assert `len(np.unique(index_map)) == k` and `k == len(unique_indices)`. Does NOT check `dedup_eligible` (planner does — fix L).

**K-bucketing (fix B).** Raw `k` varying across batches → XLA retrace per distinct k, erasing savings. Mirror `LENGTH_BUCKETS` (`tiling/buckets.py:19`):

```python
K_DEDUP_BUCKETS = (1, 2, 4, 8, 16, 32)
def get_k_bucket(k: int) -> int:
    for b in K_DEDUP_BUCKETS:
        if k <= b: return b
    raise ValueError(f"k={k} exceeds K_DEDUP_BUCKETS {K_DEDUP_BUCKETS}")
```

`to_dedup_gather()` pads `unique_indices` to `k_bucket` with `np.pad(..., mode="edge")` (padded slots repeat the last valid index → compute valid results that no `index_map` position selects, so they cannot corrupt output). `index_map` is NOT padded. Recompilation key is `k_bucket`, not raw `k`.

Phase 0b calls `to_dedup_gather()` per `DedupSpec` → `AxisDecision(strategy=DedupGather(...))`.

**Files:** `dedup.py` (create), `tiling/__init__.py` (export `DedupSpec`, `K_DEDUP_BUCKETS`, `get_k_bucket`).
**Gate:** `uv run pytest tests/tiling/test_planner_phase0.py tests/tiling/test_strategy.py -q` + `test_k_bucketing_pads_unique_indices`.
**Ordering:** After Task **2a** (re-gate fix — both edit `planner.py:115-191`; 2a adds the `AxisSpec` field + eligibility guard, 2b adds the Phase 0b `dedup_names` block — sequence to avoid a write-conflict).

### Task 3: `DedupGather` dispatch across all THREE dispatch sites (fix C)

**Site 1 — `kernel_dispatch.py:_dispatch_axis` (41-74)**: add branch after `Scan`, before fallback:
```python
if isinstance(strategy, DedupGather):
    unique_idx = jnp.asarray(strategy.unique_indices, dtype=jnp.int32)  # (K_bucket,)
    index_map  = jnp.asarray(strategy.index_map,     dtype=jnp.int32)   # (N,)
    xs_unique  = strategy.dedup_fn(xs, unique_idx)        # in-trace gather
    ys_unique  = _safe_map(body, xs_unique, batch_size=None)  # K_bucket runs
    return strategy.gather_fn(ys_unique, index_map)      # in-trace scatter
```
**xs contract (fix E):** for the structure axis, `xs` into `_dispatch_axis` is `jnp.arange(batch_size)` (integer indices; `kernel_dispatch.py:225/317/382/466`). Default gather yields `jnp.arange(batch_size)[unique_indices]` = K_bucket unique integer indices; the body indexes `batched_ensemble[structure_idx]` inside its closure. Scatter `ys_unique[index_map]` broadcasts to N. Use the real `_safe_map` (`src/aminx/utils/safe_map.py`), not a reimplementation.

**Site 2 — `tiling/dispatch.py:make_axis_dispatch` (67-74)**: add explicit arm raising `DispatchRejected` (DedupGather is handled by `_dispatch_axis`, doesn't map to an iterator) — prevents the `raise TypeError` fallthrough at line 74.

**Site 3 — `host/plan.py:_validate_plan_topology` (247)**: no edit needed — only imports/checks Scan & Vmap (line 261); DedupGather skips both cleanly. Document as confirmed-no-op. Do NOT add a rejection rule here (fix L; eligibility lives in `plan()`).

**Files:** `kernel_dispatch.py`, `tiling/dispatch.py`.
**Gate:** `uv run pytest tests/tiling/test_dispatch.py -q` + `test_make_axis_dispatch_rejects_dedup_gather` (in the existing `test_dispatch.py`; re-gate fix — the prior `tests/host/test_kernel_dispatch.py` does not exist). The `_dispatch_axis` DedupGather branch is exercised by `tests/tiling/test_dedup_gather.py` (Task 4).
**Ordering:** After Tasks 1, 2a, 2b.

### Task 4: Correctness invariant tests (`tests/tiling/test_dedup_gather.py`)
- `test_dedup_gather_bit_identical`: N=4, {0,2}&{1,3} identical (K=2), deterministic lax.map body; `jnp.array_equal` (float32), `atol=1e-6` (bf16).
- `test_dedup_gather_k_not_n_lax_map` (replaces invalid Python-counter test): `io_callback` counter under `@jax.jit`; assert `counter == get_k_bucket(3)`. (Python call counter invalid under vmap/lax.map — body traced once; a pure-Python non-JIT variant is a supplementary sanity check only.)
- `test_dedup_gather_jit_compiles`: no `TracerError`; `xs_unique.shape[0]==k_bucket`.
- `test_dedup_gather_output_shape`: `result.shape[0]==n`.
- `test_dedup_spec_rejected_on_non_eligible_axis` (test_planner_phase0.py): `pytest.raises(TilingError)`.
- `test_make_axis_dispatch_rejects_dedup_gather` (test_dispatch.py): `pytest.raises(DispatchRejected)`.

**Gate:** `uv run pytest tests/tiling/test_dedup_gather.py tests/tiling/test_planner_phase0.py tests/tiling/test_dispatch.py -v`. **Ordering:** After 1, 2a, 2b, 3.

### Task 5: Exports
Export `DedupGather`, `DedupSpec`, `DedupFn`, `GatherFn`, `K_DEDUP_BUCKETS`, `get_k_bucket` from `tiling/__init__.py`. Add `DedupGather` to `__all__` in `strategy.py` (line 75); update module docstring (four strategies). `DedupBundle` NOT exported (dropped).
**Gate:** `python -c "from aminx.tiling.strategy import DedupGather; from aminx.tiling.dedup import DedupSpec, get_k_bucket"` exits 0. **Ordering:** After 1, 2b.

### Task 6: Type-check + lint
`uv run ty check` + `uv run ruff check .` over modified/created files; Protocols satisfy ty strict; `np.ndarray` fields typed precisely.
**Gate:** `uv run ty check && uv run ruff check src/aminx/tiling/ src/aminx/host/kernel_dispatch.py src/aminx/tiling/dispatch.py tests/tiling/test_dedup_gather.py` exits 0. **Ordering:** last.

---

## Open-Question Resolutions
- **OQ#1 (vmap-closure runtime guard):** RESOLVED — mechanism is in-trace JIT-compatible; no host-side constraint; no `cur_sublevel()` guard.
- **OQ#2 (gather_fn placement):** RESOLVED — module-level helper in `dedup.py`, also the default factory; testable in isolation.
- **OQ#3 (DedupBundle):** RESOLVED — DROPPED. `xs_unique` is already a `(K_bucket, ...)` pytree slice; a named wrapper adds surface with no invariant/benefit.
- **OQ#4 (n_noises eligible?):** RESOLVED — NO. Initial set `{n_structures}` only.

---

## Integration Design

| File | Lines (verified) | Change |
|------|-----------------|--------|
| `tiling/strategy.py` | 1–76 | `DedupFn`/`GatherFn` protocols; `DedupGather`; union (73); `__all__` (75) |
| `tiling/dedup.py` | (new) | `DedupSpec`; `K_DEDUP_BUCKETS`; `get_k_bucket()`; `to_dedup_gather()` |
| `tiling/planner.py` | 50–59 | `dedup_eligible: bool = False` on `AxisSpec` |
| `tiling/planner.py` | 115–191 | Phase 0b; `dedup_names` Phase-1 exclusion; post-phase eligibility guard |
| `tiling/axes.py` | 48 | `dedup_eligible=True` on `N_STRUCTURES` only |
| `tiling/dispatch.py` | 67–74 | `DedupGather` arm → `DispatchRejected`; import |
| `host/kernel_dispatch.py` | 55–74 | `DedupGather` branch (after `Scan`, before fallback); import |
| `host/plan.py` | 247 | No change — confirmed-no-op passthrough |
| `tiling/__init__.py` | exports | 6 new names (not `DedupBundle`) |

Union: `AxisStrategy = Vmap | SafeMap | Scan` → `... | DedupGather`.

---

## PRNG Semantics (fix F)

In all four `_sample_batch` paths, `encode_key = jax.random.fold_in(base_key, structure_idx)` (`kernel_dispatch.py:201/280/358/433`). `structure_idx` is the integer index into `batched_ensemble`. Under dedup, N positions sharing a unique structure share the same `structure_idx` → **the same `encode_key`**. This is correct: the encoder is deterministic per (structure, key); shared-backbone entries are semantically identical and should produce identical encodings.

**Caller contract:** `DedupGather` is valid only on axes whose body is deterministic given the unique entry. For `n_structures` this holds (`fold_in(base_key, structure_idx)` is structure-identity-based). Callers introducing non-determinism (stochastic encode) must NOT set `dedup_eligible=True`. `n_samples` is ineligible because `_run_one_sample(k)` takes a distinct PRNG key per draw.

---

## Invariants → Test Obligations

| Invariant | Test | File | Gate |
|-----------|------|------|------|
| Bit-identical (float32) | `test_dedup_gather_bit_identical` | `test_dedup_gather.py` | `jnp.array_equal` |
| Body runs k_bucket not N (lax.map) | `test_dedup_gather_k_not_n_lax_map` | same | `io_callback` counter == k_bucket |
| xs_unique = (k_bucket, ...) | `test_dedup_gather_jit_compiles` | same | shape assertion |
| Output shape (N, ...) | `test_dedup_gather_output_shape` | same | `result.shape[0]==n` |
| JIT-compiles | `test_dedup_gather_jit_compiles` | same | no `TracerError` |
| Ineligible axis rejected | `test_dedup_spec_rejected_on_non_eligible_axis` | `test_planner_phase0.py` | `pytest.raises(TilingError)` |
| In union | `test_dedup_gather_in_union` | `test_strategy.py` | isinstance |
| make_axis_dispatch rejects | `test_make_axis_dispatch_rejects_dedup_gather` | `test_dispatch.py` | `pytest.raises(DispatchRejected)` |

**Observability note:** Python call counter valid only in pure-Python non-JIT `safe_map`; under `vmap`/`lax.map` the body is traced once. Valid runtime observable = `io_callback` (spike-proven, 1457251).

---

## Risk Table

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| XLA retrace as raw k varies | High w/o bucketing | `K_DEDUP_BUCKETS`; key on k_bucket |
| Padded k_bucket slots compute waste | Low | mode="edge" padding → valid results selected by no scatter position; waste bounded by bucket gap |
| encode_key shared across same-structure positions | By design | Correct for deterministic-per-unique bodies; contract documented |
| Wrong caller unique_indices/index_map | Medium | `DedupSpec.__post_init__` asserts `len(np.unique(index_map))==k` and `k==len(unique_indices)` |
| Phase-1 SafeMap demotion fires before DedupGather assigned | High w/o fix | Phase 0b adds `dedup_names`; Phase 1 skips them |
| `make_axis_dispatch` TypeError on DedupGather | Certain w/o fix | Site 2 `DispatchRejected` arm |
| `gather_fn` closes over traced values | Low | Default has no captures; caller contract for customs |
| Ordered io_callback sink + scatter order dependency | Medium | Not safe w/ `ordered=True` sinks; existing `encoder_sink` unordered (safe) |
| Body non-determinism (dropout) loses diversity | Low | Caller contract: only mark deterministic structurally-identical inputs |

---

## Task DAG

```
Task 1 (strategy.py types)
  └── Task 2a (AxisSpec flag + planner guard)
        └── Task 2b (DedupSpec + K-bucketing + dedup.py)
              └── Task 3  (_dispatch_axis + dispatch.py + plan.py no-op)   [needs 1, 2a, 2b]
                    └── Task 4 (tests)        [needs 1, 2a, 2b, 3]
                          └── Task 5 (exports)  [needs 1, 2b]
                                └── Task 6 (ty + ruff)  [needs all]
```
2a → 2b are sequential (re-gate fix — both edit `planner.py:115-191`, so they cannot run in parallel); Task 3 needs 1, 2a, 2b.

---

## Remaining Open Questions (for re-gate)
1. `K_DEDUP_BUCKETS` values `(1,2,4,8,16,32)` — confirm upper bound vs expected production batch sizes, or leave configurable.
2. Default gather/scatter as field `default_factory` vs `None`-then-apply-in-dispatch — confirm `ty` strict accepts Protocol instances as field defaults; if not, use `None` default and apply helper in `_dispatch_axis`.
3. Should `DedupSpec.__post_init__` mirror `CarrySpec`'s eager rejection of structurally-invalid axis names (a `_DEDUP_INELIGIBLE_NAMES` frozenset), in addition to the planner-side eligibility check? Not blocking; fixer judgment.

---

## References
- Spike: `scripts/spikes/dedup_encode_kvn_spike.py` + `.bth.toml` (commits 1457251/c67e5e4)
- Recon `260603_het-batch-dedup_recon01`; gate audit `260603_het-batch-dedup_gate01`; spike `260603_het-batch-dedup_spike01`
- `tiling/strategy.py` 1–76; `tiling/planner.py` 49–59 / 115–191; `tiling/carry.py` 24–58; `tiling/axes.py` 18–129 (N_STRUCTURES:48, N_NOISES:78); `tiling/dispatch.py` 22–74 (TypeError:74); `tiling/buckets.py` 19/27–43
- `host/kernel_dispatch.py` 41–74 / 201,280,358,433 (encode_key) / 225,317,382,466 (jnp.arange xs); `host/plan.py` 247
