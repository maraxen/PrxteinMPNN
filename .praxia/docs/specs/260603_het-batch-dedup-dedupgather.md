# Specification: Heterogeneous Dedup-Batching (`DedupGather` strategy)

> task_id: 260603_het-batch-dedup · sprint: 22 · backlog: #930 · status: DRAFT (pre-adversarial-gate)
> Authored by specification-specialist. DESIGN ONLY — no implementation this sprint.

## Overview

Add a fourth `AxisStrategy` variant, `DedupGather`, that collapses N logical axis elements to K unique physical elements, runs the expensive body exactly K times, then scatters results back to all N positions — expressed as a general, pluggable, JIT-compatible axis property in the existing tiling stack.

**Non-goals for this sprint:**
- No implementation of production code.
- No changes to `InferencePlan.encode` / `.decode` call sites beyond `_dispatch_axis`.
- No bucketing interaction beyond the eligibility flag (bucketing redesign is a separate concern).
- No automatic dedup-eligibility inference (caller declares it explicitly).

---

## Acceptance Criteria

1. A `DedupBundle` named tuple (or frozen dataclass) is importable from `prxteinmpnn.tiling.strategy`.
2. `DedupGather` is a member of the `AxisStrategy` sealed union (the `|` type alias in `strategy.py`).
3. `_dispatch_axis` in `kernel_dispatch.py` handles `isinstance(strategy, DedupGather)` without fallthrough.
4. An axis with `dedup_eligible=True` on its `AxisSpec` can be assigned `DedupGather` strategy by a caller; axes with `dedup_eligible=False` reject `DedupGather` at `BatchPlanner.plan()` time (raises `TilingError`).
5. A round-trip test confirms that outputs from `DedupGather` dispatch are bit-identical (float32) or within `atol=1e-6` (bfloat16) to the naive per-element path on a batch containing known duplicates.
6. An instrumented test confirms the body executes K times (not N) when N > K using a Python-side call counter (no JIT; pure Python mock body).
7. `DedupGather` compiles under `jax.jit` and composes with the other three strategies in the nested-axis pattern used by `_sample_batch`.

---

## Fixer Tasks

### Task 1: Define `DedupBundle`, dedup-fn and gather-fn protocols, and `DedupGather` dataclass

**What.** Add three new public names to `src/prxteinmpnn/tiling/strategy.py`:

1. `DedupBundle` — a frozen dataclass (or `NamedTuple`) carrying the shape-static, JIT-compatible inputs needed to run one unique element through the body. It is the single element type the body receives inside `_dispatch_axis`; it does not contain the full N-element batch.
2. `DedupFn` protocol — a `Protocol` matching `(xs_flat: PyTree, index_map: Int[Array, "N"]) -> DedupBundle_pytree`. Host-prepared dedup function; receives full xs (N elements) + pre-computed integer index map, returns a pytree of shape `(K, ...)` containing only unique entries. NOT JIT-compiled — called host-side during dispatch setup.
3. `GatherFn` protocol — a `Protocol` matching `(ys_unique: PyTree, index_map: Int[Array, "N"]) -> PyTree`. IS JIT-compatible; called inside the compiled body to scatter K results to N positions.
4. `DedupGather` — a frozen `@dataclass` carrying: `dedup_fn: DedupFn`; `gather_fn: GatherFn`; `index_map: np.ndarray` shape `(N,)` int32 host numpy (values in `[0, K)`); `k: int` (static, known at trace time).

The `eqx.partition` concern: `DedupGather` is a plain frozen dataclass, not an `eqx.Module`. Inside a `filter_jit` context `dedup_fn`/`gather_fn` are Python callables (closures) treated as static/non-array leaves. `_dispatch_axis` calls `dedup_fn` host-side before any JIT context, so `dedup_fn` never crosses the JIT boundary. `gather_fn` does cross it and must itself be `jax.jit`-compatible (caller contract). No `eqx.partition` is needed provided `_dispatch_axis` is structured correctly (Task 3).

**Files:** `src/prxteinmpnn/tiling/strategy.py` (modify), `src/prxteinmpnn/tiling/__init__.py` (modify, add exports).
**Gate:** `uv run pytest tests/tiling/test_strategy.py -q` passes, incl. new `test_dedup_gather_stores_fields`.
**Scope estimate:** ~60 LOC `strategy.py`; ~5 LOC `__init__.py`.

### Task 2: Add `dedup_eligible` flag to `AxisSpec` and wire eligibility guard into `BatchPlanner.plan()`

**What.** Add `dedup_eligible: bool = False` to `AxisSpec` (`planner.py:49-59`). Update all `AxisSpec` construction sites in `src/prxteinmpnn/tiling/axes.py` to pass it (default `False` except eligible set in OQ#3).

In `BatchPlanner.plan()` (`planner.py:115`), add a post-phase validation pass: if any `AxisDecision` carries `DedupGather` but its `AxisSpec.dedup_eligible` is `False`, raise `TilingError`. Fires at plan-construction time, before any JAX trace.

**Planner assignment:** `plan()` does NOT assign `DedupGather` automatically — caller-driven, consistent with the `Scan`/`CarrySpec` precedent (Phase 0). Caller constructs a `DedupSpec` and passes it analogously.

**Task 2b:** Define `DedupSpec` in a new `src/prxteinmpnn/tiling/dedup.py` (keep `carry.py` focused on scan semantics). Mirrors `CarrySpec`:

```python
@dataclass(frozen=True)
class DedupSpec:
    axis_name: str
    dedup_fn: DedupFn       # host-side; extracts K unique entries
    gather_fn: GatherFn     # JIT-compatible scatter
    index_map: np.ndarray   # shape (N,), int32, host numpy
    k: int                  # number of unique entries
```

`BatchPlanner.plan()` Phase 0b reads `DedupSpec` list and produces `AxisDecision` with `DedupGather`, analogous to `CarrySpec`→`Scan`.

**Files:** `planner.py` (modify `AxisSpec`, `plan()`), `axes.py` (modify constructions), `dedup.py` (create), `__init__.py` (export `DedupSpec`).
**Gate:** `uv run pytest tests/tiling/test_planner_phase0.py tests/tiling/test_strategy.py -q`; new `test_dedup_spec_rejected_on_non_eligible_axis`.
**Scope:** ~40 LOC `planner.py`, ~30 LOC `axes.py`, ~50 LOC `dedup.py`.

### Task 3: Implement `DedupGather` dispatch case in `_dispatch_axis`

**What.** In `kernel_dispatch.py`, add a fourth `isinstance` branch to `_dispatch_axis` (lines 41-74):

```python
if isinstance(strategy, DedupGather):
    xs_unique = strategy.dedup_fn(xs, strategy.index_map)        # (K, ...), host side, NOT in jit
    ys_unique = _safe_map(body, xs_unique, batch_size=None)      # or Vmap if K small
    index_map_jnp = jnp.asarray(strategy.index_map, dtype=jnp.int32)  # (N,)
    return strategy.gather_fn(ys_unique, index_map_jnp)
```

**Critical structural note:** Step 1 (`dedup_fn`) must execute outside any `jax.jit` scope. `_dispatch_axis` for a `DedupGather` axis must not be nested inside a vmapped closure → **constraint on eligibility**: `DedupGather` valid only on the outermost axis layer. Eligibility check (Task 2) enforces this; risk table calls it out.

Default gather (structure axis): `def default_gather(ys_unique, index_map): return jax.tree.map(lambda y: y[index_map], ys_unique)` — pure JAX, JIT-compatible, vmap-composable.

**Files:** `kernel_dispatch.py` (modify, import `DedupGather`).
**Gate:** `uv run pytest tests/tiling/test_dispatch.py -q`; new `test_dispatch_axis_dedup_gather_host_call_count`.
**Scope:** ~25 LOC + ~5 LOC import.

### Task 4: Write the correctness invariant tests

New `tests/tiling/test_dedup_gather.py`:
- **`test_dedup_gather_bit_identical`**: N=4, elements {0,2} and {1,3} identical (K=2). Assert `jnp.allclose(result_dedup, result_naive, atol=0.0)` float32; bfloat16 variant `atol=1e-2`.
- **`test_dedup_gather_encodes_k_not_n`**: Python call counter body (no JIT). n=6, k=3. Assert `len(call_log) == 3`.
- **`test_dedup_gather_jit_compiles`**: wrap body in `jax.jit`; assert no `TracerError`, output shape `(N, ...)`.
- **`test_dedup_gather_rejects_ineligible_axis`**: `DedupSpec` for `dedup_eligible=False` axis → `TilingError`.

**Gate:** `uv run pytest tests/tiling/test_dedup_gather.py -v` all pass. **Scope:** ~120 LOC.

### Task 5: Update exports / public API surface
Export `DedupBundle`, `DedupGather`, `DedupSpec`, `DedupFn`, `GatherFn` from `tiling/__init__.py` (+ top-level if re-exported). Add `DedupGather` to `__all__` in `strategy.py`; update module docstring (four strategies).
**Gate:** `python -c "from prxteinmpnn.tiling.strategy import DedupGather, DedupBundle; from prxteinmpnn.tiling.dedup import DedupSpec"` exits 0. **Scope:** ~15 LOC.

### Task 6: Type-check and lint pass
`uv run ty check` + `uv run ruff check .` over modified files; `DedupFn`/`GatherFn` Protocols satisfy ty strict (no `Any` leaks without rationale).
**Gate:** `uv run ty check && uv run ruff check src/prxteinmpnn/tiling/ src/prxteinmpnn/host/kernel_dispatch.py tests/tiling/test_dedup_gather.py` exits 0.

---

## Open-Question Resolutions

### 1. Where does unique-entry detection run? — **Host-side, pre-JIT.**
Dedup needs equality over arbitrary numpy structures; JAX can't hash array contents at trace time. Caller computes `_, index_map = np.unique(structure_ids, return_inverse=True)` in Python, passes host-numpy `int32[N]` on `DedupGather.index_map`. `dedup_fn` slices `xs` → `xs_unique` shape `(K, ...)` at dispatch time. Only the scatter (`gather_fn`) is JIT-compiled; `xs_unique[index_map]` is a static gather, K a static Python int. **Rejected:** in-JIT `jnp.unique` — needs static `size=K`, pads with sentinels, undefined over heterogeneous pytrees.

### 2. `index_map` representation — **Flat padded-regular `int32[N]`, values in `[0, K)`.**
The `np.unique` `return_inverse` output: `index_map[i] = k`. Statically shaped `(N,)`, JIT-compatible gather, lossless, cheap. **Rejected:** ragged (K × max_occurrences + sentinel) — wasteful under unequal cluster sizes; needs dynamic shapes.

### 3. Which axes are dedup-eligible? — **`dedup_eligible: bool` on `AxisSpec`; initial set `{n_structures, n_noises}`.**
Structural property of the axis → declare at axis-definition time (`axes.py`), default `False`. Eligible: `n_structures` (heterogeneous; primary motivating case — shared backbone), `n_noises`. Ineligible: `n_samples` (distinct PRNG keys — dedup would suppress draws), `n_temperatures` (cheap scalar), `n_states` (structurally distinct by construction), `n_residues`/`n_ligand_atoms` (sub-structure; different abstraction).
**`heterogeneous` interaction:** `dedup_eligible=True` + `heterogeneous=True` compatible (the primary case). When a `DedupSpec` is supplied for a heterogeneous axis, `DedupGather` takes precedence over the Phase-1 `SafeMap` demotion (planner must skip Phase-1 demotion for that axis).
**Bucketing interaction:** `DedupGather` and bucketing mutually exclusive on the same axis this sprint; guard rejects `DedupGather` where a `BucketAssignment` is active.

---

## Integration Design: Exact Files and Seams

| File | Lines (current) | Change |
|------|-----------------|--------|
| `tiling/strategy.py` | 1–76 | Add `DedupBundle`, `DedupFn`, `GatherFn`, `DedupGather`; extend `AxisStrategy` union (line 73); update `__all__` (line 75) |
| `tiling/dedup.py` | (new) | `DedupSpec` dataclass with `__post_init__` checking `axis.dedup_eligible` + `len(np.unique(index_map))==k` |
| `tiling/planner.py` | 49–59 (`AxisSpec`) | Add `dedup_eligible: bool = False` |
| `tiling/planner.py` | 115–191 (`plan`) | Phase 0b: `DedupSpec` → `AxisDecision(strategy=DedupGather(...))`; post-phase validation guard; Phase-1 exclusion set `dedup_names` |
| `tiling/axes.py` | 18–129 | `dedup_eligible=True` on `N_STRUCTURES` (~48) and `N_NOISES` (~78) |
| `host/kernel_dispatch.py` | 41–74 (`_dispatch_axis`) | Add `isinstance(strategy, DedupGather)` branch (after `Scan`); import `DedupGather` |
| `tiling/__init__.py` | exports | Export the 5 new names |

Union change: `AxisStrategy = Vmap | SafeMap | Scan` → `... | DedupGather`.

---

## Invariants → Test Obligations

| Invariant | Test | File | Gate |
|-----------|------|------|------|
| Bit-identical (float32) | `test_dedup_gather_bit_identical` | `tests/tiling/test_dedup_gather.py` | `jnp.allclose(atol=0.0)` |
| Body runs K not N | `test_dedup_gather_encodes_k_not_n` | same | `assert len(call_log)==k` (NO jit) |
| JIT-compiles | `test_dedup_gather_jit_compiles` | same | `jax.jit(f)(xs)` no exception |
| Ineligible axis rejected | `test_dedup_spec_rejected_on_non_eligible_axis` | `tests/tiling/test_planner_phase0.py` | `pytest.raises(TilingError)` |
| In AxisStrategy union | `test_dedup_gather_in_union` | `tests/tiling/test_strategy.py` | isinstance / explicit |
| Output shape `(N, ...)` | `test_dedup_gather_output_shape` | `tests/tiling/test_dedup_gather.py` | `result.shape[0]==n` |

**Note:** the K-not-N call-count test must NOT use `jax.jit` (under JIT the body is traced once, not executed N/K times). JIT-side proof would need HLO cost inspection (future work, not a sprint gate).

---

## Risk Table

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| `DedupGather` nested inside a vmapped closure → host `dedup_fn` impossible | High | Eligibility: only outermost-layer axes `dedup_eligible=True`; assert `dedup_fn` called from Python scope (optional `jax.core.cur_sublevel()` guard) |
| Scatter introduces order dependency if body has order-sensitive `io_callback` sinks | Medium | Document: not safe with order-sensitive sinks; existing `encoder_sink` is unordered (safe); don't mix Scan-ordered sinks + DedupGather on same axis |
| Body non-determinism (e.g. dropout) loses diversity for "duplicates" | Low | Caller contract: only mark structurally-identical inputs as duplicates |
| Wrong caller-supplied `k` (K> wastes; K< silently wrong) | Medium | `DedupSpec.__post_init__`: `assert len(np.unique(index_map))==k` (construction-time) |
| XLA recompiles when `k` changes across batches | Medium | `k` is a static Python int; changing it retraces (expected). Caller normalizes k per-experiment or accepts recompilation. Document. |
| `dedup_eligible=True` on `n_structures` conflicts with Phase-1 SafeMap demotion | High | Phase 0b runs before Phase 1; Phase 1 skips `dedup_names` (analogous to `het_names`) |
| Padding waste when K≪N | Low | Not a concern: `xs_unique` is `(K, ...)`, not padded |
| `gather_fn` accidentally closes over traced values | Low | Default `lambda ys, idx: jax.tree.map(lambda y: y[idx], ys)` has no captures; caller contract for customs |

---

## Task Ordering

```
Task 1 (strategy.py types) -> Task 2 (AxisSpec flag + DedupSpec + planner) -> Task 3 (_dispatch_axis)
  -> Task 4 (tests) -> Task 5 (exports) -> Task 6 (type+lint)
```
Tasks 1,2 draftable in parallel; Task 3 depends on both. Tasks 4–6 depend on 1–3.

---

## Open Questions Not Resolved (for gate / user)

1. Hard runtime assertion that `DedupGather` dispatch is not inside a vmapped closure (`jax.core.cur_sublevel()` / context var), or is the plan-time eligibility flag sufficient?
2. Default `gather_fn` placement — classmethod/default-factory on `DedupGather` (ergonomic) vs module-level helper in `dedup.py` (testable in isolation)?
3. Is `DedupBundle` necessary as a named type, or a documentation-only alias for the pytree slice (`xs_unique` is already a pytree slice of `xs`)?
4. Is `n_noises` eligibility worth the API surface (noise sweeps are usually all-distinct), or restrict initial eligible set to `n_structures` only?

---

## References
- Recon `260603_het-batch-dedup_recon01`
- `tiling/strategy.py` (union 1–76), `tiling/planner.py` (`AxisSpec` 49, `plan()` 115), `tiling/carry.py` (`CarrySpec` precedent), `tiling/axes.py` (18–129)
- `host/kernel_dispatch.py` (`_dispatch_axis` 41–74, `_sample_batch` 77–507), `host/plan.py` (`encode`/`decode` 388–459), `types/encodings.py` (`EncoderOutput` 10–22)
