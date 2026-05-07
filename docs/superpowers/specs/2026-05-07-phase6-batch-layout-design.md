# Phase 6 Track A — Batch Layout Policy & Memory-Efficient Dispatch

**Date:** 2026-05-07
**Status:** Approved
**Track B** (proxide/prolix migration, shim removal, docstring sweep) deferred to a future sprint.

---

## Problem

The codebase has three independent hand-rolled bucketing implementations and no unified policy for when to `vmap` vs `safe_map` across batched dimensions. The current default — pad every multistate structure to a global bucket and vmap over all axes — inflates memory multiplicatively when:

- Axes are heterogeneous (proteins of different lengths, multistate stacks with varying per-state sequence lengths)
- Combinatorial sweeps tile multiple axes simultaneously (states × samples × temperatures × noises)
- GPU budgets are modest relative to the product of vmapped axis sizes

This design replaces manual stacking with a computation-level batch planner that decides which axes to `vmap` (homogeneous, budget-permitting) and which to iterate via `safe_map` (heterogeneous or budget-exceeded). `safe_map` handles iteration without requiring a common padded shape, directly eliminating the `pad_length_bucket_128` stacking convention for multistate paths. The planner is prxteinmpnn-agnostic and designed for future extraction to jaxbeans.

---

## Constraints

1. **`vmap` and `safe_map` are equivalent for all axes**: every mapped function in this model operates independently on each element — no cross-element communication, no collectives. `jax.vmap(f)(xs)` and `jax.lax.map(f, xs)` produce identical floating-point results. The planner may freely choose either for any homogeneous axis. The parity gate validates this.
2. **`plan()` is pure Python before JIT**: no JAX calls, no tracing, runnable on the host before any compilation.
3. **`utils/batching.py` has zero prxteinmpnn-specific imports**: pure utility module, designed for jaxbeans extraction.
4. **`BatchingConfig` stays unchanged**: 9 existing static fields untouched. BatchPlanner is constructed alongside at host entry points, not embedded in the config.
5. **Memory profiling (StableHLO-based scaling model) deferred**: Phase 6 uses a conservative theoretical estimator. The conditional vs unconditional logit path distinction is a concrete example of why a hardcoded activation multiplier is unreliable — the right long-term model is an HLO-backed empirical fit. StableHLO analysis (XLA buffer assignment → scaling model per `(device_id, fn_hash)`) is Phase 7+ scope. The estimator interface is injected so this swap requires no planner rewrite.
6. **`LENGTH_BUCKETS` stays; `pad_length_bucket_128` is replaced**: `LENGTH_BUCKETS=(100,200,400,800,1200)` (protein-level bucketing in `padding.py`) remains as a data-pipeline grouping tool. `pad_length_bucket_128` (multistate stacking in `state_vmap_prep.py`) is replaced by planner-driven `safe_map` dispatch — heterogeneous axes iterate without requiring a common padded length.

---

## Architecture

### Core types — `utils/batching.py`

All types are `@dataclass(frozen=True)` — **not** `eqx.Module`. The planner is host-side only; no pytree discipline required.

`batch_size=0` is the sentinel for full vectorisation (`jax.vmap`). Any positive value routes to `jax.lax.map(..., batch_size=N)`. Call sites always route through `safe_map(f, xs, batch_size=N)` — no branching on mode strings. **Note:** `utils/safe_map.py` currently uses `batch_size is None` as the vmap sentinel; PR-0 updates it to also treat `batch_size=0` as vmap (None preserved for backward compat).

Axes marked `heterogeneous=True` have elements that may differ in JAX shape across the axis — `vmap` requires homogeneous shapes and cannot be used. The planner pre-demotes these to `safe_map` before the budget loop.

```python
@dataclass(frozen=True)
class AxisSpec:
    name: str
    axis_index: int               # canonical ordering for innermost-first algorithm
    cardinality: int              # typical/max size of this axis
    default_batch_size: int       # 0 = vmap; positive = safe_map chunk size
    heterogeneous: bool           # True = shapes vary across axis; vmap invalid; always safe_map
    doc: str

@dataclass(frozen=True)
class AxisDecision:
    axis: AxisSpec
    batch_size: int               # 0 = vmap; positive = safe_map chunk size
    reasoning: str

@dataclass(frozen=True)
class BatchPlan:
    decisions: list[AxisDecision]
    total_memory_estimate: float   # bytes
    axes_by_index: dict[int, AxisSpec]

    def exceeded_budget(self) -> bool: ...   # returns bool, never raises

@dataclass(frozen=True)
class BatchPlanner:
    axes: list[AxisSpec]
    budget_bytes: float
    estimate_memory: Callable      # injected — theoretical default; HLO-backed later

    def plan(self) -> BatchPlan: ...
```

### Memory estimation

`estimate_memory_theoretical(decisions, base_shape_bytes, activation_multiplier)`:

```
estimate = base_shape_bytes
         × ∏(cardinality_i  for axes_i where batch_size == 0)
         × activation_multiplier
```

Axes with `batch_size > 0` contribute O(1) memory (one chunk at a time) and are excluded from the product. `activation_multiplier` is **required, no default** — the caller must supply it based on their execution context:

| Context | Typical multiplier |
|---------|-------------------|
| Inference only (forward pass) | 2–3× |
| Training, no checkpointing | 4–8× (activations kept for backward) |
| Training + `jax.checkpoint` | 1.5–2× (activations recomputed, not stored) |

These are rough starting points. The conditional vs unconditional logit path is a concrete example of why the right number is not derivable without HLO analysis — the activation profile differs substantially between paths and the hardcoded table cannot capture it. The StableHLO injection point (constraint 5) is how you replace this.

**By-hand example:** `base=1 MB`, `n_states=4` (batch_size=0, vmap — homogeneous), `n_samples=8` (batch_size=0, vmap), `n_temperatures=4` (batch_size=4, safe_map), inference context → `1MB × (4 × 8) × 2.5 = 80 MB`.

The budget is computed by the caller as `device_ceiling × headroom_fraction − param_bytes`. Headroom default: 0.80. The planner never queries the device directly.

**Future swap:** Replace `estimate_memory_theoretical` with an HLO-backed version that parses `BufferAssignmentProto` from `jax.make_jaxpr + XLA compilation` for a few representative shapes, fits a scaling model, and caches the result per `(device_id, fn_hash)`. No planner changes needed.

### Greedy algorithm

Two phases:

1. **Pre-demote heterogeneous axes**: any axis with `heterogeneous=True` is assigned `batch_size=cardinality` (safe_map) unconditionally. These axes never enter the budget loop — vmap requires homogeneous shapes and is not valid for them.

2. **Greedy budget loop over homogeneous axes**: fixed innermost-first ordering (axis_index ascending). Assign `batch_size=0` (vmap) to each axis until the cumulative estimate exceeds budget, then demote to `batch_size=cardinality`. `exceeded_budget()` returns `True` when the plan still exceeds budget even with all homogeneous axes demoted — dispatcher logs a WARNING; no exception.

---

## Axes registry — `utils/batching_registry.py`

Ten canonical `AxisSpec` instances covering all 9 `BatchingConfig` fields:

| Axis | BatchingConfig field(s) | default_batch_size | heterogeneous | rationale |
|------|------------------------|--------------------|---------------|-----------|
| `n_residues` | (length bucket, via `LENGTH_BUCKETS`) | 0 (vmap) | False | fixed after bucketing within a computation |
| `n_states` | — (MultistateStackPayload.n_states) | cardinality (safe_map) | **True** | states may have different per-state sequence lengths; replaces `pad_length_bucket_128` |
| `n_ligand_atoms` | — (atom dim, `ligand_mpnn.py:437`) | 0 (vmap) | False | per-residue atom count is fixed per structure |
| `n_structures` | `batch_size` | cardinality (safe_map) | **True** | proteins in a batch have different lengths before LENGTH_BUCKETS binning |
| `n_samples` | `samples_batch_size`, `samples_chunk_size` | cardinality (safe_map) | False | output accumulates; iterating avoids tiling sample axis into memory |
| `n_temperatures` | `temperature_batch_size` | cardinality (safe_map) | False | scalar sweep axis |
| `n_noises` | `noise_batch_size` | cardinality (safe_map) | False | scalar sweep axis |
| `n_jacobian_pairs` | `jacobian_batch_size` | cardinality (safe_map) | False | residue-pair product can be very large (deferred) |
| `n_combine` | `combine_batch_size` | cardinality (safe_map) | False | multistate combine step (deferred) |
| `n_apc_pairs` | `apc_batch_size`, `apc_residue_batch_size` | cardinality (safe_map) | False | all-pair contact scoring (deferred) |

`default_batch_size=0` signals full vectorisation (vmap); any positive value signals safe_map over that chunk size. `cardinality` means the planner sets `batch_size=axis.cardinality` as the initial safe_map default. The planner may override any homogeneous axis in either direction; heterogeneous axes are always safe_map.

---

## Implementation PRs

These are implementation steps within roadmap Phase 6 — not new roadmap phases.

### PR-0: Foundation

Creates `utils/batching.py`, `utils/batching_registry.py`, and updates `utils/safe_map.py` to treat `batch_size=0` as a vmap sentinel (alongside `None`). No call-site changes. Parity gate must pass before PR-A opens.

### PR-A: Advisory wiring

`BatchingConfig` is unchanged. `BatchPlanner` is constructed alongside it at each host entry point; `plan()` is called before any JIT dispatch and its output is **logged at DEBUG level only — no execution paths change**.

| PR | File | Function | Line | Axes wired |
|----|------|----------|------|------------|
| PR-A | `run/sampling.py` | `_sample_batch` | 744 | N_TEMPERATURES, N_NOISES, N_SAMPLES |
| PR-A | `run/scoring.py` | `score` | 110 | N_NOISES |

### PR-B: Active heterogeneous routing

For axes with `heterogeneous=True` (`n_states`, `n_structures`), the planner's safe_map decision is enforced at dispatch time — no advisory logging, actual dispatch change. This replaces the `pad_length_bucket_128` stacking pattern in multistate paths: instead of padding states to a common multiple-of-128 length and vmapping, the planner iterates via `safe_map(f, states, batch_size=n_states)`.

Execution changes are confined to the heterogeneous-axis dispatch sites; homogeneous axes remain on their existing vmap paths.

### PR-C (optional): Active budget-driven routing

For homogeneous axes that the planner assigns `batch_size > 0` (budget exceeded), swap `jax.vmap(f)(xs)` to `safe_map(f, xs, batch_size=N)`. Only opened if `exceeded_budget=True` appears in >10% of cluster runs over a 7-day observation window after PR-A lands (see trigger below).

**Deferred (future sprint):**
- `run/conformational_inference.py` — fixed per-frame `jax.vmap`, no BatchingConfig fields.
- `run/jacobian.py` — `_compute_jacobian_from_logit_fn` and its N_JACOBIAN_PAIRS, N_COMBINE, N_APC_PAIRS axes. Distinct resource profiles warrant separate treatment.

PRs are sequential (0 → A → B → C). Each must pass the parity gate before the next opens.

---

## Parity gate (every PR)

```bash
PYTHONPATH=prxteinmpnn/src uv run pytest \
  prxteinmpnn/tests/sampling/test_sample.py \
  prxteinmpnn/tests/model/test_ligand_wave_parallel.py \
  prxteinmpnn/tests/sampling/test_state_vmap_exact_jit.py \
  prxteinmpnn/tests/sampling/test_sample_call_kw_contract.py \
  -q
```

---

## PR-C trigger — when to do the actual swap

After PR-A lands, every sampling/scoring run will emit a DEBUG log line like:

```
BatchPlan: n_temperatures=batch_size=4, n_samples=batch_size=0 (vmap), n_noises=batch_size=8, exceeded_budget=False
```

If `exceeded_budget=True` appears in >10% of your cluster runs over a 7-day window, open PR-C. If it never appears, skip PR-C — the current vmap-everywhere approach for homogeneous axes fits your budget.

---

## What this is not

- **Not a general-purpose JAX library** (yet): stays in `utils/batching.py` with no prxteinmpnn imports. Extract to jaxbeans in a later phase.
- **Not replacing StableHLO profiling**: that is the right long-term memory model. This design provides the injection point for it.
- **Not breaking any existing call sites**: `BatchingConfig` constructor signature is unchanged; `scripts/engaging/` imports are unaffected.
