# Phase 6 Track A — Batch Layout Policy & Memory-Efficient Dispatch

**Date:** 2026-05-07
**Status:** Approved
**Track B** (proxide/prolix migration, shim removal, docstring sweep) deferred to a future sprint.

---

## Problem

The codebase has three independent hand-rolled bucketing implementations and no unified policy for when to `vmap` vs `safe_map` across batched dimensions. The current default — pad every multistate structure to a global bucket and vmap over all axes — inflates memory multiplicatively when:

- Axes are heterogeneous (proteins of different lengths batched together)
- Combinatorial sweeps tile multiple axes simultaneously (states × samples × temperatures × noises)
- GPU budgets are modest relative to the product of vmapped axis sizes

No general-purpose JAX library addresses this. The ecosystem's standard answer is to solve shape heterogeneity in the data pipeline (Grain bin-packing, `tf.data.bucket_by_sequence_length`) and pad+mask at compute time. This design builds a computation-level batch planner on top of that — a reusable, prxteinmpnn-agnostic utility for deciding which axes to `vmap` and which to iterate, designed from day one for future extraction to jaxbeans.

---

## Constraints

1. **Parity-pinned paths stay `vmap`**: `state_vmap_exact` against the ligandmpnn reference and the ligand-atom dimension (triple-vmapped in `ligand_mpnn.py:437`) must never be demoted to `safe_map`. The planner routes around them unconditionally.
2. **`plan()` is pure Python before JIT**: no JAX calls, no tracing, runnable on the host before any compilation.
3. **`utils/batching.py` has zero prxteinmpnn-specific imports**: pure utility module, designed for jaxbeans extraction.
4. **`BatchingConfig` stays unchanged**: 9 existing static fields untouched. BatchPlanner is constructed alongside at host entry points, not embedded in the config.
5. **Memory profiling (StableHLO-based scaling model) deferred**: Phase 6 uses a conservative theoretical estimator. StableHLO analysis (XLA buffer assignment → empirical scaling model) is Phase 7+ scope. The estimator interface is injected so this swap requires no planner rewrite.
6. **All bucketing conventions documented, not merged**: `LENGTH_BUCKETS=(100,200,400,800,1200)` (protein-level bucketing in `padding.py`) and `pad_length_bucket_128` (multistate stacking in `state_vmap_prep.py`) serve different purposes and remain separate.

---

## Architecture

### Core types — `utils/batching.py`

All types are `@dataclass(frozen=True)` — **not** `eqx.Module`. The planner is host-side only; no pytree discipline required.

```python
@dataclass(frozen=True)
class AxisSpec:
    name: str
    axis_index: int               # canonical ordering for innermost-first algorithm
    cardinality: int              # typical/max size of this axis
    parity_pinned: bool           # if True, planner never demotes to safe_map
    default_mode: Literal["vmap", "safe_map"]
    doc: str

@dataclass(frozen=True)
class AxisDecision:
    axis: AxisSpec
    chosen_mode: Literal["vmap", "safe_map"]
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
         × ∏(cardinality_i  for axes_i where chosen_mode == 'vmap')
         × activation_multiplier
```

Safe-mapped axes contribute O(1) memory (one slice at a time) and are excluded from the product. Parity-pinned axes are always `vmap`; always included. `activation_multiplier` is **required, no default** — the caller must supply it based on their execution context:

| Context | Typical multiplier |
|---------|-------------------|
| Inference only (forward pass) | 2–3× |
| Training, no checkpointing | 4–8× (activations kept for backward) |
| Training + `jax.checkpoint` | 1.5–2× (activations recomputed, not stored) |

These are starting points, not calibrated values. The StableHLO profiling path (deferred) is how you get the right number for a specific model and device. Until then, when in doubt use the inference multiplier for sampling/scoring runs and the training range for any gradient path.

**By-hand example:** `base=1 MB`, `n_states=4` (vmap, pinned), `n_samples=8` (vmap), `n_temperatures=4` (safe_map), inference context → `1MB × (4 × 8) × 2.5 = 80 MB`.

The budget is computed by the caller as `device_ceiling × headroom_fraction − param_bytes`. Headroom default: 0.80. The planner never queries the device directly.

**Future swap:** Replace `estimate_memory_theoretical` with an HLO-backed version that parses `BufferAssignmentProto` from `jax.make_jaxpr + XLA compilation` for a few representative shapes, fits a scaling model, and caches the result per `(device_id, fn_hash)`. No planner changes needed.

### Greedy algorithm

Fixed innermost-first ordering (axis_index ascending = innermost first). Greedy: assign `vmap` to each axis until cumulative estimate exceeds budget, then demote to `safe_map`. Parity-pinned axes are never demoted regardless of budget. `exceeded_budget()` returns `True` when the final plan still exceeds budget (e.g., because parity-pinned axes alone fill it) — dispatcher logs a WARNING; no exception.

---

## Axes registry — `utils/batching_registry.py`

Ten canonical `AxisSpec` instances covering all 9 `BatchingConfig` fields:

| Axis | BatchingConfig field(s) | parity_pinned | default_mode |
|------|------------------------|---------------|--------------|
| `n_residues` | (length bucket, via `LENGTH_BUCKETS`) | True | vmap |
| `n_states` | — (MultistateStackPayload.n_states) | True | vmap |
| `n_structures` | `batch_size` | False | safe_map |
| `n_samples` | `samples_batch_size`, `samples_chunk_size` | False | safe_map |
| `n_temperatures` | `temperature_batch_size` | False | safe_map |
| `n_noises` | `noise_batch_size` | False | safe_map |
| `n_ligand_atoms` | — (triple vmap at `ligand_mpnn.py:437`) | True | vmap |
| `n_jacobian_pairs` | `jacobian_batch_size` | False | safe_map |
| `n_combine` | `combine_batch_size` | False | safe_map |
| `n_apc_pairs` | `apc_batch_size`, `apc_residue_batch_size` | False | safe_map |

`n_residues` and `n_ligand_atoms` and `n_states` are parity-pinned because their vmap paths are validated against the ligandmpnn reference and must not be rerouted. All other axes default to `safe_map` and can be promoted to `vmap` if budget allows.

---

## Implementation PRs

These are implementation steps within roadmap Phase 6 — not new roadmap phases.

### PR-A through PR-D: advisory wiring (no execution changes)

`BatchingConfig` is unchanged. `BatchPlanner` is constructed alongside it at each host entry point; `plan()` is called before any JIT dispatch and its output is **logged at DEBUG level only — no execution paths change**. This lets you observe what the planner would do without risk.

| PR | File | Function | Line | Axes wired |
|----|------|----------|------|------------|
| PR-A | `run/sampling.py` | `_sample_batch` | 744 | N_TEMPERATURES, N_NOISES, N_SAMPLES |
| PR-B | `run/scoring.py` | `score` | 110 | N_NOISES |
| PR-C | `run/jacobian.py` | `_compute_jacobian_from_logit_fn` | 58 | N_JACOBIAN_PAIRS, N_NOISES, N_COMBINE, N_APC_PAIRS |
| PR-D | `run/conformational_inference.py` | — | — | docs only — fixed per-frame `jax.vmap`, no BatchingConfig fields |

PRs are sequential (A → B → C → D). Each must pass the parity gate before the next opens.

### PR-E (optional): active safe_map adoption

This PR actually changes dispatch — swapping `jax.vmap → jax.lax.map(batch_size=N)` at the hot paths for axes the planner chooses as `safe_map`. This **reduces peak memory** but adds overhead (sequential execution for that axis, more compile time). Only opened if measurement shows the advisory logs firing frequently (see trigger below).

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

## PR-E trigger — when to do the actual swap

After PR-A through PR-D land, every sampling/scoring/jacobian run will emit a DEBUG log line like:

```
BatchPlan: n_temperatures=safe_map, n_samples=vmap, n_noises=safe_map, exceeded_budget=False
```

If `exceeded_budget=True` appears in >10% of your cluster runs over a 7-day window, it means your workloads are genuinely memory-pressured and PR-E is worth doing. If it never appears, skip PR-E entirely — the current vmap-everywhere approach fits your budget and there's no benefit.

**What PR-E actually changes:** replaces specific `jax.vmap(f)(xs)` calls with `jax.lax.map(f, xs, batch_size=N)` for axes the planner marks `safe_map`. The result is lower peak memory but higher wall time for those axes. You choose the trade-off by looking at the logs.

---

## What this is not

- **Not a general-purpose JAX library** (yet): stays in `utils/batching.py` with no prxteinmpnn imports. If the abstraction proves useful across other projects, extract to jaxbeans in a later phase.
- **Not replacing StableHLO profiling**: that is the right long-term memory model. This design provides the injection point for it.
- **Not breaking any existing call sites**: `BatchingConfig` constructor signature is unchanged; `scripts/engaging/` imports are unaffected.
