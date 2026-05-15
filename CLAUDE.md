# prxteinmpnn CLAUDE.md

## Commands

| Action | Command |
| :--- | :--- |
| **Type Check** | `uv run ty check` |
| **Lint** | `uv run ruff check .` |
| **Format** | `uv run ruff format .` |
| **Tests** | `uv run pytest` |
| **JAX advisory** | `uv run jaxlint check src --no-doc` (optional; not a CI gate) |

## Tech Stack

- **Language**: Python 3.12+
- **ML Framework**: JAX + Equinox
- **Package Manager**: uv
- **Type Checking**: ty (strict)
- **Linting**: ruff
- **Testing**: pytest

## Code Style

- Strict typing with `ty`, format with `ruff`
- JAX: use `jax.jit`, `jax.vmap`, `jax.lax.scan` patterns
- Equinox: modules as dataclasses, `eqx.filter_jit` for PyTrees
- Numerical tolerance tests, cross-framework validation

---

## CURRENT_SPRINT

**Branch:** `refactor-full` | **Root:** `/home/marielle/projects/tev_design/prxteinmpnn`

### Instructions for agents

- Keep this DAG current as tasks complete: update status inline (`[ ]` → `[x]`).
- Mirror every status change to the praxia backlog (`mcp__praxia__backlog` action: `update` or `complete`).
- Do not begin a Wave N task until all its dependencies are marked `[x]`.
- Commit after each task with `task-id` in the message.

---

### Sprint 1 — Quality Hardening (Waves 0–4) ✅

All waves complete as of 2026-05-15. Commits on `refactor-full`.

**Remaining tail items:**

- [x] `tests/inference/test_vmap_axis_contract.py` — 11 tests; commits 45a04958, 9fd99858
- [ ] `tests/run/test_conformational_states_protocol.py` — duck-type compat: `n_states: int` accepted by `RunSpecification`
- [x] **Item 177**: JAX persistent compilation cache — `host/prep.py`; commit 45a04958
- [ ] `eqx.static_field` → `eqx.field(static=True)` deprecation cleanup in `types/configs.py`

---

### Sprint 2 — Composability (COMP-1 → COMP-9)

**Praxia backlog IDs:** 184–192

```
Wave A (no deps):
  [ ] COMP-1 (#184)  Instantiate EncoderOutput — replace bare encode tuples

Wave B (after COMP-1):
  [ ] COMP-2 (#185)  Wire ar_logit_transform into sample_autoregressive.kernel
  [ ] COMP-3 (#186)  Add DecodeStepFn + SampleStepFn stages to StageSet
  [ ] COMP-6 (#189)  Expose make_encode_fn in inference/encode.py
  [ ] COMP-9 (#192)  Consolidate LOGIT_STRATEGIES + run/decode_registry.py

Wave C (after COMP-2 + COMP-3):
  [ ] COMP-4 (#187)  Canonicalize bias into stage_set.logit_transform
  [ ] COMP-5 (#188)  Collapse three kernels into one StageSet-driven driver

Wave D (after COMP-5):
  [ ] COMP-7 (#190)  Open kernel_dispatch to accept resolved DecodeFn from spec

Wave E (after COMP-6 + COMP-7):
  [ ] COMP-8 (#191)  Implement InferencePlan / Pipeline protocol  [closes #176]
```

**Invariants (do not change):**
- `InferenceBundle` and sub-bundles — JIT boundary, untouchable
- `LOGIT_STRATEGIES` eqx.Module PyTree pattern — `state_weights` must remain traced leaves
- Kernel math (scatter logic, scan layouts, grad/remat in optimize_ste) — rewire only, never rewrite
- `SamplerFn` / `ScoreFn` top-level signatures — composability work happens below these

---

### Downstream (post-Sprint 2)

- **IREE-WASM**: StableHLO → WASM for browser CPU inference (gate `iree` as optional extras)
- **JAX StableHLO WebGPU+WASM**: `jax.export` → WebGPU shader path
- Both share the `jax.export` artifact from TASK-17 as input
