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
- [x] `tests/run/test_conformational_states_protocol.py` — 6 tests; commit c37a6e4e
- [x] **Item 177**: JAX persistent compilation cache — `host/prep.py`; commit 45a04958
- [x] `eqx.static_field` → `eqx.field(static=True)` deprecation cleanup; commit 92e38197

---

### Sprint 2 — Composability (COMP-1 → COMP-9) ✅

**Praxia backlog IDs:** 184–192

```
Wave A (no deps):
  [x] COMP-1 (#184)  Instantiate EncoderOutput — replace bare encode tuples

Wave B (after COMP-1):
  [x] COMP-2 (#185)  Wire ar_logit_transform into sample_autoregressive.kernel
  [x] COMP-3 (#186)  Add DecodeStepFn + SampleStepFn stages to StageSet
  [x] COMP-6 (#189)  Expose make_encode_fn in inference/encode.py
  [x] COMP-9 (#192)  Consolidate LOGIT_STRATEGIES + run/decode_registry.py

Wave C (after COMP-2 + COMP-3):
  [x] COMP-4 (#187)  Canonicalize bias into stage_set.logit_transform
  [x] COMP-5 (#188)  Collapse three kernels into one StageSet-driven driver

Wave D (after COMP-5):
  [x] COMP-7 (#190)  Open kernel_dispatch to accept resolved DecodeFn from spec

Wave E (after COMP-6 + COMP-7):
  [x] COMP-8 (#191)  Implement InferencePlan / Pipeline protocol  [closes #176]
```

**Invariants (do not change):**
- `InferenceBundle` and sub-bundles — JIT boundary, untouchable
- `LOGIT_STRATEGIES` eqx.Module PyTree pattern — `state_weights` must remain traced leaves
- Kernel math (scatter logic, scan layouts, grad/remat in optimize_ste) — rewire only, never rewrite
- `SamplerFn` / `ScoreFn` top-level signatures — composability work happens below these

---

### Sprint 4 — InferencePlan / Campaign Composability Wiring (#531–536)

**Praxia backlog IDs:** 531–536

```
Wave A (no deps):
  [x] COMP-532  Split build_inference_bundle → (bundle, config) only
                make_stage_set added to inference/logits.py; host/plan.py + all call sites updated
                commit: ec09ecb8

Wave B (after Wave A):
  [x] COMP-533  Move strategy→kernel resolution into make_inference_plan
                _sample_batch now consumer (keyword-only stage_set param);
                runner.py constructs once + functools.partial for streaming;
                sample.py exempted (SamplerFn constraint, COMP-535+).

Wave C (after Wave B):
  [ ] COMP-534  Wire _sample_batch through InferencePlan

Wave D (after Wave C):
  [ ] COMP-535  Expose plan.encode() / plan.decode()

Wave E (after Wave D):
  [ ] COMP-536  Implement campaign.py manifest functions
```

---

### Sprint 3 — Documentation + Known Debt

**Documentation waves (auditor+reviewer gated per wave):**

```
Wave 4 (P0 — done ✅):
  [x] types/bundles.py — all bundle classes (GeometryBundle, ConditioningBundle, LigandBundle,
      WaveScheduleBundle, InferenceBundle, EncodedFeatures, EncoderOutput, PackerResult, PackerBundle)
  [x] types/stages.py — StageSet (topology rules + slots), ConditionalDecodeStep, UnconditionalDecodeStep
  [x] host/plan.py — InferenceComponents, InferencePlan, make_inference_plan + helper functions

Wave 5 (P1 — in progress):
  [x] inference/logits.py — BatchLogitFn, ArithmeticMeanLogits, GeometricMeanLogits, ProductOfProbabilities, ARLogitFuse, TieGroupFuseFn, TieGroupLogsumexpMean, TieGroupProductOfExperts
  [x] model/packer.py — Packer class + __init__ / decode
  [x] model/ligand_mpnn.py — PrxteinLigandMPNN class + __init__
  [x] model/decoder.py — pack_decoder_unconditional_layer_edge_features, pack_conditional_decoder_static_edges, DecoderLayer, Decoder + __call__ / call_conditional

Wave 6 (P2/P3 — done ✅):
  [x] encoder.py — EncoderLayer, Encoder, PhysicsEncoder + magic constants (scale=30.0)
  [x] driver.py — infer_topology, decode, _decode_conditional, _decode_unconditional, decode_ar
  [x] host/runner.py — sample(), _sample_non_streaming_averaged
  [x] Final jaxlint advisory scan (jaxlint internal sensor crash — not a code issue)
```

**Known Debt (non-blocking):**

- **Ruff lint**: 403 fixable errors across `src/` — style/annotation drift, non-blocking.
  Run: `uvx ruff check src 2>&1 | tail -3` to check current count.
- **Tied-positions parity warnings**: 4 `RuntimeWarning` in
  `test_ligand_tied_sampling_weighted_sum_product_alignment` and
  `test_ligand_tied_scoring_arithmetic_mean_alignment` — warn-only, do not affect pass/fail.
- **COMP-533: sample.py stage_set exemption**: `sampling/sample.py:182` calls
  `make_stage_set` inside `sample_sequences` (a `SamplerFn` implementation).
  Cannot be moved without changing the `SamplerFn` protocol signature.
  Planned for COMP-535+ once `plan.encode()` / `plan.decode()` are exposed.
- **COMP-NEW: Unify result-sink topology**: Non-streaming path uses
  `all_sequences.append` (in-memory). Streaming path uses
  `streaming_tensor_sink_session`. Averaged paths should be arbitrary
  (not hardcoded); `FuseFn` should allow custom `io_callback` hooks, but
  always stream via `io_callback` first with FuseFn hooks downstream.
  Scope: collapse streaming/non-streaming within the regular path;
  leave averaged kernel topology separate. ~300 LOC, low blast radius.

---

### Downstream (post-Sprint 2)

- **IREE-WASM**: StableHLO → WASM for browser CPU inference (gate `iree` as optional extras)
- **JAX StableHLO WebGPU+WASM**: `jax.export` → WebGPU shader path
- Both share the `jax.export` artifact from TASK-17 as input
