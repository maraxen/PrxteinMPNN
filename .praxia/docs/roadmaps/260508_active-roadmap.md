# aminx Active Roadmap

> **Replaces:** `.agents/REFACTOR_ROADMAP.md` (deprecated 2026-05-08 — phases 0–6 complete).
> **Last updated:** 2026-06-02

---

## What Is Done

| Work | Last commit |
|------|-------------|
| Phase 5 (mpnn.py split, SamplingDriver, DesignSink, io\_callback streaming) | `6dd995d` |
| Phase 6 Track A (BatchPlanner, safe\_map for n\_structures in sampling + scoring) | `73c63a0` |
| Code quality: ruff fixes, N806/N803 per-file suppression, FBT partial | `b1e37dc` |
| **MODELINPUTS PR-1**: `WaveParallelPayload`, `BackboneGeometry`, `SamplingInputs`, `ScoringInputs`, `SamplingStaticConfig`, `ScoringStaticConfig`, `DecodeFnRegistry` | `04c9248` |
| **MODELINPUTS PR-2**: Host adapter layer (`make_sampling_inputs_from_spec`, `make_static_config_from_spec`) | `5e9eefe` |
| **MODELINPUTS PR-3**: `batch_fn` as `static_argnames` at `_sample_sequences_jitted` boundary; `state_weights` promoted off `**kwargs` | `6d449b3` |
| **Pipeline Protocol** (all 13 tasks): `LogitTransformFn`, `EncoderOutput`, `EncoderPreFn`/`EncoderPostFn`/`ModelProtocol`/`Pipeline` protocols, UID-based hook registry, `PipelineFns`, `LogitTransformFn` wired into unconditional + conditional scoring (protein + ligand), four concrete pipelines (Unconditional, Conditional, Autoregressive, STE), clean method aliases (`score_unconditional_from_payload` etc.), `DeprecationWarning` shims for `state_vmap_exact` naming | `38415b8` |
| **Sprint 5** (AxisStrategy sealed union, safe\_scan, CarrySpec, AxisBoundary, BatchPlanner Phase 0, unified driver) | `97a703b7` |
| **Sprint 6** (DecodeMode sealed union, ConditionalDecode/UnconditionalDecode/AutoregressiveDecode/STEDecode, MapIterator/ScanIterator, make\_decode\_fn factory, driver.py retired to 113 lines) | See `inference/decode/` |

---

## Active Work (Near-Term)

### A. MODELINPUTS PR-4 — Push `model.__call__` boundary

**Goal:** `Aminx.__call__(self, inputs: SamplingInputs) -> (OneHot, Logits)` — all three branch methods accept `SamplingInputs` as a single pytree, not positional arrays.

**Why:** Eliminates the remaining 5-7 positional array unpacking at the model call site; enables `jax.lax.switch` on a single pytree operand; required before PR-5 StableHLO export.

**Files:** `src/aminx/model/mpnn.py`, `src/aminx/model/ligand_mpnn.py`

**Reference:** `.praxia/REFACTOR_MODELINPUTS.md §PR-4`

---

### B. MODELINPUTS PR-5 — StableHLO export verification + `Optional[Array]` cleanup

**Goal:** Smoke-test `jax.export.export(jitted_model)(SamplingInputs(...))` end-to-end. Delete remaining `Optional[Array]` params from JIT-boundary methods. Update external scripts.

**Files:** model + run files; `scripts/run_design_grid.py`, `scripts/run_unconditional_logits_grid.py`

**Depends on:** PR-4 complete.

**Reference:** `.praxia/REFACTOR_MODELINPUTS.md §PR-5`

---

### C. EncoderPreFn / EncoderPostFn wiring

**Goal:** The `EncoderPreFn` and `EncoderPostFn` Protocol slots exist in `PipelineFns` and are registered via the UID registry, but the actual call sites inside the encoder are not wired. Wire them.

- `EncoderPreFn`: `(BackboneGeometryStack, state_index) -> FeaturesInput` — inserted before `self.features(...)` call in the encoder.
- `EncoderPostFn`: `(EncoderOutput, state_index) -> EncoderOutput` — inserted after the encoder returns `(node_features, edge_features, neighbor_indices, mask)`.

**Files:** `src/aminx/model/mpnn.py` (encoder call site), `src/aminx/model/ligand_mpnn.py`

**Why now:** Needed for cosine-similarity multistate residue scoring (EncoderPostFn) and custom node-feature initialization (EncoderPreFn). Protocols + registry are already in place.

---

### D. `multi_state_temperature` removal from `_from_payload` signatures

**Goal:** `multi_state_temperature` is a geometric-mean logit-combination parameter. With `LogitTransformFn` now in place, it should be captured in the transform closure rather than threaded through every `_from_payload` signature. Remove it from the public method signatures and the `score_*_from_payload` / `sample_*_from_payload` aliases; have callers encode it in their registered `LogitTransformFn`.

**Files:** `src/aminx/model/mpnn.py`, `src/aminx/model/ligand_mpnn.py`, `src/aminx/pipeline/unconditional.py`, `src/aminx/pipeline/conditional.py`

**Note:** Only remove from the `_from_payload` / pipeline-facing surface. The inner scan path still uses it; leave that untouched until the multistate scan path is also Pipeline-ized.

---

## Medium-Term

### E. BatchPlanner coverage — jacobian + conformational inference

**Goal:** The BatchPlanner/safe\_map dispatch was wired for sampling (`n_structures`) and scoring, but jacobian and conformational inference paths were explicitly deferred. Apply the same pattern.

**Files:** `src/aminx/run/jacobian.py`, `src/aminx/run/conformational_inference.py`

**Reference:** Phase 6 deferred items in session history.

---

### F. Phase 6 Track B — Proxide/Prolix migration + deprecation shim removal

**Goal:** Migrate structure/IO utilities duplicated between `aminx` and `proxide` to proxide. Migrate trajectory/MD wrappers to prolix. Remove the `RunSpecification` deprecation shims added in Phase 1/3 (after one minor-version window).

**Depends on:** Proxide/Prolix version stabilization.

---

### G. jaxbeans DEPEND wiring

**Goal:** Wire the jaxbeans DEPEND pieces that Phase 5 declared but didn't land: `core/profiling` (`assert_zero_copy_overhead`), `utils/mapping` (`safe_map` canonical source), `utils/io` (`atomic_write`, `MultiPartWriter`), `core/safety` (`PreemptionHandler`), `jax_io/sources` (`BinaryDatasetWriter`).

**Note:** Currently using mirrored utils in `aminx/utils/`. Swap imports once jaxbeans hits 0.1.0 (or is added as workspace member).

---

## Deferred Indefinitely

| Item | Blocker |
|------|---------|
| Phase 5h: `ensemble/dbscan.py` + `ensemble/pca.py` → jaxbeans | Requires jaxbeans maintainer to confirm `ml/clustering/` target and open jaxbeans-side PR |
| Phase 7: Lint / type / coding standards enforcement | Owner to specify exact standards before work can be scoped |
| Generalized `StateIndex` type (richer than `Int[Array, "S"]`) | Deferred from pipeline plan; revisit when multistate design use cases clarify the required interface |
| Lazy `__init__.py` (PEP 562) | Cold-import measurement after Phase 1 didn't flag a problem; demoted to opt-in experiment |

---

## Key Files Quick Reference

| Purpose | File |
|---------|------|
| Active sprint plan (ModelInputs) | `.praxia/REFACTOR_MODELINPUTS.md` |
| Pipeline Protocol plan (done) | `docs/superpowers/plans/2026-05-08-pipeline-protocol.md` |
| Pipeline package | `src/aminx/pipeline/` |
| Hook registry | `src/aminx/pipeline_registry.py` |
| PipelineFns | `src/aminx/pipeline_fns.py` |
| Protocols | `src/aminx/protocols.py` |
| ModelInputs types | `src/aminx/model_inputs.py` |
| Payloads (MultistateStackPayload, WaveParallelPayload) | `src/aminx/payloads.py` |
| BatchPlanner | `src/aminx/utils/batching.py` |
| Deprecated roadmap (historical) | `.agents/REFACTOR_ROADMAP.md` |
