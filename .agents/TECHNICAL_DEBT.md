# Technical Debt & Future Work

**Last Updated:** 2026-05-06 (Phase 3b sprint plan; Phase 0a / §11 #10 split)

This document tracks known technical debt, experimental features, and planned improvements.

---

## Phase 0a spike vs roadmap §11 item 10

**Roadmap:** `.agents/REFACTOR_ROADMAP.md` §227–238 (Phase 0a), §11 checklist #10 (amended 2026-05-06).

| Slice | Meaning |
| :--- | :--- |
| **Spike slice (Phase 3b PR1)** | `tests/sampling/spikes/test_state_vmap_exact_spike.py`: numeric match `state_vmap_exact` vs explicit `jax.vmap` stack at `get_tolerances("float32")`; HLO text stats via `UserWarning` (`pytest -W default`); go/no-go for **entering** Phase 4 recorded in the PR that lands PR1+ |
| **§11 #10 full item** | Above **plus** “matching Phase 4 implementation” — satisfied only when Phase 4 registry/unification (or routing-on-no-go) merges |

**Q6 artifact:** SPIKE PR / sprint notes must record numeric result and HLO narrative; Phase 4 PR references that record.

**Recorded verdict (local CI agent, 2026-05-06):** fast spike test **GO** — `score_unconditional_state_vmap_exact` matches explicit `jax.vmap` stack at `get_tolerances(float32)`; HLO warning emitted (`spike_hlo_state_vmap_exact bytes=...`). Heavy arm not run (`REFERENCE_PATH` unset). Formal PR should restate after human review.

---

## Phase 3 PR6 — `scripts/` specification constructors (audit)

**Status:** complete for this checkout (2026-05-06)  
**Roadmap:** `.agents/REFACTOR_ROADMAP.md` §11 checklist #8, §14 sprint status

Patterns searched: `SamplingSpecification|ScoringSpecification|TrainingSpecification|RunSpecification\(` under `scripts/**/*.py`.

| File | Constructor / import | Verdict | Notes |
|------|------------------------|---------|-------|
| `scripts/collect_parity_evidence.py` | `SamplingSpecification(...)`, `ScoringSpecification(...)` | **current** | Uses public subclass ctors; compatible with `RunSpecification` façade + `run_spec` sync. No JSON migration required for this offline tool. |
| `scripts/260410/verify_massive_sampling.py` | `SamplingSpecification(...)` | **current** | Smoke / load-test script; same constructor surface as library. |
| `scripts/overfit/overfit_check.py` | `TrainingSpecification(...)` | **current** | Training smoke path; mirrors `training/specs` API. |
| `scripts/260410/verify_design_storage.py` | `DesignArrayRecordWriter(...)` only | **out of scope** | Exercises ArrayRecord I/O, not run specs. |
| `scripts/engaging/` | — | **absent** | Not present in this checkout; engaging-cluster scripts (if any) must be re-audited when vendored here. |

**Policy:** New scripts should prefer `run_specification_from_json` / `prxteinmpnn spec validate` for saved configs (roadmap §13 Q4 JSON-first).

---

## 1. Precision Casting (Experimental)

**Status:** 🟡 Experimental  
**Priority:** High  
**Source:** Training branch merge

### Description

The training pipeline now supports mixed precision (bf16/fp16/fp32) training via model parameter casting. While training works correctly, checkpoint loading/resumption needs additional testing and stabilization.

### Current Behavior

1. Model is initialized in float32
2. Parameters are cast to target precision after loading
3. Optimizer state is initialized with precision-cast parameters
4. Checkpoint restoration requires matching abstract optimizer state
5. Training reads the dtype label from the composed :class:`~prxteinmpnn.run.spec.RunSpec` (``run_spec.precision.compute``); :class:`~prxteinmpnn.training.specs.TrainingSpecification`\ ``.precision`` remains the user-facing field and is mirrored there at construction time.

### Known Issues

- **Float32 → bf16 loading:** Requires explicit dtype conversion when loading float32 checkpoints into bf16 training
- **Optimizer state mismatch:** If saved optimizer state dtype doesn't match current model dtype, restoration may fail silently or produce incorrect results

### Required Work

- [ ] **Always save weights in float32:** Ensure `eqx.tree_serialise_leaves` saves weights in portable float32 format
- [ ] **Add dtype conversion on load:** Implement automatic dtype conversion when restoring checkpoints
- [ ] **Add integration tests:**
  - Test bf16 training → save → load → resume in bf16
  - Test bf16 training → save → load → resume in fp32
  - Test fp32 training → save → load → resume in bf16
- [ ] **Document precision strategy:** Add user guide section on precision selection

### Code pointers

- `src/prxteinmpnn/run/spec.py` — `PrecisionConfig`, `build_run_spec` / `_run_spec_precision_compute`
- `src/prxteinmpnn/training/trainer.py`: `get_compute_dtype()`, `_training_precision()`, `_init_checkpoint_and_model()`

### References

- `.agents/TRAINING_MERGE.md`: Section 5.2 (Model Casting), Section 12.1

---

## 2. Resource Allocation Configuration

**Status:** 🟡 Partially implemented  
**Priority:** Medium

### Description

**Go (2026-05-05):** The pinned resolver environment’s `proxide` (`0.1.0a3` per `uv run python -m importlib.metadata version proxide`) exposes `create_protein_dataset(..., ram_budget_mb=..., max_workers=..., max_buffer_size=...)`. Inference prep (`run/prep.py`) and training dataloaders (`training/trainer.py`) call `compute_resource_allocation` via `proxide_dataset_resource_kwargs` in `run/resources.py` and pass those keyword arguments through. Optional `max_buffer_size` is forwarded from `getattr(spec, "max_buffer_size", None)` when a spec gains that field later; today it is typically `None`.

### Required Work

- [x] Add `host_resource_allocation_strategy`, `ram_budget_mb`, `max_workers` to `RunSpecification`
- [x] Implement `compute_resource_allocation()` helper (`run/resources.py`)
- [x] Add `psutil` dependency (`pyproject.toml`)
- [x] Call `compute_resource_allocation` from `prep_protein_stream_and_model` / training setup and thread limits into `create_protein_dataset` (or proxide equivalents)
- [ ] Consider pinning `proxide` in `pyproject.toml` to a version range known to keep these kwargs stable
- [ ] Test on various hardware configurations

### Code pointers

- `src/prxteinmpnn/run/prep.py` — `prep_protein_stream_and_model` / `proxide_dataset_resource_kwargs`
- `src/prxteinmpnn/run/resources.py` — `compute_resource_allocation`, `proxide_dataset_resource_kwargs`
- `src/prxteinmpnn/training/trainer.py` — `_create_dataloaders`, final test `create_protein_dataset` path

### References

- `.agents/TRAINING_MERGE.md`: Section 3

---

## 3. Grain Debug Mode

**Status:** ✅ No unconditional debug in tree (audit 2026-05-05)  
**Priority:** Low

### Description

Historical training merge enabled Grain `py_debug_mode` unconditionally. Current `src/` contains no `grain.config` / `py_debug_mode` toggles.

### Required Work

- [ ] If Grain debug is needed again: gate behind `PRXTEINMPNN_GRAIN_DEBUG=1` and document briefly

### References

- `.agents/TRAINING_MERGE.md`: Section 3.4

---

## 4. HDF5 ARM64 Compatibility

**Status:** ✅ Primary stack uses `h5py` (audit 2026-05-05)  
**Priority:** Low (residual audit only)

### Description

Project dependencies list `h5py`; no `tables` / PyTables imports under `src/`. Residual risk: optional scripts, notebooks, or unpinned environments.

### Required Work

- [x] Standardize on `h5py` in `pyproject.toml`
- [ ] Quick audit of `scripts/` and docs for PyTables references
- [ ] Spot-check on ARM64 if CI does not cover it

### References

- `.agents/TRAINING_MERGE.md`: Section 2

---

## 5. Gradient Accumulation

**Status:** 🟡 Implemented (needs testing)  
**Priority:** Medium

### Description

Gradient accumulation support via `accum_steps` parameter is implemented but needs comprehensive testing.

### Required Work

- [ ] Verify gradient scaling is mathematically correct
- [ ] Test with `accum_steps` = 1, 2, 4, 8
- [ ] Document effective batch size calculation
- [ ] Add example in training guide

### Code pointers

- `src/prxteinmpnn/training/trainer.py` — `TODO(tech-debt)` §5 (`accum_steps` path)

### References

- `.agents/TRAINING_MERGE.md`: Section 5.3

---

## 6. Docstring preservation and API docs

**Status:** 🟡 Ongoing hygiene  
**Priority:** Medium

### Description

Post–training-merge, keep module and public API docstrings complete on core surfaces (especially modules that absorbed large diffs).

### Affected Files (prioritize)

- `src/prxteinmpnn/utils/data_structures.py`
- `src/prxteinmpnn/model/decoder.py`
- `src/prxteinmpnn/model/mpnn.py`

### Required Work

- [ ] Spot-check public symbols vs. docstrings; fill gaps where behavior is non-obvious
- [ ] Align any stale references to removed modules (`io/operations.py`, etc.)

### Code pointers

- `src/prxteinmpnn/utils/data_structures.py` — `TODO(tech-debt)` §6
- `src/prxteinmpnn/model/decoder.py` — `TODO(tech-debt)` §6

### References

- `.agents/TRAINING_MERGE.md`: Section 6, 7, 8

---

## 7. Proxide/Prolix migration and deduplication

**Status:** 🟡 In progress  
**Priority:** High

### Description

`proxide` and `prolix` are declared dependencies (`pyproject.toml`). Parsing is routed through proxide (`io/parsing/dispatch.py`); datasets use `proxide.ops.dataset`. Remaining work is **audit and delete** any dead shims, duplicated constants, and docs that still describe pre-migration layouts.

### Required Work

- [x] Add proxide / prolix to dependencies
- [ ] Inventory remaining overlap (utils, weights, training paths) vs. proxide/prolix APIs
- [ ] Remove obsolete modules and update tests
- [ ] Drop legacy dependencies if any creep back in (none named `jax_md` in `pyproject.toml` today)

### Code pointers

- `src/prxteinmpnn/io/parsing/dispatch.py` — `TODO(tech-debt)` §7

### Modules to Preserve (conceptual)

- PrxteinMPNN-specific model I/O and training glue; avoid re-home generic parsing/physics that proxide/prolix already own

### References

- `.agents/TRAINING_MERGE.md`: Sections 12-16

---

## 8. Submodule Development Workflow

**Status:** 🔵 Reference  
**Priority:** Low

### Description

For development spanning proxide, prolix, and PrxteinMPNN, use git submodules with uv editable installs. This is documented in `TRAINING_MERGE.md` Section 16.

### Key Points

- **NEVER merge submodules to main**
- Development branches only
- Use `[tool.uv.sources]` for editable installs
- Remove submodule config before merging to main

### References

- `.agents/TRAINING_MERGE.md`: Section 16

---

## Legend

- 🟢 Planned - Work not started
- 🟡 In Progress / Experimental - Partially complete or needs testing
- 🔴 Blocked - Cannot proceed without external input
- 🔵 Reference - Documentation/process, not code work
- ✅ Complete

---

## 9. MPNN module split and public surface

**Status:** 🟢 Planned  
**Priority:** High

### Description

`src/prxteinmpnn/model/mpnn.py` is a large, multi-concern module. For long-term maintainability, split it into focused units (encoding, decoding branches, ligand handling, switches, helpers) with clearer boundaries, naming, and documentation.

### Required work

- [ ] Split implementation across multiple modules under `model/` with a thin `mpnn.py` or package `__init__` re-exporting the stable API
- [ ] Tighten data contracts at module boundaries (typed inputs/outputs, consistent optional fields)
- [ ] Improve docstrings and variable names where the split exposes unclear roles
- [ ] Update imports across the codebase and keep parity tests green

### References

- `src/prxteinmpnn/model/mpnn.py` — top-of-file `TODO(tech-debt)` + state-batching note
- `docs/TODO_BLOCKED_MODULES.md`, `docs/FULL_FUNCTIONALITY_TODO.md` — may still describe removed modules; reconcile when editing those docs (see §13)

---

## 10. Contracts: dataclasses, `Protocol`, and related abstractions

**Status:** 🟢 Planned  
**Priority:** Medium

### Description

Evaluate where **dataclasses**, **`typing.Protocol`**, and small **type aliases** clarify expectations vs. where they add ceremony. Goal: explicit, checkable contracts for factories (e.g. logits Fns), run specs, and structure batch payloads without over-abstracting hot JAX paths.

### Required work

- [ ] Audit high-churn APIs (`run/specs.py`, sampling factories, model `__call__` surfaces) for missing or informal contracts
- [ ] Prototype `Protocol` definitions for callable factories (pattern already suggested in `docs/TODO_BLOCKED_MODULES.md` for logits modules)
- [ ] Document conventions (when to use frozen dataclass vs. Equinox module vs. PyTree dict) in one short internal note or module docstring policy

### Code pointers

- `src/prxteinmpnn/run/specs.py` — `TODO(tech-debt)` §10 (`RunSpecification`)
- `src/prxteinmpnn/utils/types.py` — `TODO(tech-debt)` §10 (shared aliases)

### References

- `src/prxteinmpnn/sampling/conditional_logits.py`, `unconditional_logits.py` — factories exist; tighten `Protocol` usage at boundaries
- `docs/TODO_BLOCKED_MODULES.md` (may be stale vs. current imports; see §13)

---

## 11. StableHLO export and WASM compilation compatibility

**Status:** 🟢 Planned  
**Priority:** Medium

### Description

Ensure critical compiled paths remain exportable to **StableHLO** and compatible with **WASM** toolchains where applicable (avoid or isolate Python-only control flow, host callbacks, and non-lowering ops in exported regions). No in-repo references yet; treat as a cross-cutting requirement for new model and sampling refactors.

### Required work

- [ ] Identify entrypoints intended for export (inference-only forward, scoring kernels, etc.)
- [ ] Add smoke tests that `jax.export` / lowering succeeds for those entrypoints (policies for `shard_map`, custom calls, effects)
- [ ] Document unsupported patterns (e.g. `io_callback` in export paths) and keep them behind explicit boundaries
- [ ] Track WASM compiler requirements (IREE, `jit` portability) and pin any needed JAX/XLA constraints

### Code pointers

- `src/prxteinmpnn/model/__init__.py` — `TODO(tech-debt)` §11

---

## 12. Move ensemble analytics (DBSCAN, PCA, …) to jaxbeans

**Status:** 🟢 Planned  
**Priority:** Medium

### Description

General-purpose **DBSCAN**, **PCA**, and related ensemble utilities in `prxteinmpnn.ensemble` belong in **jaxbeans** (shared JAX utilities) so PrxteinMPNN stays domain-focused and other projects can reuse them.

### Modules and consumers (current)

- **Sources:** `src/prxteinmpnn/ensemble/dbscan.py`, `src/prxteinmpnn/ensemble/pca.py` (and call sites in `run/conformational_inference.py`, `ensemble/ci.py`, `run/specs.py`)
- **Tests:** `tests/ensemble/test_dbscan.py`, `tests/ensemble/test_pca.py`

### Required work

- [ ] Add equivalent APIs to jaxbeans; depend on jaxbeans from PrxteinMPNN
- [ ] Replace `prxteinmpnn.ensemble.{dbscan,pca}` imports with jaxbeans; keep thin re-exports or delete after a deprecation window
- [ ] Move or duplicate tests into jaxbeans; trim PrxteinMPNN tests to integration-only if needed
- [ ] Coordinate `pcax` / typing (`PCAInputData` in `utils/types.py`) as part of the move

### Code pointers

- `src/prxteinmpnn/ensemble/dbscan.py` — `TODO(tech-debt)` §12
- `src/prxteinmpnn/ensemble/pca.py` — `TODO(tech-debt)` §12

---

## 13. Repository hygiene and stale documentation

**Status:** 🟢 Planned  
**Priority:** Medium

### Description

Housekeeping across the tree: trim or refresh docs that no longer match code (e.g. feature-parity lists, “blocked module” notes pre-dating `conditional_logits.py` / `unconditional_logits.py`), normalize `docs/` vs `.agents/` guidance, remove dead scripts, keep CI (`ruff`, `ty`, `pytest`) warnings trending to zero, and ensure high-signal README/AGENTS pointers stay accurate.

### Required work

- [ ] Refresh `docs/FULL_FUNCTIONALITY_TODO.md`, `docs/TODO_BLOCKED_MODULES.md`, and physics integration docs against current modules
- [ ] Archive or delete obsolete planning files if superseded by `TECHNICAL_DEBT.md`
- [ ] Repo-wide pass: unused imports, orphaned tests, duplicate `RunSpecification`/`sample` top-of-file boilerplate where harmless to dedupe
- [ ] Optionally add a lightweight `scripts/` + `docs/` index in-tree (one list) so navigation stays obvious

### Code pointers

- `src/prxteinmpnn/__init__.py` — package entry `TODO(tech-debt)` §13

---

## 14. Host-side I/O streaming (`io_callback` / `effects_barrier`)

**Status:** 🟡 In progress (see inline TODOs)  
**Priority:** High

### Description

Reduce device materialization and Python-side concat bottlenecks by standardizing on `jax.experimental.io_callback` + `jax.effects_barrier()` for large outputs. Detailed checklist lives in `TODO_io_callback.txt`; sampling, scoring, designs, and metrics call sites carry `TODO(io_callback integration)` comments.

### References

- `TODO_io_callback.txt`
- `src/prxteinmpnn/run/sampling.py` — module-level `TODO(tech-debt)` §14

---

## Source-indexed TODOs (quick inventory)

| § | Location | Theme |
|:--|:---------|:------|
| 14 | `TODO_io_callback.txt` | Master checklist for **`io_callback` + `effects_barrier`** |
| 14 | `src/prxteinmpnn/io/designs.py` | Host handoff via io_callback; skip redundant `device_get` |
| 14 | `src/prxteinmpnn/run/sampling.py` | Streaming / HDF5 / concat paths |
| 14 | `src/prxteinmpnn/run/jacobian.py` | Jacobian batching / D2H review |
| 14 | `src/prxteinmpnn/run/scoring.py` | Batched lists |
| 14 | `src/prxteinmpnn/profiling/sampler_profile.py` | Bench vs. io_callback interaction |
| 14 | `src/prxteinmpnn/training/metrics.py`, `trainer.py` | Telemetry `device_get` |
| 9 | `src/prxteinmpnn/model/mpnn.py` | Mega-module split + attention batching idea |
| — | `src/prxteinmpnn/sampling/ste_optimize.py` | Derive `n_states` from mapping / weights |
| 13 | `docs/FULL_FUNCTIONALITY_TODO.md`, `docs/TODO_BLOCKED_MODULES.md`, `docs/PHYSICS_*.md` | **Stale until audited** — parity / blocked-module narrative may predate current code |

---

## User notes

- We will need to have proxide actually on PyPi in the stable release. Right now we will just use the latest from GitHub.
