# Technical Debt & Future Work

**Last Updated:** 2025-12-19

This document tracks known technical debt, experimental features, and planned improvements.

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

### References

- `src/prxteinmpnn/training/trainer.py`: `get_compute_dtype()`, `_init_checkpoint_and_model()`
- `.agents/TRAINING_MERGE.md`: Section 5.2 (Model Casting), Section 12.1

---

## 2. Resource Allocation Configuration

**Status:** 🟢 Planned (from training merge)  
**Priority:** Medium

### Description

The training branch hardcoded resource allocation values. The merge plan includes making these configurable via `RunSpecification`.

### Required Work

- [ ] Add `host_resource_allocation_strategy`, `ram_budget_mb`, `max_workers` to `RunSpecification`
- [ ] Implement `compute_resource_allocation()` helper
- [ ] Add `psutil` dependency if not present
- [ ] Update `create_protein_dataset()` to use configurable resources
- [ ] Test on various hardware configurations

### References

- `.agents/TRAINING_MERGE.md`: Section 3

---

## 3. Grain Debug Mode

**Status:** 🟢 Planned (from training merge)  
**Priority:** Low

### Description

The training branch unconditionally enabled Grain debug mode, which should be environment-configurable.

### Required Work

- [ ] Make debug mode toggleable via `PRXTEINMPNN_GRAIN_DEBUG` environment variable
- [ ] Document in README or configuration guide

### References

- `.agents/TRAINING_MERGE.md`: Section 3.4

---

## 4. HDF5 ARM64 Compatibility

**Status:** 🟢 Planned (from training merge)  
**Priority:** High

### Description

The `tables` (PyTables) package has compatibility issues on Grace Hopper/ARM64 architecture. Replace with `h5py` which has native arm64 wheels.

### Required Work

- [ ] Replace `tables` with `h5py` in `pyproject.toml`
- [ ] Update any `tables`-specific imports/API calls to `h5py`
- [ ] Test on ARM64 hardware

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

### References

- `.agents/TRAINING_MERGE.md`: Section 5.3

---

## 6. Docstring Preservation

**Status:** ⚠️ Action Required During Merge  
**Priority:** High

### Description

The training branch removed docstrings from several files. These MUST be preserved from main during merge.

### Affected Files

- `src/prxteinmpnn/utils/data_structures.py`
- `src/prxteinmpnn/model/decoder.py`
- `src/prxteinmpnn/model/mpnn.py`
- `src/prxteinmpnn/io/operations.py`

### Action

When merging, use `git checkout main -- <file>` as starting point and manually apply functional changes from training branch while keeping documentation.

### References

- `.agents/TRAINING_MERGE.md`: Section 6, 7, 8

---

## 7. Proxide/Prolix Migration

**Status:** 🟢 Planned  
**Priority:** High

### Description

PrxteinMPNN contains duplicated code that now exists in proxide (parsing, force fields) and prolix (physics calculations). This duplication should be removed by migrating to use these libraries as dependencies.

### Modules to Migrate

**To Proxide:**

- `src/prxteinmpnn/io/parsing/` (most files)
- `src/prxteinmpnn/physics/force_fields.py`
- `src/prxteinmpnn/utils/residue_constants.py`

**To Prolix:**

- `src/prxteinmpnn/physics/electrostatics.py`
- `src/prxteinmpnn/physics/vdw.py`
- `src/prxteinmpnn/physics/constants.py`

### Modules to Preserve

- `src/prxteinmpnn/physics/features.py` - PrxteinMPNN-specific node feature computation
- `src/prxteinmpnn/io/loaders.py` - Grain-based dataset creation
- `src/prxteinmpnn/io/operations.py` - Collation/batching

### Required Work

- [ ] Add proxide to dependencies
- [ ] Add prolix to dependencies
- [ ] Update imports in `io/parsing/dispatch.py`
- [ ] Update imports in `physics/features.py`
- [ ] Delete deprecated modules
- [ ] Remove jax_md dependency
- [ ] Update all affected tests

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
