# Training Branch Merge & Proxide/Prolix Migration Guide

**Date Created:** 2025-12-19  
**Last Updated:** 2025-12-19  
**Status:** Part A COMPLETE, Part B COMPLETE (100% Integrated)

**Branches:** `main` ← `origin/training`  
**Merge Base Commit:** `ba52e980` (changed multistate)

## Overview

This document covers two major integration efforts:

### Part A: Training Branch Merge [DONE]

The `training` branch contains significant updates to the training pipeline including:

- Floating-point precision control (bf16/fp16/fp32)
- Gradient accumulation support
- Improved checkpoint loading
- Data processing scripts for PDB preprocessing
- NumPy-based collation (avoiding premature JAX conversion)
- Numerical stability fixes in model forward pass

### Part B: Proxide/Prolix Migration [IN PROGRESS]

Completing the migration to use **proxide** (structure parsing, force field handling, oxidize Rust extension) and **prolix** (MD simulation, physics calculations) as external dependencies. This will:

- Remove duplicated physics code from PrxteinMPNN
- Consolidate force field handling
- Enable shared improvements across projects

This guide provides step-by-step integration instructions with careful attention to preserving main branch content where appropriate.

---

## Table of Contents

### Part A: Training Branch Merge

1. [Pre-Merge Preparation](#1-pre-merge-preparation)
2. [Dependency Changes](#2-dependency-changes)
3. [RunSpecification Enhancements](#3-runspecification-enhancements)
4. [TrainingSpecification Updates](#4-trainingspecification-updates)
5. [Trainer Module Updates](#5-trainer-module-updates)
6. [Model Updates (precision fixes)](#6-model-updates-precision-fixes)
7. [Data Loading & Operations](#7-data-loading--operations)
8. [Data Structures](#8-data-structures)
9. [New Data Processing Scripts](#9-new-data-processing-scripts)
10. [Checkpoint Module](#10-checkpoint-module)
11. [Post-Merge Cleanup](#11-post-merge-cleanup)

### Part B: Proxide/Prolix Migration

12. [Migration Overview](#12-migration-overview)
13. [Proxide Integration](#13-proxide-integration)
14. [Prolix Integration](#14-prolix-integration)
15. [Deprecation & Removal](#15-deprecation--removal)
16. [Development Workflow with Submodules](#16-development-workflow-with-submodules)

### Appendix

17. [Known Issues & Future Work](#17-known-issues--future-work)

---

## 1. Pre-Merge Preparation

### 1.1 Create a Feature Branch

```bash
git checkout main
git pull origin main
git checkout -b merge/training-integration
```

### 1.2 Review Current State

```bash
# View commits on training branch not in main
git log main..origin/training --oneline

# View file changes
git diff main..origin/training --stat
```

### 1.3 Commits to Integrate

| Commit | Message | Key Changes |
|--------|---------|-------------|
| `cf026118` | initial commit, debugged training data prep | Data processing scripts |
| `57d98259` | training fixes | Trainer loop improvements |
| `4262b133` | updated training set up | Loader/collation changes |
| `8dbfc84f` | added grad accum to specs | `accum_steps` field |
| `fcc6fb58` | fp precision toggle | Precision casting throughout |

---

## 2. Dependency Changes

### 2.1 Remove `tables` (PyTables) - Use `h5py` Instead

**Rationale:** The `tables` package has arm64/aarch64 compatibility issues on Grace Hopper architecture. Replace with `h5py` which has native arm64 wheels.

**Action:**

In `pyproject.toml`:

```diff
dependencies = [
    ...
-   "tables>=3.10.2",
+   "h5py>=3.10.0",
    ...
]
```

**Verification:** Ensure any existing `tables` imports are migrated to `h5py`:

```bash
grep -r "import tables" src/ scripts/
grep -r "from tables" src/ scripts/
```

If found, replace with equivalent `h5py` API calls.

### 2.2 Update `uv.lock`

After modifying `pyproject.toml`:

```bash
uv lock
```

---

## 3. RunSpecification Enhancements

### 3.1 Add Resource Allocation Fields

Add the following new fields to `src/prxteinmpnn/run/specs.py` in the `RunSpecification` class:

```python
@dataclass
class RunSpecification:
    # ... existing fields ...
    
    # Host Resource Allocation
    host_resource_allocation_strategy: Literal["auto", "full"] = "auto"
    """Strategy for host resource allocation.
    
    - "auto": Use defaults based on context (90% RAM for training, 50% for inference)
    - "full": Use maximum available resources as detected by automatic inspection
    
    When manually specified values are provided (ram_budget_mb, max_workers),
    they take precedence over inferred defaults.
    """
    
    ram_budget_mb: int | None = None
    """RAM budget for data loading in megabytes.
    
    Priority order:
    1. If "full" strategy: Use maximum available RAM
    2. If explicitly set: Use this value
    3. If None with "auto": 90% of host RAM for training, 50% for inference
    """
    
    max_workers: int | None = None
    """Maximum number of data loading workers.
    
    Priority order:
    1. If "full" strategy: Use all CPU cores
    2. If explicitly set: Use this value  
    3. If None with "auto": All cores for training, 50% for inference
    """
```

### 3.2 Add Resource Computation Helper

Add utility function (can be in `run/specs.py` or a new `run/resources.py`):

```python
import os
import psutil
from typing import Literal

def compute_resource_allocation(
    strategy: Literal["auto", "full"],
    ram_budget_mb: int | None,
    max_workers: int | None,
    context: Literal["training", "inference"] = "inference",
) -> tuple[int, int]:
    """Compute effective RAM budget and max workers.
    
    Returns:
        Tuple of (ram_budget_mb, max_workers)
    """
    total_ram_mb = psutil.virtual_memory().total // (1024 * 1024)
    total_cpus = os.cpu_count() or 4
    
    if strategy == "full":
        # Full strategy: use maximum resources
        effective_ram = total_ram_mb
        effective_workers = total_cpus
    else:
        # Auto strategy: context-dependent defaults
        if context == "training":
            default_ram_pct = 0.90  # 90% for training
            default_worker_pct = 1.0  # All cores for training
        else:
            default_ram_pct = 0.50  # 50% for inference
            default_worker_pct = 0.5  # Half cores for inference
        
        effective_ram = int(total_ram_mb * default_ram_pct)
        effective_workers = max(1, int(total_cpus * default_worker_pct))
    
    # Manually specified values take precedence
    if ram_budget_mb is not None:
        effective_ram = ram_budget_mb
    if max_workers is not None:
        effective_workers = max_workers
    
    return effective_ram, effective_workers
```

### 3.3 Integrate into Loaders

Update `src/prxteinmpnn/io/loaders.py` to use resource allocation:

```python
from prxteinmpnn.run.specs import compute_resource_allocation

def create_protein_dataset(
    # ... existing params ...
    host_resource_allocation_strategy: Literal["auto", "full"] = "auto",
    ram_budget_mb: int | None = None,
    max_workers: int | None = None,
    context: Literal["training", "inference"] = "inference",
    # ...
):
    # Compute effective resources
    effective_ram, effective_workers = compute_resource_allocation(
        strategy=host_resource_allocation_strategy,
        ram_budget_mb=ram_budget_mb,
        max_workers=max_workers,
        context=context,
    )
    
    # ... rest of function ...
    
    performance_config = prefetch_autotune.pick_performance_config(
        ds=ds,
        ram_budget_mb=effective_ram,
        max_workers=effective_workers,
        max_buffer_size=None,
    )
```

### 3.4 Remove Hardcoded Debug Mode

In `src/prxteinmpnn/io/loaders.py`, make debug mode configurable:

```diff
-grain.config.update("py_debug_mode", True)
+import os
+if os.environ.get("PRXTEINMPNN_GRAIN_DEBUG", "0") == "1":
+    grain.config.update("py_debug_mode", True)
```

---

## 4. TrainingSpecification Updates

### 4.1 Add New Fields

Update `src/prxteinmpnn/training/specs.py`:

```python
@dataclass
class TrainingSpecification(RunSpecification):
    # ... existing fields ...
    
    # Gradient Accumulation
    accum_steps: int = 1
    """Number of gradient accumulation steps.
    
    Effective batch size = batch_size (which must be divisible by accum_steps).
    """
    
    def __post_init__(self) -> None:
        super().__post_init__()
        
        # ... existing validation ...
        
        if self.accum_steps < 1:
            msg = "accum_steps must be >= 1"
            raise ValueError(msg)
        
        if self.batch_size % self.accum_steps != 0:
            msg = f"batch_size ({self.batch_size}) must be divisible by accum_steps ({self.accum_steps})"
            raise ValueError(msg)
```

### 4.2 DO NOT Remove Fields

The training branch removes `model_weights` and `model_version` from `TrainingSpecification`. These are inherited from `RunSpecification`, so the removal is correct. However, verify they are properly inherited before merging.

The field `preprocessed_index_path` is already in `RunSpecification` (line 99), so it should not be duplicated in `TrainingSpecification`.

---

## 5. Trainer Module Updates

### 5.1 Precision Handling

The training branch adds comprehensive precision handling. Key changes:

```python
def get_compute_dtype(precision: str) -> jnp.dtype:
    """Get JAX dtype from precision string."""
    if precision == "bf16":
        return jnp.bfloat16
    elif precision == "fp16":
        return jnp.float16
    else:
        return jnp.float32
```

### 5.2 Model Casting

Add model parameter casting in `_init_checkpoint_and_model`:

```python
# Cast model to target precision
compute_dtype = get_compute_dtype(spec.precision)
if compute_dtype != jnp.float32:
    logger.info(f"Casting model parameters to {compute_dtype}")
    
    def _cast_fn(x):
        return x.astype(compute_dtype) if eqx.is_inexact_array(x) else x
    
    model = jax.tree_util.tree_map(_cast_fn, model)
```

### 5.3 Gradient Accumulation

The training branch adds gradient accumulation via `spec.accum_steps`. Integrate by:

1. Asserting `batch_size % accum_steps == 0` in dataloader creation
2. Passing `accum_steps` and `compute_dtype` to `train_step`
3. Updating `train_step` signature to accept these parameters

### 5.4 Checkpoint Loading Improvements

The training branch improves checkpoint restoration to support:

- Loading from `.eqx` files (final model)
- Loading from Orbax checkpoints
- Proper dtype handling when loading float32 weights into bf16 model

**Key change:** `restore_checkpoint` now requires `abstract_opt_state` parameter.

### 5.5 JIT Caching

The training branch adds JIT caching optimization:

```python
filter_jitted_train_step = eqx.filter_jit(train_step)
filter_jitted_eval_step = eqx.filter_jit(eval_step)
```

This should be on the step functions, called once before the training loop.

---

## 6. Model Updates (Precision Fixes)

### 6.1 Decoder (`src/prxteinmpnn/model/decoder.py`)

**Accept from training branch:**

- Stability fix: Accumulate message sums in float32:

  ```python
  message_f32 = message.astype(jnp.float32)
  aggregated_message_f32 = jnp.sum(message_f32, -2) / scale
  aggregated_message = aggregated_message_f32.astype(compute_dtype)
  ```

- Cast attention masks to prevent dtype promotion:

  ```python
  mask_cast = attention_mask.astype(compute_dtype)
  ```

**PRESERVE from main branch:**

- All docstrings and documentation

### 6.2 MPNN (`src/prxteinmpnn/model/mpnn.py`)

**Accept from training branch:**

- Dtype matching for `jax.random.gumbel`:

  ```python
  jax.random.gumbel(key, shape, dtype=logits_with_bias.dtype)
  ```

- Consistent output dtype in `_call_unconditional` and `_call_conditional`

**PRESERVE from main branch:**

- All docstrings and documentation

### 6.3 Features (`src/prxteinmpnn/model/features.py`)

**Accept:**

```diff
-backbone_noise = jnp.array(0.0, dtype=jnp.float32)
+backbone_noise = jnp.array(0.0)
```

### 6.4 Coordinates (`src/prxteinmpnn/utils/coordinates.py`)

**Accept:**

```python
noise = jax.random.normal(coord_key, coords.shape, dtype=coords.dtype)
scaled_noise = (backbone_noise * noise).astype(coords.dtype)
return coords + scaled_noise
```

---

## 7. Data Loading & Operations

### 7.1 Operations (`src/prxteinmpnn/io/operations.py`)

**Accept from training branch:**

- NumPy-based padding and stacking (avoid premature JAX conversion)
- `Protein.from_tuple_numpy()` method usage
- `override` parameter in `pad_and_collate_proteins`

**PRESERVE from main branch:**

- All docstrings and documentation
- Type hints

**Merge Strategy:**

1. Take the functional changes from training
2. Restore docstrings from main
3. Keep proper type annotations

### 7.2 Loaders (`src/prxteinmpnn/io/loaders.py`)

**Accept:**

- `override=use_preprocessed` parameter passing

**Modify (per Section 3):**

- Remove hardcoded `grain.config.update("py_debug_mode", True)`
- Remove hardcoded `ram_budget_mb=65536, max_workers=64`
- Use configurable resource allocation

---

## 8. Data Structures

### 8.1 `src/prxteinmpnn/utils/data_structures.py`

**Accept from training branch:**

- `none_or_numpy()` helper function
- `Protein.from_tuple_numpy()` classmethod

**PRESERVE from main branch:**

- All docstrings for `Protein`, `EstatInfo`, `EMFitterResult`, etc.
- Detailed class attribute documentation

**Merge Example:**

```python
def none_or_numpy(array: np.ndarray | None) -> np.ndarray | None:
    """Convert array to numpy array, or return None if input is None.
    
    Args:
        array: Input numpy array or None.
        
    Returns:
        Converted numpy array or None.
    """
    if array is None:
        return None
    return np.asarray(array)
```

---

## 9. New Data Processing Scripts

### 9.1 Create Directory Structure

Move training data processing scripts to a dedicated directory:

```bash
mkdir -p scripts/data_processing
```

### 9.2 Move Scripts

The following files should be moved from `src/prxteinmpnn/training/data/` to `scripts/data_processing/`:

| Source | Target |
|--------|--------|
| `combine_shards.py` | `scripts/data_processing/combine_shards.py` |
| `create_index.py` | `scripts/data_processing/create_index.py` |
| `debug_shards.py` | `scripts/data_processing/debug_shards.py` |
| `debug_worker.py` | `scripts/data_processing/debug_worker.py` |
| `process_parallel.py` | `scripts/data_processing/process_parallel.py` |
| `upload_dataset.py` | `scripts/data_processing/upload_dataset.py` |

Also move:

- `scripts/debug_performance.py` → `scripts/data_processing/debug_performance.py`
- `scripts/original_training_data/original_training.py` → `scripts/data_processing/original_training.py`

### 9.3 Add README

Create `scripts/data_processing/README.md`:

```markdown
# Data Processing Scripts

Scripts for preprocessing and managing training datasets.

## Scripts

### `process_parallel.py`
Parallel processing of PDB/Torch `.pt` files into ArrayRecord format.

### `combine_shards.py`
Combine multiple ArrayRecord shards into a single file.

### `create_index.py`
Create train/valid/test split index from cluster metadata.

### `upload_dataset.py`
Upload processed dataset to cloud storage.

### `debug_*.py`
Debugging utilities for data pipeline issues.

## Usage

See each script's `--help` for detailed usage instructions.
```

### 9.4 Keep PDB Metadata

Keep `src/prxteinmpnn/training/data/pdb_2021aug02/pdb_2021aug02/README` in place as dataset documentation.

---

## 10. Checkpoint Module

### 10.1 `src/prxteinmpnn/training/checkpoint.py`

**Accept from training branch:**

- `abstract_opt_state` parameter in `restore_checkpoint`:

```python
def restore_checkpoint(
    manager: ocp.CheckpointManager,
    model_template: PrxteinMPNN,
    abstract_opt_state: optax.OptState | None = None,  # NEW
    step: int | None = None,
) -> tuple[PrxteinMPNN, optax.OptState, TrainingMetrics, int]:
```

**Add validation:**

```python
if abstract_opt_state is None:
    msg = "abstract_opt_state must be provided to restore optimizer state correctly"
    raise ValueError(msg)
```

---

## 11. Post-Merge Cleanup

### 11.1 Run Tests

```bash
# Run full test suite
uv run pytest tests/ -v

# Run training-specific tests
uv run pytest tests/training/ -v

# Type checking
uv run pyright src/prxteinmpnn/
```

### 11.2 lint

```bash
uv run ruff check src/ scripts/
uv run ruff format src/ scripts/
```

### 11.3 Documentation Update

Update any relevant documentation to reflect new parameters:

- `README.md`
- API documentation
- Training guide

### 11.4 Remove Artifacts

Check for and remove any debug artifacts:

```bash
git diff origin/training -- | grep -E "print\(|breakpoint\(|pdb\."
```

---

## 12. Migration Overview

### 12.1 Project Relationships

```
PrxteinMPNN (this project)
├── Uses: proxide (parsing, force fields, structure handling)
└── Uses: prolix (MD simulation, physics calculations)

proxide (maraxen/proxide)
├── oxidize (Rust extension via maturin)
├── io/parsing (structure parsing)
├── physics/force_fields (force field handling)
└── utils/residue_constants

prolix (maraxen/prolix)  
├── physics/* (Coulomb, LJ, GBSA, etc.)
├── simulation/* (integrators, minimization)
└── Uses: proxide (for parsing)
```

### 12.2 Migration Goals

| Goal | Status | Notes |
|------|--------|-------|
| Replace local parsing with proxide | 🟡 Planned | `io/parsing/` → proxide |
| Replace local force fields with proxide | 🟡 Planned | `physics/force_fields.py` → proxide |
| Replace local physics with prolix | 🟡 Planned | `physics/electrostatics.py`, `physics/vdw.py` → prolix |
| Remove jax_md dependency | 🟡 Planned | Prolix handles MD primitives |
| Preserve PrxteinMPNN-specific physics features | ✅ Keep | `physics/features.py` - computes node features |

---

## 13. Proxide Integration

### 13.1 What Proxide Provides

Proxide consolidates structure parsing AND feature extraction:

- **Structure parsing**: PDB, PQR, mmCIF, HDF5 trajectories
- **Force field handling**: AMBER, CHARMM force field loading
- **Rust extension (oxidize)**: Fast parsing, hydrogen addition, parameterization
- **Residue constants**: Atom types, residue mappings
- **Physics Features**: Electrostatic and vdW features computed at parse time
- **Geometry Features**: RBF (radial basis function) features

### 13.2 OutputSpec - Feature Extraction Control

Proxide's `OutputSpec` struct controls which features are extracted during parsing:

```python
from oxidize import OutputSpec, parse_structure

spec = OutputSpec(
    # Geometry features
    compute_rbf=True,           # Radial Basis Functions
    rbf_num_neighbors=30,       # K nearest neighbors for RBF
    
    # Physics features
    compute_electrostatics=True,
    electrostatics_noise=None,  # Optional noise for data augmentation
    compute_vdw=True,
    
    # Force field parameterization
    parameterize_md=True,
    force_field="amber14-all",
)

result = parse_structure("protein.pdb", spec)
# result contains:
#   - "rbf_features": (N_res, K, 400) - 25 pairs × 16 bases
#   - "electrostatic_features": (N_res, 5) - field magnitude at backbone atoms
#   - "vdw_features": (N_res, 5) - LJ energy at backbone atoms
```

### 13.3 Feature Comparison: Proxide vs Current PrxteinMPNN

| Feature | Proxide (oxidize) | PrxteinMPNN (JAX) | Notes |
|---------|-------------------|-------------------|-------|
| **Electrostatics** | Field magnitude `\|E\|` at backbone | Forces projected onto backbone frame | Different representation |
| **vdW** | LJ energy at backbone | Forces projected onto backbone frame | Energy vs Force |
| **RBF** | 25 pairs × 16 bases = 400 features | 25 pairs × 16 bases = 400 features | **Equivalent!** |
| **When computed** | During parsing (Rust) | Runtime in forward pass (JAX) | 10-100x faster in Rust |
| **Differentiable** | No (precomputed) | Yes (JAX autodiff) | Trade-off |

**Note on RBF:** PrxteinMPNN already uses RBF features via `utils/radial_basis.py` and `model/features.py`. The proxide implementation uses the same 25 backbone pairs and 16 Gaussian bases, so they should be equivalent. The main difference is:

- **PrxteinMPNN**: Computes RBF at runtime in JAX (differentiable, but slower)
- **Proxide**: Precomputes RBF during parsing in Rust (faster, but fixed)

### 13.4 Migration Decision: Physics Features

**Two approaches:**

#### Option A: Use Proxide Features (Recommended for Training)

Use proxide's Rust-computed features at parsing time. Faster, but not differentiable.

**Pros:**

- 10-100x faster (Rust vs JAX)
- Computed once during preprocessing
- Cleaner architecture (no JAX physics code)

**Cons:**

- Not differentiable (can't backprop through features)
- Different feature representation (field magnitude vs projected forces)

**When to use:** Training, where features are precomputed and fixed.

#### Option B: Keep JAX Features (For Gradient-Based Methods)

Keep `physics/features.py` for cases requiring gradients through physics features.

**Pros:**

- Fully differentiable
- Maintains existing API

**Cons:**

- Slower than Rust
- Duplicated logic

**When to use:** Research on physics-aware loss functions, feature learning.

### 13.5 Recommended Hybrid Approach

1. **For training data preprocessing**:
   - Use proxide `OutputSpec` with `compute_electrostatics=True, compute_vdw=True, compute_rbf=True`
   - Store precomputed features in ArrayRecord files
   - Remove runtime physics computation from training loop

2. **For inference**:
   - Use proxide for fast feature extraction
   - No need for JAX physics at inference time

3. **For research/gradient-based methods**:
   - Optionally keep `physics/features.py` as a fallback
   - Mark as experimental/research-only

### 13.6 Updated Modules to Replace

| PrxteinMPNN Module | Replacement | Action |
|--------------------|-------------|--------|
| `src/prxteinmpnn/io/parsing/` | `proxide.io.parsing` | Replace imports |
| `src/prxteinmpnn/physics/force_fields.py` | `proxide.physics.force_fields` | Delete, use proxide |
| `src/prxteinmpnn/physics/electrostatics.py` | `oxidize.parse_structure` with `compute_electrostatics=True` | Delete |
| `src/prxteinmpnn/physics/vdw.py` | `oxidize.parse_structure` with `compute_vdw=True` | Delete |
| `src/prxteinmpnn/physics/features.py` | Keep as optional | Mark experimental |
| `src/prxteinmpnn/io/parsing/physics_utils.py` | `proxide.io.parsing.physics_utils` | Delete |
| `src/prxteinmpnn/utils/residue_constants.py` | `proxide.utils.residue_constants` | Verify, replace |

### 13.7 Integration Steps

1. **Add proxide dependency**:

   ```toml
   # pyproject.toml
   dependencies = [
       # ... existing ...
       "proxide @ git+https://github.com/maraxen/proxide.git",
   ]
   ```

2. **Update data preprocessing** to use `OutputSpec`:

   ```python
   from oxidize import OutputSpec, parse_structure
   
   spec = OutputSpec(
       compute_rbf=True,
       rbf_num_neighbors=30,
       compute_electrostatics=True,
       compute_vdw=True,
       parameterize_md=True,
       force_field="amber14-all",
   )
   
   result = parse_structure(pdb_path, spec)
   
   # Access features
   rbf_features = result["rbf_features"]           # (N_res, K, 400)
   electro_features = result["electrostatic_features"]  # (N_res, 5)
   vdw_features = result["vdw_features"]           # (N_res, 5)
   ```

3. **Update ProteinTuple/data structures** to include physics features:

   ```python
   @dataclass
   class ProteinTuple:
       # ... existing fields ...
       electrostatic_features: np.ndarray | None = None  # (N_res, 5)
       vdw_features: np.ndarray | None = None            # (N_res, 5)
       rbf_features: np.ndarray | None = None            # (N_res, K, 400)
   ```

4. **Update model encoder** to consume precomputed features (if using RBF):

   ```python
   # Model changes to accept rbf_features input
   # This may require architecture changes
   ```

5. **Delete deprecated modules** after migration

---

## 14. Prolix Integration

### 14.1 Clarification: Proxide vs Prolix for Physics

**For PrxteinMPNN specifically:**

- **Use proxide** for electrostatic/vdW features (computed in Rust during parsing)
- **Prolix is NOT required** for PrxteinMPNN's physics features

Prolix is the MD simulation library. It provides runtime physics calculations for:

- Actual molecular dynamics simulations
- Energy minimization
- Explicit solvation

PrxteinMPNN does not perform MD simulations - it only needs physics *features* for the MPNN encoder, which proxide now provides.

### 14.2 When Prolix IS Needed

Prolix would only be needed if PrxteinMPNN:

- Runs energy minimization as part of design
- Performs MD refinement of generated sequences
- Uses differentiable physics (which proxide doesn't provide)

**Current recommendation:** Do NOT add prolix as a dependency unless you need MD capabilities.

### 14.3 What Prolix Provides (Reference)

For future reference, if MD is needed:

- **Electrostatics**: Coulomb forces, PME
- **Van der Waals**: Lennard-Jones energy/forces
- **GBSA**: Implicit solvent
- **Integrators**: Langevin, Velocity Verlet
- **Minimization**: L-BFGS, steepest descent
- **Constraints**: SETTLE, SHAKE

### 14.4 Updated Dependency Strategy

```toml
# pyproject.toml

dependencies = [
    # Required for structure parsing and physics features
    "proxide @ git+https://github.com/maraxen/proxide.git",
    
    # NOT REQUIRED - only if MD simulation is needed
    # "prolix @ git+https://github.com/maraxen/prolix.git",
]

[project.optional-dependencies]
md = [
    # Optional: for MD refinement workflows
    "prolix @ git+https://github.com/maraxen/prolix.git",
]
```

### 14.5 Removing jax_md Dependency

Since proxide handles all physics features via Rust, we can remove `jax_md`:

```diff
# pyproject.toml
dependencies = [
    # ...
-   "jax-md>=0.2.26",
    # ...
]
```

---

## 15. Deprecation & Removal

### 15.1 Files to Delete After Migration

```
src/prxteinmpnn/
├── io/
│   └── parsing/
│       ├── biotite.py          # → proxide
│       ├── coords.py           # → proxide  
│       ├── mappings.py         # → proxide
│       ├── mdcath.py           # → proxide
│       ├── mdtraj.py           # → proxide
│       ├── physics_utils.py    # → proxide
│       ├── pqr.py              # → proxide
│       ├── structures.py       # → proxide
│       └── utils.py            # → proxide
└── physics/
    ├── constants.py            # → prolix
    ├── electrostatics.py       # → prolix
    ├── force_fields.py         # → proxide
    ├── projections.py          # → prolix (or keep)
    └── vdw.py                  # → prolix
```

### 15.2 Files to Keep

```
src/prxteinmpnn/
├── io/
│   ├── __init__.py             # Update exports
│   ├── array_record_source.py  # PrxteinMPNN-specific
│   ├── dataset.py              # PrxteinMPNN-specific  
│   ├── loaders.py              # PrxteinMPNN-specific
│   ├── operations.py           # PrxteinMPNN-specific
│   ├── prefetch_autotune.py    # PrxteinMPNN-specific
│   ├── process.py              # PrxteinMPNN-specific
│   ├── weights.py              # Model weight loading
│   └── parsing/
│       ├── __init__.py         # Update to re-export from proxide
│       └── dispatch.py         # Keep as facade, update imports
└── physics/
    ├── __init__.py             # Update exports
    └── features.py             # KEEP - PrxteinMPNN-specific features
```

### 15.3 Migration Verification Checklist

- [ ] All tests pass after proxide import updates
- [ ] All tests pass after prolix import updates
- [ ] `physics/features.py` still works with new imports
- [ ] Training pipeline completes successfully
- [ ] Inference pipeline completes successfully
- [ ] Force field loading works via proxide
- [ ] jax_md removed from dependencies
- [ ] No remaining `from prxteinmpnn.physics.electrostatics` imports
- [ ] No remaining `from prxteinmpnn.physics.vdw` imports

---

## 16. Development Workflow with Submodules

### 16.1 Purpose

When actively developing features that span proxide, prolix, and PrxteinMPNN, use git submodules for editable installs. This enables synchronized development across all three repositories.

**⚠️ IMPORTANT:** The main branch should NOT contain `.gitmodules` or uv source overrides. These are for development branches only.

### 16.2 Development Branch Setup

```bash
# Create development branch
git checkout -b dev/feature-name

# Add submodules (creates .gitmodules)
git submodule add https://github.com/maraxen/proxide.git external/proxide
git submodule add https://github.com/maraxen/prolix.git external/prolix

# Initialize and update
git submodule update --init --recursive
```

### 16.3 Configure uv for Editable Installs

Create/update `pyproject.toml` for development with local sources:

```toml
# Add to pyproject.toml (DEVELOPMENT ONLY - do not merge to main)

[tool.uv.sources]
proxide = { path = "external/proxide", editable = true }
prolix = { path = "external/prolix", editable = true }
```

Then sync:

```bash
uv sync
```

### 16.4 Development Workflow

```bash
# 1. Make changes in proxide submodule
cd external/proxide
git checkout -b feature/my-change
# ... make changes ...
git add -A && git commit -m "feat: my change"
git push origin feature/my-change
cd ../..

# 2. Make changes in prolix submodule  
cd external/prolix
git checkout -b feature/my-change
# ... make changes ...
git add -A && git commit -m "feat: my change"
git push origin feature/my-change
cd ../..

# 3. Make corresponding changes in PrxteinMPNN
# ... make changes ...
git add -A && git commit -m "feat: integrate my-change"

# 4. Test everything together
uv run pytest tests/ -v
```

### 16.5 Before Merging to Main

**Remove submodule configuration:**

```bash
# Remove submodules
git submodule deinit -f external/proxide
git submodule deinit -f external/prolix
rm -rf .git/modules/external
git rm -rf external/

# Remove uv source overrides from pyproject.toml
# (manual edit - remove [tool.uv.sources] section)

# Update dependencies to use released versions
# pyproject.toml should have:
dependencies = [
    "proxide @ git+https://github.com/maraxen/proxide.git@main",
    "prolix @ git+https://github.com/maraxen/prolix.git@main",
]

# Commit cleanup
git add -A
git commit -m "chore: remove development submodules for main merge"
```

### 16.6 Cloning with Submodules (for other developers)

If checking out a development branch with submodules:

```bash
git clone --recurse-submodules https://github.com/maraxen/PrxteinMPNN.git
cd PrxteinMPNN
git checkout dev/feature-name
git submodule update --init --recursive
uv sync
```

---

## 17. Known Issues & Future Work

### 17.1 Precision Casting (Experimental)

**Status:** The precision casting approach works for training but checkpoint loading needs stabilization.

**Issues:**

- Loading float32 weights into bf16 model requires explicit dtype conversion
- Optimizer state dtype must match model parameter dtype

**TODO:**

- [ ] Add comprehensive tests for checkpoint → resume flow
- [ ] Ensure weights are ALWAYS saved in float32 for portability
- [ ] Test bf16 ↔ fp32 round-trip

### 17.2 Resource Allocation

**TODO:**

- [ ] Add `psutil` to dependencies if not present
- [ ] Test auto-detection on various hardware (CPU, GPU, Grace Hopper)
- [ ] Document resource strategy in user guide

### 17.3 Gradient Accumulation

**TODO:**

- [ ] Verify gradient scaling is correct with accumulation
- [ ] Test with different `accum_steps` values
- [ ] Document effective batch size calculation

### 17.4 Proxide/Prolix Migration

**TODO:**

- [x] Complete proxide integration (parsing, force fields)
- [x] Complete prolix integration (electrostatics, vdW)
- [x] Update physics/features.py to use new imports
- [x] **Equivalence Validation (CRITICAL)**:
  - [x] Create parity test script comparing JAX-computed vs Proxide-computed features on standard structures. (Verified 0.0 numerical difference on finite values)
  - [x] Verify RBF equivalence (Kept JAX implementation in model to support gradients).
  - [x] Verify electrostatic/vdW feature ranges match old implementation.
- [x] Resolve dimension mismatch in model encoder (handling 5/10/400-dim feature vectors).
- [x] Fix tests/training/dataloading/test_preprocess.py (Verified serial pass, parallel worker mocks updated).
- [x] Final removal of deprecated local modules (electrostatics.py, vdw.py, projections.py, etc.).
- [x] Remove jax_md dependency after ensuring no loss of functionality.
- [x] Update all tests to use new imports.
- [x] Add integration tests for proxide/prolix ecosystem.

---

## Merge Execution Checklist

### Part A: Training Branch

- [ ] Create feature branch
- [ ] Replace `tables` with `h5py` in dependencies
- [ ] Add resource allocation fields to `RunSpecification`
- [ ] Add `accum_steps` to `TrainingSpecification`
- [ ] Update trainer with precision handling
- [ ] Update model files (preserve docstrings!)
- [ ] Update operations with NumPy backend (preserve docstrings!)
- [ ] Update data_structures (preserve docstrings!)
- [ ] Move data processing scripts to `scripts/data_processing/`
- [ ] Update checkpoint module
- [ ] Make grain debug mode environment-configurable
- [ ] Run tests
- [ ] Run linting
- [ ] Update documentation
- [ ] Create PR for review

### Part B: Proxide/Prolix Migration

- [ ] Add proxide dependency
- [ ] Add prolix dependency
- [ ] Update parsing imports to use proxide
- [ ] Update force field imports to use proxide
- [ ] Update physics imports to use prolix
- [ ] Update `physics/features.py` to use new imports
- [ ] Delete deprecated modules
- [ ] Remove jax_md dependency
- [ ] Update all tests
- [ ] Verify training pipeline
- [ ] Verify inference pipeline
- [ ] Create PR for review

---

## Quick Reference: Git Commands

```bash
# Start merge
git checkout main
git checkout -b merge/training-integration

# Cherry-pick with manual conflict resolution (recommended)
git cherry-pick cf026118  # initial data prep
git cherry-pick 57d98259  # training fixes
git cherry-pick 4262b133  # training setup
git cherry-pick 8dbfc84f  # grad accum
git cherry-pick fcc6fb58  # fp precision

# Or full merge (more conflicts)
git merge origin/training

# View conflicts
git diff --name-only --diff-filter=U

# After resolving
git add <resolved-files>
git commit
```
