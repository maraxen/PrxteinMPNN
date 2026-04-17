# Refactoring Milestone: Internalized PyPI Weights & Smart Factory

## Summary

Successfully completed the transition from external HuggingFace Hub weight hosting to **internal package bundling** using Zstandard compression. This pivot ensures zero-dependency offline loading, eliminates network concurrency bugs, and introduces a Smart Factory loader that automatically detects model topology from bundled assets.

## What Was Accomplished

### 1. Internal Weight Bundling ✅

- **Compression**: All `.eqx` model weights are now compressed using Zstandard (level 19) to `.eqx.zst`.
- **Packaging**: Bundled weights directly into `src/prxteinmpnn/model_params/`.
- **Resource Loading**: Implemented memory-efficient streaming access using `importlib.resources`.

### 2. Smart Factory Loader ✅

- Refactored `src/prxteinmpnn/io/weights.py` into a unified `load_model()` dispatcher.
- Automatic detection of:
    - **Model Types**: Standard PrxteinMPNN, LigandMPNN (`v_32`), and Packer (`_sc_`).
    - **Topology**: Infers `k_neighbors`, `atom_context_num`, and `num_positional_embeddings` from checkpoint filenames.
    - **Physics**: Detects Membrane/Soluble specific parameters (e.g., `physics_feature_dim=3`).

### 3. Systematic Runtime Decoupling ✅

- Stripped out all deprecated `_k_neighbors=48` hardcoded constants from `src/prxteinmpnn/run/` and downstream pipelines.
- Recalculated `jax.vmap` positional `in_axes` mapping to account for removed function arguments, ensuring stable vectorization across different model architectures.

### 4. Infrastructure Cleanup ✅

- Deleted obsolete HuggingFace automation:
    - `scripts/push_all_to_hub.py`
    - `scripts/fetch_test_data.py`
- Updated `pyproject.toml`:
    - Removed `huggingface-hub`.
    - Added `zstandard`.
    - Included `model_params/*.eqx.zst` in package distribution.

## Key Features

### Unified Loading API

```python
from prxteinmpnn.io.weights import load_model

# Load Standard MPNN
model = load_model("proteinmpnn_v_48_020")

# Load Ligand-aware MPNN (automatically detects v_32 topology)
ligand_model = load_model("ligandmpnn_v_32_010_25")

# Load Side-chain Packer
packer = load_model("ligandmpnn_sc_v_32_002_16")
```

### Zero Runtime Network Dependency

- All assets are included in the `.whl` or `.tar.gz` distribution.
- No more `hf_hub_download` latency or file-locking crashes in multi-process environments.

## Testing Results

- Verified all 3 skeleton types (Standard, Ligand, Packer) instantiate and load bit-perfect from bundled Zstd streams.
- Confirmed `jax.vmap` pipelines run end-to-end without shape mismatches.
- All offline tests in `tests/io/test_weights.py` are passing.

---

**Date**: April 17, 2026
**Status**: ✅ Complete
**Migration Path**: Remote (HF) ➡️ Internal (PyPI Bundling)
