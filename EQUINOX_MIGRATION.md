# PrxteinMPNN Equinox Migration Plan (Side-by-Side Equivalence)

This plan implements a robust, side-by-side migration. We will first isolate the entire current functional API into a new `prxteinmpnn.functional` submodule. We will then build the new `prxteinmpnn.eqx` API piece-by-piece, validating each new component against its functional counterpart at every step.

## Milestone 1: Isolate the Functional "Legacy" API

**Objective:** Move all existing functional code into a new `prxteinmpnn.functional` submodule and ensure the entire existing test suite passes against this new, namespaced API. This provides a stable baseline for the refactor.

### Milestone 1 Actions

1. **Create functional module:**
   - Create `src/prxteinmpnn/functional/` directory as a proper Python package.
   - Create `src/prxteinmpnn/functional/__init__.py` that exports the main API.

2. **Migrate All Old Code into Modular Structure:**
   - Create `src/prxteinmpnn/functional/normalize.py`: Copy `layer_normalization` and `normalize` from `src/prxteinmpnn/utils/normalize.py`.
   - Create `src/prxteinmpnn/functional/dense.py`: Copy `dense_layer` from `src/prxteinmpnn/model/dense.py`.
   - Create `src/prxteinmpnn/functional/encoder.py`: Copy all encoder functions from `src/prxteinmpnn/model/encoder.py` (e.g., `encoder_parameter_pytree`, `encode_layer_fn`, `make_encoder`, etc.).
   - Create `src/prxteinmpnn/functional/decoder.py`: Copy all decoder functions from `src/prxteinmpnn/model/decoder.py` (e.g., `decoder_parameter_pytree`, `decode_message_fn`, `make_decoder`, etc.).
   - Create `src/prxteinmpnn/functional/features.py`: Copy feature extraction functions from `src/prxteinmpnn/model/features.py` (e.g., `embed_edges`, `encode_positions`, `extract_features`, etc.).
   - Create `src/prxteinmpnn/functional/projection.py`: Copy `final_projection` from `src/prxteinmpnn/model/projection.py`.
   - Create `src/prxteinmpnn/functional/model.py`: Copy `get_mpnn_model` from `src/prxteinmpnn/mpnn.py` and rename to `get_functional_model`.

3. **Update All Existing Tests:**
   - Go through the entire `tests/` directory.
   - Any test that imports from `prxteinmpnn.mpnn`, `prxteinmpnn.model.encoder`, etc., must be updated to import from `prxteinmpnn.functional`.
   - For example, in `tests/scoring/test_score.py`, the import for `get_mpnn_model` will change to `from prxteinmpnn.functional import get_functional_model as get_mpnn_model`.

4. **Validate:**
   - Run the complete test suite. All existing tests must pass.

**Deliverable:** A stable `src/prxteinmpnn/functional.py` module that contains a snapshot of the entire old API. The `tests/` suite is fully functional and validates this baseline. The original files (`mpnn.py`, `model/encoder.py`, etc.) are now ready to be refactored.

---

## Milestone 2: Foundational eqx Layers & Conversion Helpers

**Objective:** Create the foundational eqx modules (`LayerNorm`, `DenseLayer`) and the helper functions to convert old weights to these new formats.

### Milestone 2 Actions

1. **Create New Files:**
   - Create `src/prxteinmpnn/eqx.py` (this will house all new `eqx.Module` classes).
   - Create `src/prxteinmpnn/conversion.py` (this will house weight conversion helpers).
   - Create a new, permanent test file: `tests/test_eqx_equivalence.py`.

2. **Create Conversion Helpers:**
   - In `src/prxteinmpnn/conversion.py`, create:
     - `create_linear(w, b) -> eqx.nn.Linear`
     - `create_eqx_layernorm(scale, offset) -> E.LayerNorm`
     - `create_eqx_dense(p_dict, prefix) -> E.DenseLayer`

3. **Define eqx Modules:**
   - In `src/prxteinmpnn/eqx.py`, define:
     - `class LayerNorm(eqx.Module)` (as planned in the original doc).
     - `class DenseLayer(eqx.Module)` (as planned in the original doc).
   - In `src/prxteinmpnn/utils/normalize.py` and `src/prxteinmpnn/model/dense.py`, delete the (now redundant) old functions.

4. **Write Equivalence Tests:**
   - In `tests/test_eqx_equivalence.py`:
     - Import `prxteinmpnn.functional as F`.
     - Import `prxteinmpnn.eqx as E`.
     - Import `prxteinmpnn.conversion as C`.
     - Load a `.pkl` file.
     - `test_layernorm_equivalence`: Call `F.layer_normalization` to get `old_output`. Use `C.create_eqx_layernorm` to instantiate `E.LayerNorm` and call it to get `new_output`. Assert `jnp.allclose`.
     - `test_dense_equivalence`: Call `F.dense_layer` to get `old_output`. Use `C.create_eqx_dense` to instantiate `E.DenseLayer` and call it to get `new_output`. Assert `jnp.allclose`.

**Deliverable:** `eqx.py` and `conversion.py` exist. `test_eqx_equivalence.py` runs and passes for `LayerNorm` and `DenseLayer`, proving numerical equivalence for our foundational blocks.

---

## Milestone 3: Core Model Layers & Refactored Kernels

**Objective:** Build `EncoderLayer` and `DecoderLayer` and refactor the JIT-compiled kernels (`encode_layer_fn`, etc.) to accept these new modules.

### Milestone 3 Actions

1. **Define eqx Skeletons:**
   - In `src/prxteinmpnn/eqx.py`, define `class EncoderLayer(eqx.Module)` and `class DecoderLayer(eqx.Module)`, composing them from `E.LayerNorm` and `E.DenseLayer`.

2. **Update Conversion Helpers:**
   - In `src/prxteinmpnn/conversion.py`, add helpers to populate `EncoderLayer` and `DecoderLayer` from a param dict.

3. **Refactor Functional Kernels:**
   - In `src/prxteinmpnn/model/encoder.py`:
     - Refactor `encode_layer_fn(params, ...)` to `encode_layer_fn(layer: E.EncoderLayer, ...)`.
     - Delete `encoder_parameter_pytree`.
   - In `src/prxteinmpnn/model/decoder.py`:
     - Refactor `decode_message_fn`, etc., to accept `layer: E.DecoderLayer`.
     - Delete `decoder_parameter_pytree`.

4. **Write Equivalence Tests:**
   - In `tests/test_eqx_equivalence.py`:
     - `test_encoder_layer_equivalence`: Call `F.encode_layer_fn` (from the legacy API) to get `old_output`. Create a new `E.EncoderLayer`, populate it, and pass it to the new `encode_layer_fn` (from `model.encoder`) to get `new_output`. Assert `jnp.allclose`.
     - Repeat for `test_decoder_layer_equivalence`.

**Deliverable:** `eqx.py` now contains all layer modules. `model/encoder.py` and `model/decoder.py` are refactored. `test_eqx_equivalence.py` now has passing tests for all core layers.

---

## Milestone 4: Full eqx Model, API, & Weight Conversion

**Objective:** Assemble the final `PrxteinMPNN` module, refactor the main `mpnn.py` API, and create the production script to convert all `.pkl` files.

### Milestone 4 Actions

1. **Refactor Remaining Kernels:**
   - In `src/prxteinmpnn/model/features.py` and `src/prxteinmpnn/model/projection.py`, refactor the kernels to accept eqx modules (e.g., `eqx.nn.Linear`, `eqx.nn.Embedding`).

2. **Define Top-Level Model:**
   - In `src/prxteinmpnn/eqx.py`, define the final `class PrxteinMPNN(eqx.Module)`.
   - Implement its `encode`, `decode`, and `__call__` methods, which use the refactored kernels (e.g., `jax.lax.scan` over `encode_layer_fn`).

3. **Refactor Main API:**
   - In `src/prxteinmpnn/mpnn.py`, delete the (now empty/old) `make_...` functions.
   - Refactor `get_mpnn_model` to:
     - Import `PrxteinMPNN` from `prxteinmpnn.eqx`.
     - Instantiate an empty `PrxteinMPNN` with the correct hyperparameters.
     - Load the corresponding `.eqx` file path.
     - Return `eqx.tree_deserialise_leaves(model_path, model, ...)`.

4. **Create Conversion Script:**
   - Create `scripts/convert_weights.py`.
   - This script imports from `prxteinmpnn.eqx` and `prxteinmpnn.conversion`.
   - It will load `.pkl` files, instantiate `PrxteinMPNN`, populate it using the conversion helpers, and `eqx.tree_serialise_leaves` to save the new `.eqx` file.
   - Run this script to generate all `.eqx` files.

5. **Write Full Model Equivalence Test:**
   - In `tests/test_eqx_equivalence.py`:
     - `test_full_model_equivalence`:
       - `old_model_fns = F.get_functional_model("v_48_020")`.
       - Run the full old model to get `logits_old`.
       - `new_model = get_mpnn_model("v_48_020")` (from the new `mpnn.py`).
       - Run `new_model(features)` to get `logits_new`.
       - Assert `jnp.allclose(logits_old, logits_new)`.

**Deliverable:** A passing `test_full_model_equivalence`. The new eqx API is functionally complete and verified. All `.eqx` weight files are generated.

---

## Milestone 5: Final API Switch & Cleanup

**Objective:** Update the user-facing scoring and sampling modules to use the new eqx model object and clean up all legacy code.

### Milestone 5 Actions

1. **Update Call Sites:**
   - `src/prxteinmpnn/scoring/score.py`: Update `make_score_fn` to import `get_mpnn_model` from `prxteinmpnn.mpnn` and expect the `PrxteinMPNN` object. Change the internal logic to `logits = mpnn_model(features)`.
   - `src/prxteinmpnn/sampling/sample.py`: Update `make_sample_fn` similarly, refactoring `sampling_step` to use `mpnn_model.encode` and `mpnn_model.decode`.

2. **Update Final Tests:**
   - Update `tests/scoring/test_score.py`, `tests/sampling/test_sample.py`, and `tests/test_mpnn.py` to use and validate the new API.

3. **Validate:**
   - Run the entire test suite. All tests (equivalence, scoring, sampling) must pass.

4. **Final Cleanup:**
   - Delete `src/prxteinmpnn/functional.py`.
   - Delete `tests/test_eqx_equivalence.py`.
   - Delete `src/prxteinmpnn/conversion.py`. (Optional: Keep the conversion script in `scripts/` for future use).

**Deliverable:** A fully migrated, cleaned, and tested codebase. The functional submodule and equivalence tests are removed, as the main test suite now fully validates the eqx implementation.

---

## Progress Tracking

- [x] Milestone 1: Isolate the Functional "Legacy" API ✅ **COMPLETE**
  - [x] Created `src/prxteinmpnn/functional/` module structure
  - [x] Created `normalize.py`, `dense.py`, `projection.py`, `features.py`, `encoder.py`, `decoder.py`, `model.py`
  - [x] Created `__init__.py` with complete API exports
  - [x] Created `tests/functional/` directory for functional tests
  - [x] Moved and updated all relevant tests to use `prxteinmpnn.functional` imports
  - [x] All 41 functional tests pass successfully
- [ ] Milestone 2: Foundational eqx Layers & Conversion Helpers
- [ ] Milestone 3: Core Model Layers & Refactored Kernels
- [ ] Milestone 4: Full eqx Model, API, & Weight Conversion
- [ ] Milestone 5: Final API Switch & Cleanup
